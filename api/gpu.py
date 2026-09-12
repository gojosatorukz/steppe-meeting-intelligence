"""CUDA enablement and the VRAM phase handoff between STT and the LLM.

Two jobs, both of which have to happen before anything else touches the card.

1. **Make CUDA loadable.** CTranslate2 resolves `libcublas.so.12` / `libcudnn.so.9`
   with a plain `dlopen(soname)` the first time it runs an encoder pass — not at
   model construction. The pip wheels (`nvidia-cublas-cu12`, `nvidia-cudnn-cu12`)
   drop those files inside site-packages, where the dynamic loader has no reason
   to look, so that `dlopen` fails with

       RuntimeError: Library libcublas.so.12 is not found or cannot be loaded

   *even though the packages are installed* — the symptom that made CUDA look
   broken on this box. The fix is to load them ourselves, by absolute path and
   with `RTLD_GLOBAL`, before `ctranslate2` is imported: a library already in the
   process satisfies a later `dlopen` of the same soname, so CTranslate2's lookup
   then succeeds with no `LD_LIBRARY_PATH` and no re-exec. `uv run uvicorn
   main:app` just works, which is the requirement.

   This module therefore MUST be imported before `faster_whisper`/`ctranslate2`
   — `pipeline` does that on its first line.

2. **Referee the 5.7 GiB card.** Two STT models (~1 GiB resident) plus the 4B LLM
   (3.4 GiB + KV cache) do not fit together. So the GPU is handed back and forth:
   STT owns it while transcribing, and `structure()` borrows it for the LLM phase
   only. `llm_phase()` is that handoff — release STT, run the chat calls, unload
   the LLM, *wait for Ollama to actually let go of the VRAM*, re-acquire STT —
   under one lock so an upload and a live utterance can never interleave phases.
   If STT still cannot get back onto the card, the module latches a
   `degraded_reason()` rather than reporting a healthy GPU it does not have, and
   retries on the next `phase_lock()`.

Lock ordering, project-wide: `gpu._gpu_lock` is the OUTER lock and
`pipeline._model_lock` the inner one. Anything that loads a model onto the card
takes them in that order; nothing may take them the other way round.

On the CPU path every function here is a no-op (see `set_enabled`), so the CPU
fallback keeps exactly the behaviour it had before the GPU existed.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import logging
import os
import threading
import time
from collections.abc import Iterator
from typing import Any

import ollama

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# 1. CUDA library preload
# --------------------------------------------------------------------------

# CTranslate2 4.8.x is built against CUDA 12 and cuDNN 9. Loading the umbrella
# sonames is enough in the normal case: each wheel's `.so` carries an $ORIGIN
# RUNPATH, so `libcublasLt.so.12` and the `libcudnn_*.so.9` back-ends resolve as
# siblings of the file we name here.
_CUBLAS = ("nvidia.cublas.lib", "libcublas.so.12")
_CUDNN = ("nvidia.cudnn.lib", "libcudnn.so.9")

# ...but if that $ORIGIN resolution ever fails (a relocated wheel, a stripped
# RUNPATH), cuDNN 9's split back-ends have to be pulled in by hand, and they
# depend on each other in this order.
_CUDNN_SIBLINGS = (
    "libcudnn_graph.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
)

#: Absolute paths of the shared objects this process pulled in, in load order.
preloaded: list[str] = []
#: Human-readable reasons a preload did not happen. Empty means all good.
preload_errors: list[str] = []


def _package_dir(module: str) -> str | None:
    """Directory of a namespace package like `nvidia.cublas.lib`.

    These wheels ship no `__init__.py`, so `__file__` is None and only the
    namespace package's search path names the directory.
    """
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):  # pragma: no cover - env dependent
        return None
    if spec is None:
        return None
    for location in spec.submodule_search_locations or []:
        return str(location)
    return os.path.dirname(spec.origin) if spec.origin else None


def _load(path: str) -> bool:
    """`dlopen` one library into the global namespace. Never raises."""
    try:
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        preload_errors.append(f"{path}: {exc}")
        return False
    preloaded.append(path)
    return True


def _preload_one(module: str, soname: str) -> bool:
    directory = _package_dir(module)
    if directory is None:
        preload_errors.append(f"{module} is not installed (needed for {soname})")
        return False
    path = os.path.join(directory, soname)
    if not os.path.isfile(path):
        preload_errors.append(f"{path} does not exist")
        return False
    if _load(path):
        return True
    if soname != _CUDNN[1]:
        return False
    # cuDNN only: the umbrella failed to pull its back-ends in, so name them.
    logger.warning("preloading cuDNN back-ends explicitly after %s failed", path)
    for sibling in _CUDNN_SIBLINGS:
        sibling_path = os.path.join(directory, sibling)
        if os.path.isfile(sibling_path):
            _load(sibling_path)
    return _load(path)


def preload_cuda_libraries() -> bool:
    """Pull cuBLAS and cuDNN into this process. Idempotent; never raises.

    Returns True when both are resident, which is what makes a later
    CTranslate2 CUDA encoder pass possible.
    """
    if preloaded and not preload_errors:
        return True
    del preloaded[:], preload_errors[:]
    ok = _preload_one(*_CUBLAS)
    ok = _preload_one(*_CUDNN) and ok
    if ok:
        logger.debug("preloaded CUDA libraries: %s", ", ".join(preloaded))
    else:
        logger.info("CUDA libraries unavailable: %s", "; ".join(preload_errors))
    return ok


# Import-time, on purpose: everything below (and every importer of this module)
# depends on this having happened before ctranslate2 is loaded.
_LIBS_READY = preload_cuda_libraries()


def cuda_ready() -> tuple[bool, str]:
    """Can CTranslate2 actually run on the GPU here? (ready, reason)

    `get_cuda_device_count() > 0` alone is NOT enough — it reports 1 on this box
    even when the encoder pass later dies on a missing `libcublas.so.12`. So the
    preload has to have succeeded too; that is the condition the failure was
    really about.
    """
    if not _LIBS_READY:
        return False, "CUDA libraries could not be preloaded: " + "; ".join(preload_errors)
    try:
        import ctranslate2
    except Exception as exc:  # pragma: no cover - ctranslate2 is a hard dep
        return False, f"ctranslate2 is not importable: {exc}"
    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:  # pragma: no cover - driver dependent
        return False, f"get_cuda_device_count() failed: {exc}"
    if count < 1:
        return False, "no CUDA device visible to ctranslate2"
    return True, f"{count} CUDA device(s), cuBLAS + cuDNN preloaded"


# --------------------------------------------------------------------------
# 2. The phase manager
# --------------------------------------------------------------------------

# Outer lock (see the module docstring). RLock because the transcription path
# nests: transcribe_pcm() holds it across an utterance while _detect_language()
# and _transcribe_window() take it again inside.
_gpu_lock = threading.RLock()

# label -> WhisperModel, for models that actually built on the card. A model that
# fell back to the CPU is not in here: there is nothing of ours to hand over.
_stt_models: dict[str, Any] = {}

# Is the GPU path in play at all? Set once from the resolved STT device.
_enabled = False

# Set when STT could not be put back on the card. Sticky until a later acquire
# succeeds, and surfaced through /health: a service that silently keeps this to
# itself looks healthy while every transcription dies inside ctranslate2.
_degraded_reason: str | None = None

# How long to wait for Ollama to actually give the card back before re-allocating
# STT on top of it. Measured on this box: the `ps()` row disappears 0.19 s after
# the zero-keep-alive request, and `nvidia-smi` is already back to idle by then.
LLM_UNLOAD_TIMEOUT = 3.0
LLM_UNLOAD_POLL = 0.05
# A re-acquire that loses a race with a straggler allocation is worth one retry.
STT_ACQUIRE_RETRY_DELAY = 0.5


def set_enabled(enabled: bool) -> None:
    """Turn the GPU path on (STT resolved to cuda) or off (CPU path)."""
    global _enabled
    _enabled = bool(enabled)


def is_enabled() -> bool:
    """Whether the GPU path is in play — governs the lock."""
    return _enabled


def is_active() -> bool:
    """Whether a handoff is worth doing — i.e. something of ours is on the card.

    Distinct from `is_enabled()` on purpose: if every STT model ended up on the
    CPU after a failed CUDA build, there is no reason to evict the LLM from a
    card we are not using, so `keep_alive` and the handoff go back to their CPU
    behaviour while the lock (which still guards a future CUDA load) stays on.
    """
    return _enabled and bool(_stt_models)


def degraded_reason() -> str | None:
    """Why the GPU path is currently broken, or None. For /health."""
    return _degraded_reason


def _set_degraded(reason: str) -> None:
    global _degraded_reason
    _degraded_reason = reason


def clear_degraded() -> None:
    global _degraded_reason
    _degraded_reason = None


def register_stt_model(label: str, model: Any) -> None:
    """Remember a WhisperModel that lives on the GPU, so it can be moved off it."""
    _stt_models[label] = model


def unregister_stt_model(label: str) -> None:
    _stt_models.pop(label, None)


def forget_stt_models() -> None:
    """Drop every registered model (tests, and a full fallback to CPU)."""
    _stt_models.clear()


def registered_stt_models() -> tuple[Any, ...]:
    return tuple(_stt_models.values())


def registered_stt_entries() -> tuple[tuple[str, Any], ...]:
    return tuple(_stt_models.items())


def stt_release() -> int:
    """Move the STT weights off the GPU, keeping them in RAM.

    `to_cpu=True` is the whole point: the weights survive in host memory, so
    `stt_acquire()` is a ~0.3 s copy back rather than a 3 s reload from disk.
    Failures are logged, not raised — a model we cannot unload costs VRAM (the
    LLM then spills to RAM and runs slowly), but aborting the LLM phase over it
    would cost the protocol entirely.
    """
    if not is_active():
        return 0
    moved = 0
    for label, model in _stt_models.items():
        try:
            inner = model.model
            if getattr(inner, "model_is_loaded", True):
                inner.unload_model(to_cpu=True)
                moved += 1
        except Exception as exc:
            logger.warning("could not move STT model %s off the GPU: %s", label, exc)
    logger.debug("released %d STT model(s) from the GPU", moved)
    return moved


def stt_acquire() -> int:
    """Put the STT weights back on the GPU (from RAM).

    Raises if any model could not be restored — unlike the release, this one is
    not survivable by ignoring it: a model left unloaded fails inside ctranslate2
    on the next encoder pass, so the caller has to decide (retry, or degrade).
    """
    if not is_active():
        return 0
    restored, errors = 0, []
    for label, model in _stt_models.items():
        inner = model.model
        if getattr(inner, "model_is_loaded", False):
            continue
        try:
            inner.load_model()
        except Exception as exc:
            errors.append(f"{label}: {exc}")
        else:
            restored += 1
    if errors:
        raise RuntimeError("; ".join(errors))
    logger.debug("re-acquired %d STT model(s) onto the GPU", restored)
    return restored


def stt_acquire_or_degrade() -> int:
    """`stt_acquire()` with one retry, then a sticky degraded flag.

    The realistic failure is losing a race for VRAM against something that has
    not finished freeing it, so a short sleep and one more try is the fix that
    actually works. If it still fails, STT is genuinely off the card and the
    service must say so rather than answer /health with a cheerful "cuda".
    """
    try:
        restored = stt_acquire()
    except Exception as first:
        logger.warning(
            "STT could not be put back on the GPU (%s); retrying in %.1fs",
            first, STT_ACQUIRE_RETRY_DELAY,
        )
        time.sleep(STT_ACQUIRE_RETRY_DELAY)
        try:
            restored = stt_acquire()
        except Exception as second:
            reason = f"STT is off the GPU and would not load back: {second}"
            _set_degraded(reason)
            logger.error("%s — transcription will fail until this clears", reason)
            return 0
    clear_degraded()
    return restored


def unload_llm(model: str) -> bool:
    """Ask Ollama to drop `model` from VRAM right now.

    A zero keep-alive request is the documented way to do it; an empty message
    list means it costs no generation. Never raises: if Ollama is unreachable
    the protocol run is already over, and STT still has to come back.
    """
    if not is_active():
        return False
    try:
        ollama.chat(model=model, messages=[], keep_alive=0)
    except Exception as exc:
        logger.warning("could not unload LLM %s from the GPU: %s", model, exc)
        return False
    logger.debug("asked Ollama to unload %s", model)
    return True


def wait_for_llm_unloaded(model: str, timeout: float | None = None) -> bool:
    """Block until Ollama is no longer holding `model`, or the bound expires.

    `unload_llm()` returns as soon as Ollama *accepts* the request; the runner is
    torn down asynchronously. Allocating ~2 GiB of STT on top of an LLM that is
    still resident is how a 6 GiB card OOMs, so the phase waits for the `ps()`
    row to disappear before re-acquiring. Returns True if it went.
    """
    if timeout is None:                     # read at call time so it stays tunable
        timeout = LLM_UNLOAD_TIMEOUT
    deadline = time.monotonic() + timeout
    while True:
        try:
            rows = (ollama.ps() or {}).get("models") or []
        except Exception as exc:
            # Cannot tell. Better to fall through to the acquire (which retries)
            # than to block the phase for the full timeout on a dead Ollama.
            logger.warning("could not check whether %s left the GPU: %s", model, exc)
            return False
        if not any((row.get("model") or row.get("name")) == model for row in rows):
            return True
        if time.monotonic() >= deadline:
            logger.warning(
                "%s was still resident %.1fs after the unload request", model, timeout
            )
            return False
        time.sleep(LLM_UNLOAD_POLL)


@contextlib.contextmanager
def phase_lock() -> Iterator[None]:
    """Hold the GPU for one piece of work (a window, an utterance, a load).

    A no-op on the CPU path, where nothing is contended and serialising uploads
    against each other would be a pure regression.

    Doubles as the recovery point for a degraded card: if a previous handoff
    could not get STT back, try again here, now that whatever took the VRAM has
    had time to release it. Costs two attribute reads when nothing is wrong.
    """
    if not _enabled:
        yield
        return
    with _gpu_lock:
        if _degraded_reason is not None:
            logger.info("retrying the STT re-acquire before using the GPU")
            stt_acquire_or_degrade()
        yield


@contextlib.contextmanager
def llm_phase(model: str) -> Iterator[None]:
    """Lend the GPU to the LLM for the duration of the block.

    release STT -> (caller's chat calls) -> unload LLM -> *wait for it to be
    gone* -> re-acquire STT, with everything after the body in a `finally` so a
    failed protocol never leaves STT stranded off the card.
    """
    if not _enabled:
        yield
        return
    with _gpu_lock:
        handing_over = is_active()
        if handing_over:
            stt_release()
        try:
            yield
        finally:
            if handing_over:
                unload_llm(model)
                wait_for_llm_unloaded(model)
                stt_acquire_or_degrade()
