"""Tests for gpu.py — the CUDA preload and the STT <-> LLM VRAM handoff.

Two layers:

* Weight-free tests (always run) drive the phase manager with stub models and a
  stub ollama. They are the ones that pin the *order* of the handoff, the wait
  for Ollama to actually release the card, and the guarantee that a failure is
  either retried or reported — never silently left as "STT is off the GPU but
  /health says cuda", which is the state where every later transcription dies
  inside ctranslate2.
* GPU-gated tests (skipped without CUDA) prove the thing actually works: a real
  CUDA transcription inside the time budget, and the card genuinely returned to
  STT after a real `structure()` call.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

import gpu
import pipeline

AUDIO_DIR = Path(__file__).parent / "audio"
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------

class StubInner:
    """Stands in for ctranslate2's Whisper object (`WhisperModel.model`)."""

    def __init__(self, events: list[str], name: str, device: str = "cuda") -> None:
        self._events = events
        self._name = name
        self.device = device
        self.model_is_loaded = True
        self.unloads = 0
        self.loads = 0
        #: number of upcoming load_model() calls that should raise
        self.load_failures = 0

    def unload_model(self, to_cpu: bool = False) -> None:
        assert to_cpu, "weights must go to RAM, not be thrown away"
        self.unloads += 1
        self.model_is_loaded = False
        self._events.append(f"release:{self._name}")

    def load_model(self) -> None:
        if self.load_failures > 0:
            self.load_failures -= 1
            self._events.append(f"acquire-failed:{self._name}")
            raise RuntimeError("CUDA failed with error out of memory")
        self.loads += 1
        self.model_is_loaded = True
        self._events.append(f"acquire:{self._name}")


class StubModel:
    """Stands in for a faster-whisper WhisperModel."""

    def __init__(self, events: list[str], name: str, device: str = "cuda") -> None:
        self.model = StubInner(events, name, device)


class StubOllama:
    """Stands in for the ollama client: chat() unloads, ps() reports residency."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.calls: list[dict] = []
        self.raises: Exception | None = None
        #: how many ps() polls still report the model as resident
        self.busy_polls = 0
        self.ps_calls = 0
        self.ps_raises: Exception | None = None
        self.resident: str | None = None

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        self._events.append("unload-llm")
        self.resident = kwargs.get("model")
        return {"message": {"content": "{}"}}

    def ps(self):
        self.ps_calls += 1
        if self.ps_raises is not None:
            raise self.ps_raises
        if self.busy_polls > 0:
            self.busy_polls -= 1
            self._events.append("ps-busy")
            return {"models": [{"model": self.resident, "name": self.resident}]}
        self._events.append("ps-clear")
        return {"models": []}


@pytest.fixture
def phase(monkeypatch):
    """A live phase manager with two stub STT models and a stub ollama."""
    events: list[str] = []
    models = {"turbo": StubModel(events, "turbo"), "kk": StubModel(events, "kk")}

    # Any real models already on the card have to survive this test: wiping them
    # would silently disarm the handoff for every later test in the session.
    real_models = gpu.registered_stt_entries()
    was_enabled = gpu.is_enabled()
    real_degraded = gpu.degraded_reason()

    gpu.forget_stt_models()
    gpu.clear_degraded()
    for label, model in models.items():
        gpu.register_stt_model(label, model)
    gpu.set_enabled(True)

    stub_ollama = StubOllama(events)
    monkeypatch.setattr(gpu, "ollama", stub_ollama)
    # Keep the tests fast; the production values are 3.0 s / 0.05 s.
    monkeypatch.setattr(gpu, "LLM_UNLOAD_TIMEOUT", 0.30)
    monkeypatch.setattr(gpu, "LLM_UNLOAD_POLL", 0.01)
    monkeypatch.setattr(gpu, "STT_ACQUIRE_RETRY_DELAY", 0.01)

    yield events, models, stub_ollama

    gpu.forget_stt_models()
    for label, model in real_models:
        gpu.register_stt_model(label, model)
    gpu.set_enabled(was_enabled)
    if real_degraded is None:
        gpu.clear_degraded()


# --------------------------------------------------------------------------
# Phase-manager order (weight-free, always run)
# --------------------------------------------------------------------------

def test_llm_phase_releases_unloads_and_reacquires_in_order(phase):
    events, models, stub_ollama = phase

    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert events == [
        "release:turbo", "release:kk",
        "chat",
        "unload-llm",
        "ps-clear",
        "acquire:turbo", "acquire:kk",
    ], events
    for model in models.values():
        assert model.model.unloads == 1
        assert model.model.loads == 1
        assert model.model.model_is_loaded

    # The unload must be the zero-keep-alive form, and must not generate.
    assert len(stub_ollama.calls) == 1
    call = stub_ollama.calls[0]
    assert call["model"] == "stub-model"
    assert call["keep_alive"] == 0
    assert call["messages"] == []
    assert gpu.degraded_reason() is None


def test_llm_phase_reacquires_stt_when_the_body_raises(phase):
    events, models, _ = phase

    with pytest.raises(RuntimeError, match="ollama exploded"):
        with gpu.llm_phase("stub-model"):
            events.append("chat")
            raise RuntimeError("ollama exploded")

    assert events == [
        "release:turbo", "release:kk", "chat", "unload-llm", "ps-clear",
        "acquire:turbo", "acquire:kk",
    ], events
    for model in models.values():
        assert model.model.model_is_loaded, "STT must never be left off the GPU"


def test_llm_phase_survives_an_unreachable_ollama(phase):
    """The explicit unload is best-effort: STT still has to come back."""
    events, models, stub_ollama = phase
    stub_ollama.raises = ConnectionError("connection refused")
    stub_ollama.ps_raises = ConnectionError("connection refused")

    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert events == [
        "release:turbo", "release:kk", "chat", "acquire:turbo", "acquire:kk",
    ], events
    for model in models.values():
        assert model.model.model_is_loaded


def test_stt_release_skips_models_that_are_already_off_the_gpu(phase):
    events, _, _ = phase
    assert gpu.stt_release() == 2
    events.clear()
    assert gpu.stt_release() == 0, "a second release must be a no-op"
    assert events == []
    assert gpu.stt_acquire() == 2
    assert gpu.stt_acquire() == 0, "a second acquire must be a no-op"


def test_phase_manager_is_a_noop_on_the_cpu_path(phase):
    """STT_DEVICE=cpu must behave exactly as it did before the GPU existed."""
    events, models, stub_ollama = phase
    gpu.set_enabled(False)

    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert events == ["chat"]
    assert stub_ollama.calls == [], "no ollama call may be made on the CPU path"
    for model in models.values():
        assert model.model.unloads == 0 and model.model.loads == 0
    assert pipeline._llm_keep_alive() == "30m"


def test_no_handoff_when_every_model_fell_back_to_the_cpu(phase):
    """Enabled but nothing of ours on the card: evicting the LLM would be pointless.

    The lock stays on (a future CUDA load still has to be serialised), but the
    release/unload/acquire dance and the short keep_alive do not.
    """
    events, _, stub_ollama = phase
    gpu.forget_stt_models()

    assert gpu.is_enabled() and not gpu.is_active()
    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert events == ["chat"]
    assert stub_ollama.calls == []
    assert pipeline._llm_keep_alive() == "30m"


def test_keep_alive_is_short_while_the_gpu_is_shared(phase):
    assert gpu.is_active()
    assert pipeline._llm_keep_alive() == "2m"


# --------------------------------------------------------------------------
# Fix round 1, finding 1: wait for Ollama, retry the acquire, degrade honestly
# --------------------------------------------------------------------------

def test_llm_phase_waits_for_the_llm_to_leave_before_reacquiring(phase):
    """`unload_llm()` only means Ollama *accepted* the request.

    Allocating ~2 GiB of STT on top of an LLM that is still resident is how a
    6 GiB card OOMs, so the acquire may not start until `ps()` says it is gone.
    """
    events, _, stub_ollama = phase
    stub_ollama.busy_polls = 3

    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert events == [
        "release:turbo", "release:kk", "chat", "unload-llm",
        "ps-busy", "ps-busy", "ps-busy", "ps-clear",
        "acquire:turbo", "acquire:kk",
    ], events
    # No acquire may appear before the card is reported free.
    assert events.index("ps-clear") < events.index("acquire:turbo")


def test_llm_phase_gives_up_waiting_after_the_bound(phase):
    """A stuck Ollama must not hang the phase: bounded wait, then acquire anyway."""
    events, models, stub_ollama = phase
    stub_ollama.busy_polls = 10_000              # never clears

    t0 = time.perf_counter()
    with gpu.llm_phase("stub-model"):
        events.append("chat")
    elapsed = time.perf_counter() - t0

    assert gpu.LLM_UNLOAD_TIMEOUT <= elapsed < gpu.LLM_UNLOAD_TIMEOUT + 1.0, elapsed
    assert "ps-clear" not in events
    assert events[-2:] == ["acquire:turbo", "acquire:kk"], events
    for model in models.values():
        assert model.model.model_is_loaded


def test_acquire_failing_once_then_succeeding_leaves_no_degraded_flag(phase):
    events, models, _ = phase
    models["turbo"].model.load_failures = 1

    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert "acquire-failed:turbo" in events
    assert gpu.degraded_reason() is None, "one failure is a race, not a diagnosis"
    for model in models.values():
        assert model.model.model_is_loaded
    assert pipeline.stt_device() != "degraded"


def test_acquire_failing_twice_sets_the_degraded_flag_and_health_says_so(phase):
    events, models, _ = phase
    models["turbo"].model.load_failures = 2

    with gpu.llm_phase("stub-model"):
        events.append("chat")

    assert gpu.degraded_reason() is not None
    assert "out of memory" in gpu.degraded_reason()
    assert not models["turbo"].model.model_is_loaded
    # The whole point: /health must not keep claiming a healthy cuda.
    assert pipeline.stt_device() == "degraded"
    assert pipeline.stt_device() != "cuda"


def test_a_degraded_card_recovers_on_the_next_phase_lock(phase):
    """Once whatever took the VRAM lets go, the next GPU use puts STT back."""
    events, models, _ = phase
    models["kk"].model.load_failures = 2

    with gpu.llm_phase("stub-model"):
        pass
    assert gpu.degraded_reason() is not None
    assert not models["kk"].model.model_is_loaded

    with gpu.phase_lock():                       # e.g. the next transcription window
        pass

    assert gpu.degraded_reason() is None
    assert models["kk"].model.model_is_loaded
    assert pipeline.stt_device() != "degraded"


# --------------------------------------------------------------------------
# Fix round 1, finding 2: a failed CUDA build is retried, and scoped to one model
# --------------------------------------------------------------------------

class FakeWhisperFactory:
    """Stands in for `WhisperModel(...)`, failing the first N cuda builds."""

    def __init__(self, events: list[str], cuda_failures: int = 0) -> None:
        self.events = events
        self.cuda_failures = cuda_failures
        self.attempts: list[str] = []

    def __call__(self, name_or_path, device=None, compute_type=None, cpu_threads=None):
        self.attempts.append(device)
        if device == "cuda":
            if self.cuda_failures > 0:
                self.cuda_failures -= 1
                raise RuntimeError("CUDA failed with error out of memory")
        return StubModel(self.events, str(name_or_path), device=device)


@pytest.fixture
def cuda_build(monkeypatch, phase):
    """`_new_model` in cuda mode, with instant retries and restored globals."""
    saved_devices = dict(pipeline._model_devices)
    gpu.forget_stt_models()
    pipeline._model_devices.clear()
    monkeypatch.setattr(pipeline, "_stt_device", "cuda")
    monkeypatch.setattr(pipeline, "CUDA_LOAD_RETRY_DELAY", 0.0)
    yield phase[0]
    pipeline._model_devices.clear()
    pipeline._model_devices.update(saved_devices)


def test_cuda_build_that_fails_once_succeeds_on_the_retry(monkeypatch, cuda_build):
    """A transient OOM is usually another allocation about to be released."""
    factory = FakeWhisperFactory(cuda_build, cuda_failures=1)
    monkeypatch.setattr(pipeline, "WhisperModel", factory)

    model = pipeline._new_model("turbo", "turbo-id")

    assert factory.attempts == ["cuda", "cuda"], "it must retry on cuda, not give up to cpu"
    assert model.model.device == "cuda"
    assert pipeline.stt_devices() == {"turbo": "cuda"}
    assert pipeline.stt_device() == "cuda"
    assert gpu.registered_stt_models() == (model,)


def test_cuda_build_that_fails_twice_falls_back_for_that_model_only(
    monkeypatch, cuda_build
):
    """The old behaviour latched the whole service onto the CPU. It must not."""
    good = FakeWhisperFactory(cuda_build, cuda_failures=0)
    monkeypatch.setattr(pipeline, "WhisperModel", good)
    turbo = pipeline._new_model("turbo", "turbo-id")

    bad = FakeWhisperFactory(cuda_build, cuda_failures=2)
    monkeypatch.setattr(pipeline, "WhisperModel", bad)
    kk = pipeline._new_model("kk", "kk-dir")

    assert bad.attempts == ["cuda", "cuda", "cpu"], bad.attempts
    assert turbo.model.device == "cuda"
    assert kk.model.device == "cpu"

    # Per-model truth, and a summary that does not pretend to be either one.
    assert pipeline.stt_devices() == {"turbo": "cuda", "kk": "cpu"}
    assert pipeline.stt_compute_types() == {"turbo": "int8_float16", "kk": "int8"}
    assert pipeline.stt_device() == "mixed"
    assert pipeline.stt_compute_type() == "mixed"
    # Only the cuda one is handed back and forth.
    assert gpu.registered_stt_models() == (turbo,)

    # And the global request is NOT latched: the next model still tries cuda.
    assert pipeline._stt_device == "cuda"
    later = FakeWhisperFactory(cuda_build, cuda_failures=0)
    monkeypatch.setattr(pipeline, "WhisperModel", later)
    third = pipeline._new_model("turbo", "turbo-id")
    assert later.attempts == ["cuda"]
    assert third.model.device == "cuda"
    assert pipeline.stt_devices()["turbo"] == "cuda"


# --------------------------------------------------------------------------
# Serialisation (weight-free, always run)
# --------------------------------------------------------------------------

def test_gpu_lock_serialises_a_transcription_and_an_llm_phase(phase):
    """An upload and a Finish must never overlap: they want the same 5.7 GiB."""
    events, _, _ = phase
    started = threading.Event()

    def transcription() -> None:
        with gpu.phase_lock():
            events.append("stt-start")
            started.set()
            time.sleep(0.20)
            events.append("stt-end")

    def llm_phase() -> None:
        started.wait(2.0)                   # make the LLM thread arrive second
        with gpu.llm_phase("stub-model"):
            events.append("llm-start")
            time.sleep(0.05)
            events.append("llm-end")

    threads = [threading.Thread(target=transcription), threading.Thread(target=llm_phase)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5.0)
        assert not thread.is_alive()

    # Not merely "no overlap": the models may not even start moving off the card
    # until the transcription has finished with them.
    assert events == [
        "stt-start", "stt-end",
        "release:turbo", "release:kk",
        "llm-start", "llm-end",
        "unload-llm", "ps-clear",
        "acquire:turbo", "acquire:kk",
    ], events


def test_phase_lock_is_reentrant_within_one_thread(phase):
    """transcribe_pcm() holds the lock while its callees take it again."""
    with gpu.phase_lock():
        with gpu.phase_lock():
            assert True


# --------------------------------------------------------------------------
# CUDA preload (always run; asserts the diagnosis, not the hardware)
# --------------------------------------------------------------------------

def test_cuda_ready_reports_a_reason_either_way():
    ready, reason = gpu.cuda_ready()
    assert isinstance(ready, bool)
    assert reason, "a device decision must always come with a reason to log"
    if ready:
        assert any("libcublas.so.12" in p for p in gpu.preloaded), gpu.preloaded
        assert any("libcudnn.so.9" in p for p in gpu.preloaded), gpu.preloaded


def test_resolved_device_is_one_of_the_supported_states():
    assert pipeline.stt_device() in ("cpu", "cuda", "mixed", "degraded")
    assert pipeline.stt_compute_type() in ("int8", "int8_float16", "mixed")
    assert set(pipeline.stt_devices().values()) <= {"cpu", "cuda"}
    # The lock is armed exactly when the GPU path is in play at all...
    assert gpu.is_enabled() == (pipeline._stt_device == "cuda")
    # ...and a handoff is worth doing exactly when something of ours is resident.
    assert gpu.is_active() == (gpu.is_enabled() and bool(gpu.registered_stt_models()))


# --------------------------------------------------------------------------
# GPU-gated: the real thing
# --------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    pipeline._stt_device != "cuda", reason="STT did not resolve to CUDA on this box"
)


def _recall(reference: str, hypothesis: str) -> float:
    ref = Counter(w for w in _NON_WORD.split(reference.lower()) if w)
    hyp = Counter(w for w in _NON_WORD.split(hypothesis.lower()) if w)
    total = sum(ref.values())
    return sum((ref & hyp).values()) / total if total else 0.0


def _ollama_ready() -> bool:
    try:
        available = {
            m.get("model") or m.get("name")
            for m in pipeline.ollama.list().get("models", [])
        }
    except Exception:
        return False
    return any(c in available for c in pipeline.LLM_MODEL_CANDIDATES)


@requires_cuda
def test_transcribe_on_cuda_is_fast_and_accurate():
    audio = AUDIO_DIR / "kk_1.wav"
    reference = audio.with_suffix(".txt")
    if not audio.exists() or not reference.exists():
        pytest.skip(f"missing audio fixture {audio}")

    pipeline.warm_up()                      # load cost is not what we are measuring
    assert pipeline.stt_devices().get("turbo") == "cuda", pipeline.stt_devices()

    t0 = time.perf_counter()
    transcript = pipeline.transcribe(audio)
    elapsed = time.perf_counter() - t0

    recall = _recall(reference.read_text(encoding="utf-8"), transcript.text)
    print(
        f"\n[cuda kk_1] {transcript.seconds:.1f}s audio in {elapsed:.2f}s "
        f"({transcript.seconds / elapsed:.1f}x realtime) recall={recall:.2f}"
    )
    assert elapsed < 4.0, f"CUDA transcription took {elapsed:.2f}s"
    assert recall >= 0.5, f"word recall {recall:.2f} < 0.5"
    assert transcript.language == "kk", transcript.languages


@requires_cuda
@pytest.mark.skipif(not _ollama_ready(), reason="Ollama unreachable or no model pulled")
def test_structure_gives_the_gpu_back_to_stt():
    pipeline.warm_up()
    models = gpu.registered_stt_models()
    assert models, "warm_up() on cuda must register both STT models"

    pipeline.structure("тест", language_hint="ru")

    assert gpu.degraded_reason() is None, gpu.degraded_reason()
    for model in models:
        assert model.model.device == "cuda"
        assert model.model.model_is_loaded, "STT was left off the GPU after structuring"
    assert pipeline.stt_device() == "cuda"

    # And STT still works right afterwards, on the GPU, without a reload.
    transcript = pipeline.transcribe_pcm(
        pipeline.decode_audio(str(AUDIO_DIR / "kk_1.wav"), sampling_rate=pipeline.SAMPLE_RATE)[
            : 8 * pipeline.SAMPLE_RATE
        ],
        pipeline.SAMPLE_RATE,
    )
    assert transcript.text
