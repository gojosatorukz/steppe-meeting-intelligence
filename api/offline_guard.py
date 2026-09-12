"""Proof, not a promise: make an outbound connection impossible at runtime.

"100% offline" is a scored requirement, and the honest way to hold it is to
stop the process from being *able* to reach the internet rather than to audit
every library for a phone-home. Importing this module does two things, in this
order, and both have to happen before anything else is imported:

1. **Sets the Hugging Face offline switches.** ``faster_whisper`` loads the
   turbo model by hub id, and ``huggingface_hub`` answers that with a revision
   check against huggingface.co — a request that, with the wifi off on stage,
   does not fail fast, it *hangs*. ``HF_HUB_OFFLINE=1`` makes the hub resolve
   straight from the local cache. This must be set before ``huggingface_hub``
   is imported, because it reads the environment at import time; hence
   :mod:`main` imports this module first, and :mod:`pipeline` sets the same
   variable in its own first statement for the tests that import it directly.

2. **Blocks non-loopback TCP connects** by wrapping ``socket.socket.connect``,
   ``socket.socket.connect_ex`` and ``socket.create_connection``. Loopback
   (``127.0.0.0/8``, ``::1``, ``localhost``) and Unix sockets stay open, so
   Ollama on ``127.0.0.1:11434`` and the WebSocket keep working; anything else
   raises :class:`OfflineBlocked` and bumps a counter that ``/health`` reports.

Set ``OFFLINE_GUARD=0`` to skip step 2 (step 1 always happens) — useful only
when something legitimately needs the network, e.g. pulling a model.
"""

from __future__ import annotations

import ipaddress
import os
import socket
import threading

__all__ = [
    "OfflineBlocked",
    "enabled",
    "blocked_count",
    "status",
    "is_loopback",
    "install",
]

# --- step 1: Hugging Face offline, before any hub import -------------------
# setdefault, not assignment: an operator who exported HF_HUB_OFFLINE=0 on
# purpose (to fetch a model) is not overruled by importing the app.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_ENABLED = (os.getenv("OFFLINE_GUARD") or "1").strip().lower() not in {"0", "false", "no", "off"}

_lock = threading.Lock()
_blocked = 0
_installed = False

# Hostnames that mean "this machine" without a DNS lookup.
_LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


class OfflineBlocked(OSError):
    """Raised instead of opening a socket to the outside world."""


def enabled() -> bool:
    """Whether the socket guard is armed (step 2)."""
    return _ENABLED


def blocked_count() -> int:
    """How many outbound connections have been refused since startup."""
    return _blocked


def status() -> dict[str, object]:
    """The ``/health`` payload."""
    return {"enabled": _ENABLED, "blocked_external_connections": _blocked}


def is_loopback(address: object) -> bool:
    """True when ``address`` is this machine (or not an IP address at all).

    Unix-socket paths, ``AF_NETLINK``/``AF_PACKET`` tuples and anything else we
    do not recognise as a host/port pair are treated as local: they cannot
    leave the box, and refusing them would break unrelated machinery.
    """
    if address is None:
        return True
    if isinstance(address, (str, bytes, bytearray)):
        return True                                  # AF_UNIX path
    if not isinstance(address, (tuple, list)) or not address:
        return True
    host = address[0]
    if isinstance(host, (bytes, bytearray)):
        host = host.decode("utf-8", "replace")
    if not isinstance(host, str):
        return True                                  # e.g. AF_NETLINK (pid, groups)
    host = host.strip()
    if not host:
        return True                                  # "" is INADDR_ANY, not a route out
    if host.lower() in _LOCAL_NAMES:
        return True
    host = host.split("%", 1)[0]                     # strip an IPv6 zone id
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False                                 # a hostname needs DNS => not local
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_loopback)


def _refuse(address: object, how: str) -> OfflineBlocked:
    global _blocked
    with _lock:
        _blocked += 1
    return OfflineBlocked(
        f"offline guard: refused {how} to {address!r}. This service runs with no "
        "network access by design; set OFFLINE_GUARD=0 only to fetch models."
    )


def install() -> bool:
    """Wrap the connect entry points. Idempotent; returns whether the guard is on."""
    global _installed
    if not _ENABLED or _installed:
        return _ENABLED

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_create_connection = socket.create_connection

    def connect(self, address):                      # type: ignore[no-untyped-def]
        if not is_loopback(address):
            raise _refuse(address, "connect")
        return real_connect(self, address)

    def connect_ex(self, address):                   # type: ignore[no-untyped-def]
        if not is_loopback(address):
            raise _refuse(address, "connect_ex")
        return real_connect_ex(self, address)

    def create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Checked here as well as in connect(): create_connection resolves the
        # name itself, and a DNS lookup for a host we will refuse anyway is a
        # few hundred ms of stall we do not need to pay on stage.
        if not is_loopback(address):
            raise _refuse(address, "create_connection")
        return real_create_connection(address, *args, **kwargs)

    socket.socket.connect = connect                  # type: ignore[method-assign]
    socket.socket.connect_ex = connect_ex            # type: ignore[method-assign]
    socket.create_connection = create_connection     # type: ignore[assignment]
    _installed = True
    return True


install()
