"""
Thread-local network guard.
Blocks all outgoing connections ONLY for the inference thread.
Download threads are completely unaffected.
"""
import socket
import threading
import logging

logger = logging.getLogger(__name__)

_lock            = threading.Lock()
_blocked_threads: set[int] = set()
_original_connect     = socket.socket.connect
_original_getaddrinfo = socket.getaddrinfo
_patched = False


def _guarded_connect(self, address):
    if threading.current_thread().ident in _blocked_threads:
        raise BlockedNetworkError(
            f"[NetworkGuard] Outgoing connection BLOCKED during inference → {address}"
        )
    return _original_connect(self, address)


def _guarded_getaddrinfo(*args, **kwargs):
    if threading.current_thread().ident in _blocked_threads:
        raise BlockedNetworkError(
            f"[NetworkGuard] DNS lookup BLOCKED during inference → {args[0] if args else '?'}"
        )
    return _original_getaddrinfo(*args, **kwargs)


def _install_guard():
    global _patched
    if not _patched:
        socket.socket.connect = _guarded_connect
        socket.getaddrinfo    = _guarded_getaddrinfo
        _patched = True
        logger.info("NetworkGuard: socket patch installed.")


class BlockedNetworkError(ConnectionError):
    pass


class NetworkGuard:
    """
    Context manager. Blocks network for the current thread only.
    Usage:
        with NetworkGuard():
            llm.generate(...)   # any network call here raises BlockedNetworkError
    """

    def __enter__(self):
        _install_guard()
        tid = threading.current_thread().ident
        with _lock:
            _blocked_threads.add(tid)
        logger.debug(f"NetworkGuard: thread {tid} BLOCKED.")
        return self

    def __exit__(self, *_):
        tid = threading.current_thread().ident
        with _lock:
            _blocked_threads.discard(tid)
        logger.debug(f"NetworkGuard: thread {tid} UNBLOCKED.")
