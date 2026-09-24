"""Offline guard for scripts/verify.sh: any Python process started with this directory on PYTHONPATH
raises on outbound network connections (loopback/unix sockets are allowed). Enabled only when
DYVO_BLOCK_NETWORK=1, so it never affects normal use."""
import os

if os.environ.get("DYVO_BLOCK_NETWORK") == "1":
    import socket

    _orig_connect = socket.socket.connect
    _orig_connect_ex = socket.socket.connect_ex
    _LOCAL = ("127.", "::1", "localhost", "0.0.0.0")

    def _check(self, address):
        if self.family == getattr(socket, "AF_UNIX", None):
            return
        host = address[0] if isinstance(address, tuple) else str(address)
        if not str(host).startswith(_LOCAL):
            raise RuntimeError(f"[offline guard] network access blocked during verification: {address!r}")

    def connect(self, address):
        _check(self, address)
        return _orig_connect(self, address)

    def connect_ex(self, address):
        _check(self, address)
        return _orig_connect_ex(self, address)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    _orig_gai = socket.getaddrinfo

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not str(host).startswith(_LOCAL):
            raise RuntimeError(f"[offline guard] DNS lookup blocked during verification: {host!r}")
        return _orig_gai(host, *args, **kwargs)

    socket.getaddrinfo = getaddrinfo
