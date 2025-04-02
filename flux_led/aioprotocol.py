from __future__ import annotations

import asyncio
import logging
from asyncio.transports import BaseTransport, WriteTransport
from typing import Any, Callable, cast

_LOGGER = logging.getLogger(__name__)


class AIOLEDENETProtocol(asyncio.Protocol):
    """A asyncio.Protocol implementing a wrapper around the LEDENET protocol."""

    def __init__(
        self,
        data_received: Callable[[bytes], Any],
        connection_lost: Callable[[Exception | None], Any],
    ) -> None:
        self._data_receive_callback = data_received
        self._connection_lost_callback = connection_lost
        self.transport: WriteTransport | None = None

    def connection_lost(self, exc: Exception | None) -> None:
        """Handle connection lost."""
        _LOGGER.debug("%s: Connection lost: %s", self.peername, exc)
        self.close()
        self._connection_lost_callback(exc)

    def connection_made(self, transport: BaseTransport) -> None:
        """Handle connection made."""
        self.transport = cast("WriteTransport", transport)
        self.peername = transport.get_extra_info("peername")

    def write(self, data: bytes) -> None:
        """Write data to the client."""
        assert self.transport is not None
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "%s => %s (%d)",
                self.peername,
                " ".join(f"0x{x:02X}" for x in data),
                len(data),
            )
        self.transport.write(data)

    def close(self) -> None:
        """Remove the connection and close the transport."""
        assert self.transport is not None
        self.transport.write_eof()
        self.transport.close()

    def data_received(self, data: bytes) -> None:
        """Process new data from the socket."""
        if _LOGGER.isEnabledFor(logging.DEBUG):
            _LOGGER.debug(
                "%s <= %s (%d)",
                self.peername,
                " ".join(f"0x{x:02X}" for x in data),
                len(data),
            )
        self._data_receive_callback(data)
