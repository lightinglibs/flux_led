import asyncio
import logging
import socket
from typing import Optional

from flux_led.aioscanner import AIOBulbScanner
from flux_led.protocol import OUTER_MESSAGE_WRAPPER

logging.basicConfig(level=logging.DEBUG)

_LOGGER = logging.getLogger(__name__)

DEVICE_ID = 0xA1
VERSION = 9
MINOR_VERSION = 28
ON_AT_START = False
# MODEL = "AK001-ZJ2101"  # Supports auto on
# MODEL = "AK001-ZJ2145"  # Supports dimmable effects
MODEL = "AK001-ZJ2147"


def get_local_ip():
    """Find the local ip address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:
        return None
    finally:
        s.close()


class MagicHomeDiscoveryProtocol(asyncio.Protocol):
    """A asyncio.Protocol implementing the MagicHome discovery protocol."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.local_ip = get_local_ip()
        self.transport: Optional[asyncio.BaseTransport] = None

    def connection_made(self, transport):
        self.transport = transport

    def send(self, data: bytes, addr: tuple[str, int]) -> None:
        """Trigger on data."""
        _LOGGER.debug(
            "UDP %s => %s (%d)",
            addr,
            data,
            len(data),
        )
        self.transport.sendto(data, addr)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        """Trigger on data."""
        _LOGGER.debug(
            "UDP %s <= %s (%d)",
            addr,
            data,
            len(data),
        )
        assert self.transport is not None
        model_str = hex(DEVICE_ID)[2:].zfill(2).upper()
        version_str = hex(VERSION)[2:].zfill(2).upper()
        minor_version_str = str(MINOR_VERSION).zfill(2).upper()
        if data.startswith(AIOBulbScanner.DISCOVER_MESSAGE):
            self.send(
                f"{self.local_ip},B4E842{model_str}{version_str}{minor_version_str},{MODEL}".encode(),
                addr,
            )
        if data.startswith(AIOBulbScanner.VERSION_MESSAGE):
            model_str = hex(DEVICE_ID)[2:].zfill(2).upper()
            self.send(
                f"+ok={model_str}_{minor_version_str}_20210428_ZG-BL\r".encode(), addr
            )

    def error_received(self, ex: Optional[Exception]) -> None:
        """Handle error."""
        _LOGGER.debug("LEDENETDiscovery error: %s", ex)

    def connection_lost(self, ex: Optional[Exception]) -> None:
        """The connection is lost."""


class MagichomeServerProtocol(asyncio.Protocol):
    """A asyncio.Protocol implementing the MagicHome protocol."""

    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.handler = None
        self.peername = None
        self.transport: Optional[asyncio.BaseTransport] = None

    def connection_lost(self, exc: Exception) -> None:
        """Handle connection lost."""
        _LOGGER.debug("%s: Connection lost: %s", self.peername, exc)

    def connection_made(self, transport: asyncio.Transport) -> None:
        """Handle incoming connection."""
        _LOGGER.debug("%s: Connection made", transport)
        self.peername = transport.get_extra_info("peername")
        self.transport = transport

    def send(self, data: bytes, random_byte: Optional[None]) -> None:
        """Trigger on data."""
        if random_byte is not None:
            msg = self.construct_wrapped_message(data, random_byte)
        else:
            msg = data
        _LOGGER.debug(
            "TCP %s => %s (%d)",
            self.peername,
            " ".join(f"0x{x:02X}" for x in msg),
            len(msg),
        )
        self.transport.write(msg)

    def data_received(self, data: bytes) -> None:
        """Process new data from the socket."""
        _LOGGER.debug(
            "TCP %s <= %s (%d)",
            self.peername,
            " ".join(f"0x{x:02X}" for x in data),
            len(data),
        )
        assert self.transport is not None
        if data.startswith(bytearray([*OUTER_MESSAGE_WRAPPER])):
            msg = data[10:-1]
            random = data[7]
        else:
            random = None
            msg = data

        if msg.startswith(bytearray([0x81])):
            self.send(
                self.construct_message(
                    bytearray(
                        [
                            0x81,
                            DEVICE_ID,
                            0x23 if ON_AT_START else 0x24,
                            0x61,
                            0xC5,
                            0x17,
                            0x18,
                            0x00,
                            0x00,
                            0x00,
                            VERSION,
                            0xF0,
                            0xF2,
                        ]
                    )
                ),
                random,
            )

    def construct_wrapped_message(
        self, inner_msg: bytearray, random_byte: int
    ) -> bytearray:
        """Construct a wrapped message."""
        inner_msg_len = len(inner_msg)
        return self.construct_message(
            bytearray(
                [
                    *OUTER_MESSAGE_WRAPPER,
                    random_byte,
                    inner_msg_len >> 8,
                    inner_msg_len & 0xFF,
                    *inner_msg,
                ]
            )
        )

    def construct_message(self, raw_bytes: bytearray) -> bytearray:
        """Calculate checksum of byte array and add to end."""
        csum = sum(raw_bytes) & 0xFF
        raw_bytes.append(csum)
        return raw_bytes


async def go():
    loop = asyncio.get_running_loop()
    await loop.create_server(
        lambda: MagichomeServerProtocol(),
        host="0.0.0.0",
        port=5577,
    )
    await loop.create_datagram_endpoint(
        lambda: MagicHomeDiscoveryProtocol(),
        local_addr=("0.0.0.0", AIOBulbScanner.DISCOVERY_PORT),
    )
    await asyncio.sleep(86400)


asyncio.run(go())
