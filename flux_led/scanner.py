import asyncio
import contextlib
import logging
import select
import socket
import time
from datetime import date
from typing import (
    Optional,
    TypedDict,  # pylint: disable=no-name-in-module
    Union,
)

from .const import (
    ATTR_FIRMWARE_DATE,
    ATTR_ID,
    ATTR_IPADDR,
    ATTR_MODEL,
    ATTR_MODEL_DESCRIPTION,
    ATTR_MODEL_INFO,
    ATTR_MODEL_NUM,
    ATTR_REMOTE_ACCESS_ENABLED,
    ATTR_REMOTE_ACCESS_HOST,
    ATTR_REMOTE_ACCESS_PORT,
    ATTR_VERSION_NUM,
)
from .models_db import get_model_description

_LOGGER = logging.getLogger(__name__)

MESSAGE_SEND_INTERLEAVE_DELAY = 0.4
LEGACY_OUI = "ACCF23"


class FluxLEDDiscovery(TypedDict):
    """A flux led device."""

    ipaddr: str
    id: Optional[str]  # aka mac
    model: Optional[str]
    model_num: Optional[int]
    version_num: Optional[int]
    firmware_date: Optional[date]
    model_info: Optional[str]  # contains if IR (and maybe BL) if the device supports IR
    model_description: Optional[str]
    remote_access_enabled: Optional[bool]
    remote_access_host: Optional[str]  # the remote access host
    remote_access_port: Optional[int]  # the remote access port


def is_legacy_device(discovery: Optional[FluxLEDDiscovery]) -> bool:
    """Check if a discovery is a legacy device."""
    if not discovery:
        return False
    is_legacy_mac = False
    if discovery.get(ATTR_ID):
        mac = discovery[ATTR_ID]
        assert mac is not None
        is_legacy_mac = mac.startswith(LEGACY_OUI)
    return is_legacy_mac or bool(
        discovery.get(ATTR_VERSION_NUM) and not discovery.get(ATTR_MODEL_NUM)
    )


def create_udp_socket(discovery_port: int) -> socket.socket:
    """Create a udp socket used for communicating with the device."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    try:
        # Legacy devices require source port to be the discovery port
        sock.bind(("", discovery_port))
    except OSError as err:
        _LOGGER.debug("Port %s is not available: %s", discovery_port, err)
        sock.bind(("", 0))
    sock.setblocking(False)
    return sock


def merge_discoveries(target: FluxLEDDiscovery, source: FluxLEDDiscovery) -> None:
    """Merge keys from a second discovery that may be missing from the first one."""
    for k, v in source.items():
        if target.get(k) is None:
            target[k] = v  # type: ignore[literal-required]


def _strip_new_lines(msg: str) -> str:
    return msg.replace("\r", "").replace("\n", "")


def _process_discovery_message(data: FluxLEDDiscovery, decoded_data: str) -> None:
    """Process response from b'HF-A11ASSISTHREAD'

    b'192.168.214.252,B4E842E10588,AK001-ZJ2145'
    """
    data_split = _strip_new_lines(decoded_data).split(",")
    if len(data_split) < 3:
        return
    ipaddr = data_split[0]
    data.update(
        {
            ATTR_IPADDR: ipaddr,
            ATTR_ID: data_split[1],
            ATTR_MODEL: data_split[2],
        }
    )


def _process_version_message(data: FluxLEDDiscovery, decoded_data: str) -> None:
    r"""Process response from b'AT+LVER\r'

    b'+ok=07_06_20210106_ZG-BL\r'
    """
    version_data = _strip_new_lines(decoded_data[4:])
    data_split = version_data.split("_", maxsplit=3)
    if len(data_split) == 1:
        with contextlib.suppress(ValueError):
            data[ATTR_VERSION_NUM] = int(data_split[0], 16)
        return
    if len(data_split) >= 2:
        try:
            data[ATTR_MODEL_NUM] = int(data_split[0], 16)
            data[ATTR_VERSION_NUM] = int(data_split[1], 16)
        except ValueError:
            return
    assert data[ATTR_MODEL_NUM] is not None
    if len(data_split) >= 3:
        firmware_date = data_split[2]
        with contextlib.suppress(TypeError, ValueError):
            data[ATTR_FIRMWARE_DATE] = date(
                int(firmware_date[:4]),
                int(firmware_date[4:6]),
                int(firmware_date[6:8]),
            )
    if len(data_split) == 4:
        data[ATTR_MODEL_INFO] = data_split[3]
    data[ATTR_MODEL_DESCRIPTION] = get_model_description(
        data[ATTR_MODEL_NUM], data[ATTR_MODEL_INFO]
    )


def _process_remote_access_message(data: FluxLEDDiscovery, decoded_data: str) -> None:
    """Process response from b'AT+SOCKB\r'

    b'+ok=TCP,8816,ra8816us02.magichue.net\r'
    """
    data_split = _strip_new_lines(decoded_data).split(",")
    if len(data_split) < 3:
        if not data.get(ATTR_REMOTE_ACCESS_ENABLED):
            data[ATTR_REMOTE_ACCESS_ENABLED] = False
        return
    try:
        data.update(
            {
                ATTR_REMOTE_ACCESS_ENABLED: True,
                ATTR_REMOTE_ACCESS_PORT: int(data_split[1]),
                ATTR_REMOTE_ACCESS_HOST: data_split[2],
            }
        )
    except ValueError:
        return


class BulbScanner:
    DISCOVERY_PORT = 48899
    BROADCAST_FREQUENCY = 6  # At least 6 for 0xA1 models
    RESPONSE_SIZE = 64
    DISCOVER_MESSAGE = b"HF-A11ASSISTHREAD"
    VERSION_MESSAGE = b"AT+LVER\r"
    REMOTE_ACCESS_MESSAGE = b"AT+SOCKB\r"
    DISABLE_REMOTE_ACCESS_MESSAGE = b"AT+SOCKB=NONE\r"
    REBOOT_MESSAGE = b"AT+Z\r"
    ALL_MESSAGES = {DISCOVER_MESSAGE, VERSION_MESSAGE, REMOTE_ACCESS_MESSAGE}
    BROADCAST_ADDRESS = "<broadcast>"

    def __init__(self) -> None:
        self._discoveries: dict[str, FluxLEDDiscovery] = {}

    @property
    def found_bulbs(self) -> list[FluxLEDDiscovery]:
        """Return only complete bulb discoveries."""
        return [info for info in self._discoveries.values() if info["id"]]

    def getBulbInfoByID(self, id: str) -> FluxLEDDiscovery:
        for b in self.found_bulbs:
            if b["id"] == id:
                return b
        return b

    def getBulbInfo(self) -> list[FluxLEDDiscovery]:
        return self.found_bulbs

    def _create_socket(self) -> socket.socket:
        return create_udp_socket(self.DISCOVERY_PORT)

    def _destination_from_address(self, address: Optional[str]) -> tuple[str, int]:
        if address is None:
            address = self.BROADCAST_ADDRESS
        return (address, self.DISCOVERY_PORT)

    def _process_response(
        self,
        data: Optional[bytes],
        from_address: tuple[str, int],
        address: Optional[str],
        response_list: dict[str, FluxLEDDiscovery],
    ) -> bool:
        """Process a response.

        Returns True if processing should stop
        """
        if data is None:
            return False
        if data in self.ALL_MESSAGES:
            return False
        decoded_data = data.decode("ascii")
        self._process_data(from_address, decoded_data, response_list)
        if address is None or address not in response_list:
            return False
        response = response_list[address]
        return is_legacy_device(response) or (
            response[ATTR_MODEL_NUM] is not None
            and response[ATTR_REMOTE_ACCESS_ENABLED] is not None
        )

    def _process_data(
        self,
        from_address: tuple[str, int],
        decoded_data: str,
        response_list: dict[str, FluxLEDDiscovery],
    ) -> None:
        """Process data."""
        from_ipaddr = from_address[0]
        data = response_list.setdefault(
            from_ipaddr,
            FluxLEDDiscovery(
                ipaddr=from_ipaddr,
                id=None,
                model=None,
                model_num=None,
                version_num=None,
                firmware_date=None,
                model_info=None,
                model_description=None,
                remote_access_enabled=None,
                remote_access_host=None,
                remote_access_port=None,
            ),
        )
        if (
            decoded_data.startswith("+ok=T")
            or decoded_data == "+ok="
            or decoded_data == "+ok=\r"
        ):
            _process_remote_access_message(data, decoded_data)
        if decoded_data.startswith("+ok="):
            _process_version_message(data, decoded_data)
        elif "," in decoded_data:
            _process_discovery_message(data, decoded_data)

    def _get_start_messages(
        self,
    ) -> list[bytes]:
        return [self.DISCOVER_MESSAGE]

    def _get_enable_remote_access_messages(
        self,
        remote_access_host: str,
        remote_access_port: int,
    ) -> list[bytes]:
        enable_message = f"AT+SOCKB=TCP,{remote_access_port},{remote_access_host}\r"
        return [enable_message.encode()]

    def _get_disable_remote_access_messages(
        self,
    ) -> list[bytes]:
        return [self.DISABLE_REMOTE_ACCESS_MESSAGE]

    def _get_reboot_messages(
        self,
    ) -> list[bytes]:
        return [self.REBOOT_MESSAGE]

    def _send_message(
        self,
        sender: Union[socket.socket, asyncio.DatagramTransport],
        destination: tuple[str, int],
        message: bytes,
    ) -> None:
        _LOGGER.debug("udp: %s => %s", destination, message)
        sender.sendto(message, destination)

    def _send_messages(
        self,
        messages: list[bytes],
        sender: Union[socket.socket, asyncio.DatagramTransport],
        destination: tuple[str, int],
    ) -> None:
        """Send messages with a short delay between them."""
        for idx, message in enumerate(messages):
            self._send_message(sender, destination, message)
            if idx != len(messages):
                time.sleep(MESSAGE_SEND_INTERLEAVE_DELAY)

    def get_discovery_messages(
        self,
    ) -> list[bytes]:
        return [self.DISCOVER_MESSAGE, self.VERSION_MESSAGE, self.REMOTE_ACCESS_MESSAGE]

    def scan(
        self, timeout: int = 10, address: Optional[str] = None
    ) -> list[FluxLEDDiscovery]:
        """Scan for bulbs.

        If an address is provided, the scan will return
        as soon as it gets a response from that address
        """
        discovery_messages = self.get_discovery_messages()
        sock = self._create_socket()
        destination = self._destination_from_address(address)
        # set the time at which we will quit the search
        quit_time = time.monotonic() + timeout
        found_all = False
        # outer loop for query send
        while not found_all:
            if time.monotonic() > quit_time:
                break
            # send out a broadcast query
            self._send_messages(discovery_messages, sock, destination)
            # inner loop waiting for responses
            while True:
                sock.settimeout(1)
                remain_time = quit_time - time.monotonic()
                time_out = min(remain_time, timeout / self.BROADCAST_FREQUENCY)
                if time_out <= 0:
                    break
                read_ready, _, _ = select.select([sock], [], [], time_out)
                if not read_ready:
                    if time.monotonic() < quit_time:
                        # No response, send broadcast again in cast it got lost
                        self._send_messages(discovery_messages, sock, destination)
                    continue

                try:
                    data, addr = sock.recvfrom(self.RESPONSE_SIZE)
                    _LOGGER.debug("discover: %s <= %s", addr, data)
                except socket.timeout:
                    continue

                if self._process_response(data, addr, address, self._discoveries):
                    found_all = True
                    break

        return self.found_bulbs
