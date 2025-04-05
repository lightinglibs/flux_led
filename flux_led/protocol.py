"""FluxLED Protocols."""

from __future__ import annotations

import colorsys
import contextlib
import datetime
import logging
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from .const import (
    COLOR_MODE_RGB,
    COLOR_MODE_RGBW,
    MUSIC_PIXELS_MAX,
    MUSIC_PIXELS_PER_SEGMENT_MAX,
    MUSIC_SEGMENTS_MAX,
    PIXELS_MAX,
    PIXELS_PER_SEGMENT_MAX,
    SEGMENTS_MAX,
    TRANSITION_GRADUAL,
    TRANSITION_JUMP,
    TRANSITION_STROBE,
    LevelWriteMode,
    LevelWriteModeData,
    MultiColorEffects,
)
from .timer import LedTimer
from .utils import (
    scaled_color_temp_to_white_levels,
    utils,
    white_levels_to_scaled_color_temp,
)


class RemoteConfig(Enum):
    DISABLED = 0x01
    OPEN = 0x02
    PAIRED_ONLY = 0x03


class PowerRestoreState(Enum):
    ALWAYS_OFF = 0xFF
    ALWAYS_ON = 0x0F
    LAST_STATE = 0xF0


class MusicMode(Enum):
    STRIP = 0x26
    LIGHT_SCREEN = 0x27


@dataclass
class LEDENETAddressableDeviceConfiguration:
    pixels_per_segment: int  # pixels per segment
    segments: int | None  # number of segments
    music_pixels_per_segment: int | None  # music pixels per segment
    music_segments: int | None  # number of music segments
    wirings: list[str]  # available wirings in the current mode
    wiring: str | None  # RGB/BRG/GBR etc
    wiring_num: int | None  # RGB/BRG/GBR number
    num_to_wiring: dict[int, str]
    wiring_to_num: dict[str, int]
    ic_type: str | None  # WS2812B UCS.. etc
    ic_type_num: int | None  # WS2812B UCS.. number etc
    operating_mode: str | None  # RGB, RGBW


@dataclass
class PowerRestoreStates:
    channel1: PowerRestoreState | None
    channel2: PowerRestoreState | None
    channel3: PowerRestoreState | None
    channel4: PowerRestoreState | None


_LOGGER = logging.getLogger(__name__)


# Protocol names
PROTOCOL_LEDENET_ORIGINAL = "LEDENET_ORIGINAL"
PROTOCOL_LEDENET_ORIGINAL_RGBW = "LEDENET_ORIGINAL_RGBW"
PROTOCOL_LEDENET_ORIGINAL_CCT = "LEDENET_ORIGINAL_CCT"
PROTOCOL_LEDENET_9BYTE = "LEDENET"
PROTOCOL_LEDENET_9BYTE_AUTO_ON = "LEDENET_AUTO_ON"
PROTOCOL_LEDENET_9BYTE_DIMMABLE_EFFECTS = "LEDENET_DIMMABLE_EFFECTS"
PROTOCOL_LEDENET_SOCKET = "LEDENET_SOCKET"
PROTOCOL_LEDENET_8BYTE = "LEDENET_8BYTE"  # Previously was called None
PROTOCOL_LEDENET_8BYTE_AUTO_ON = "LEDENET_8BYTES_AUTO_ON"
PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS = "LEDENET_8BYTE_DIMMABLE_EFFECTS"
PROTOCOL_LEDENET_ADDRESSABLE_A1 = "LEDENET_ADDRESSABLE_A1"
PROTOCOL_LEDENET_ADDRESSABLE_A2 = "LEDENET_ADDRESSABLE_A2"
PROTOCOL_LEDENET_ADDRESSABLE_A3 = "LEDENET_ADDRESSABLE_A3"
PROTOCOL_LEDENET_CCT = "LEDENET_CCT"
PROTOCOL_LEDENET_CCT_WRAPPED = "LEDENET_CCT_WRAPPED"
PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS = "LEDENET_CHRISTMAS"
PROTOCOL_LEDENET_25BYTE = "LEDENET_25_BYTE"

TRANSITION_BYTES = {
    TRANSITION_JUMP: 0x3B,
    TRANSITION_STROBE: 0x3C,
    TRANSITION_GRADUAL: 0x3A,
}


LEDNET_MUSIC_MODE_RESPONSE_LEN = 13  # 72 01 26 01 00 00 00 00 00 00 64 64 62
LEDENET_POWER_RESTORE_RESPONSE_LEN = 7
LEDENET_ORIGINAL_STATE_RESPONSE_LEN = 11
LEDENET_STATE_RESPONSE_LEN = 14
LEDENET_POWER_RESPONSE_LEN = 4
LEDENET_ADDRESSABLE_STATE_RESPONSE_LEN = 25
LEDENET_A1_DEVICE_CONFIG_RESPONSE_LEN = 12
LEDENET_DEVICE_CONFIG_RESPONSE_LEN = 11
LEDENET_REMOTE_CONFIG_RESPONSE_LEN = 14  # 2b 03 00 00 00 00 29 00 00 00 00 00 00 57
LEDENET_TIME_RESPONSE_LEN = 12  # 10 14 16 01 02 10 26 20 07 00 0f a9
LEDENET_TIMERS_8BYTE_RESPONSE_LEN = 88
LEDENET_TIMERS_9BYTE_RESPONSE_LEN = 94
LEDENET_TIMERS_SOCKET_RESPONSE_LEN = 100

MSG_ORIGINAL_POWER_STATE = "original_power_state"
MSG_ORIGINAL_STATE = "original_state"
MSG_POWER_RESTORE_STATE = "power_restore_state"
MSG_POWER_STATE = "power_state"
MSG_STATE = "state"
MSG_TIME = "time"
MSG_TIMERS = "timers"
MSG_MUSIC_MODE_STATE = "music_mode_state"
MSG_ADDRESSABLE_STATE = "addressable_state"
MSG_DEVICE_CONFIG = "device_config"
MSG_A1_DEVICE_CONFIG = "a1_device_config"
MSG_REMOTE_CONFIG = "remote_config"

OUTER_MESSAGE_FIRST_BYTE = 0xB0

MSG_UNIQUE_START = {
    (0x01, 0x11): MSG_TIME,
    (0xF0, 0x11): MSG_TIME,
    (0x0F, 0x11): MSG_TIME,
    (0x00, 0x11): MSG_TIME,
    (0x01, 0x22): MSG_TIMERS,
    (0xF0, 0x22): MSG_TIMERS,
    (0x0F, 0x22): MSG_TIMERS,
    (0x00, 0x22): MSG_TIMERS,
    (0x01, 0x71): MSG_POWER_STATE,
    (0xF0, 0x71): MSG_POWER_STATE,
    (0x0F, 0x71): MSG_POWER_STATE,
    (0x00, 0x71): MSG_POWER_STATE,
    (0x01, 0x32): MSG_POWER_RESTORE_STATE,
    (0xF0, 0x32): MSG_POWER_RESTORE_STATE,
    (0x0F, 0x32): MSG_POWER_RESTORE_STATE,
    (0x00, 0x32): MSG_POWER_RESTORE_STATE,
    (0x78,): MSG_ORIGINAL_POWER_STATE,
    (0x66,): MSG_ORIGINAL_STATE,
    (0x81,): MSG_STATE,
    (0x01, 0x63): MSG_DEVICE_CONFIG,
    (0x00, 0x63): MSG_DEVICE_CONFIG,
    (0xF0, 0x63): MSG_DEVICE_CONFIG,
    (0x0F, 0x63): MSG_DEVICE_CONFIG,
    (0x63,): MSG_A1_DEVICE_CONFIG,
    (0x72,): MSG_MUSIC_MODE_STATE,
    (0x2B,): MSG_REMOTE_CONFIG,
}

MSG_LENGTHS = {
    MSG_TIME: LEDENET_TIME_RESPONSE_LEN,
    MSG_REMOTE_CONFIG: LEDENET_REMOTE_CONFIG_RESPONSE_LEN,
    MSG_MUSIC_MODE_STATE: LEDNET_MUSIC_MODE_RESPONSE_LEN,
    MSG_POWER_STATE: LEDENET_POWER_RESPONSE_LEN,
    MSG_POWER_RESTORE_STATE: LEDENET_POWER_RESTORE_RESPONSE_LEN,
    MSG_ORIGINAL_POWER_STATE: LEDENET_POWER_RESPONSE_LEN,
    MSG_ORIGINAL_STATE: LEDENET_ORIGINAL_STATE_RESPONSE_LEN,
    MSG_STATE: LEDENET_STATE_RESPONSE_LEN,
    MSG_ADDRESSABLE_STATE: LEDENET_ADDRESSABLE_STATE_RESPONSE_LEN,
    MSG_DEVICE_CONFIG: LEDENET_DEVICE_CONFIG_RESPONSE_LEN,
    MSG_A1_DEVICE_CONFIG: LEDENET_A1_DEVICE_CONFIG_RESPONSE_LEN,
}

OUTER_MESSAGE_WRAPPER_FIRST_BYTES = [OUTER_MESSAGE_FIRST_BYTE, 0xB1, 0xB2, 0xB3, 0x00]
OUTER_MESSAGE_WRAPPER = [*OUTER_MESSAGE_WRAPPER_FIRST_BYTES, 0x01]
OUTER_MESSAGE_WRAPPER_START_LEN = 10
CHECKSUM_LEN = 1


POWER_RESTORE_BYTES_TO_POWER_RESTORE = {
    restore_state.value: restore_state for restore_state in PowerRestoreState
}


REMOTE_CONFIG_BYTES_TO_REMOTE_CONFIG = {
    remote_config.value: remote_config for remote_config in RemoteConfig
}


def _message_type_from_start_of_msg(data: bytes) -> str | None:
    if len(data) > 1:
        return MSG_UNIQUE_START.get(
            (data[0], data[1]), MSG_UNIQUE_START.get((data[0],))
        )
    return MSG_UNIQUE_START.get((data[0],)) if len(data) else None


class LEDENETOriginalRawState(NamedTuple):
    head: int
    model_num: int
    power_state: int
    preset_pattern: int
    mode: int
    speed: int
    red: int
    green: int
    blue: int
    warm_white: int
    check_sum: int
    cool_white: int


# typical response:
# pos  0  1  2  3  4  5  6  7  8  9 10
#    66 01 24 39 21 0a ff 00 00 01 99
#     |  |  |  |  |  |  |  |  |  |  |
#     |  |  |  |  |  |  |  |  |  |  checksum
#     |  |  |  |  |  |  |  |  |  warmwhite
#     |  |  |  |  |  |  |  |  blue
#     |  |  |  |  |  |  |  green
#     |  |  |  |  |  |  red
#     |  |  |  |  |  speed: 0f = highest f0 is lowest
#     |  |  |  |  <don't know yet>
#     |  |  |  preset pattern
#     |  |  off(24)/on(23)
#     |  model_num (type)
#     msg head
#


class LEDENETRawState(NamedTuple):
    head: int
    model_num: int
    power_state: int
    preset_pattern: int
    mode: int
    speed: int
    red: int
    green: int
    blue: int
    warm_white: int
    version_number: int
    cool_white: int
    color_mode: int
    check_sum: int


# response from a 5-channel LEDENET controller:
# pos  0  1  2  3  4  5  6  7  8  9 10 11 12 13
#    81 25 23 61 21 06 38 05 06 f9 01 00 0f 9d
#     |  |  |  |  |  |  |  |  |  |  |  |  |  |
#     |  |  |  |  |  |  |  |  |  |  |  |  |  checksum
#     |  |  |  |  |  |  |  |  |  |  |  |  color mode (f0 colors were set, 0f whites, 00 all were set)
#     |  |  |  |  |  |  |  |  |  |  |  cool-white  0x00 to 0xFF
#     |  |  |  |  |  |  |  |  |  |  version number
#     |  |  |  |  |  |  |  |  |  warmwhite  0x00 to 0xFF
#     |  |  |  |  |  |  |  |  blue  0x00 to 0xFF
#     |  |  |  |  |  |  |  green  0x00 to 0xFF
#     |  |  |  |  |  |  red 0x00 to 0xFF
#     |  |  |  |  |  speed: 0x01 = highest 0x1f is lowest
#     |  |  |  |  Mode WW(01), WW+CW(02), RGB(03), RGBW(04), RGBWW(05)
#     |  |  |  preset pattern
#     |  |  off(24)/on(23)
#     |  model_num (type)
#     msg head
#
RGB_NUM_TO_WIRING = {1: "RGB", 2: "GRB", 3: "BRG"}
RGB_WIRING_TO_NUM = {v: k for k, v in RGB_NUM_TO_WIRING.items()}
RGBW_NUM_TO_WIRING = {1: "RGBW", 2: "GRBW", 3: "BRGW"}
RGBW_WIRING_TO_NUM = {v: k for k, v in RGBW_NUM_TO_WIRING.items()}
RGBW_NUM_TO_MODE = {4: "RGB&W", 6: "RGB/W"}
RGBW_MODE_TO_NUM = {v: k for k, v in RGBW_NUM_TO_MODE.items()}
RGBWW_NUM_TO_WIRING = {
    1: "RGBCW",
    2: "GRBCW",
    3: "BRGCW",
    4: "RGBWC",
    5: "GRBWC",
    6: "BRGWC",
    7: "WRGBC",
    8: "WGRBC",
    9: "WBRGC",
    10: "CRGBW",
    11: "CBRBW",
    12: "CBRGW",
    13: "WCRGB",
    14: "WCGRB",
    15: "WCBRG",
}
RGBWW_WIRING_TO_NUM = {v: k for k, v in RGBWW_NUM_TO_WIRING.items()}
RGBWW_NUM_TO_MODE = {5: "RGB&CCT", 7: "RGB/CCT"}
RGBWW_MODE_TO_NUM = {v: k for k, v in RGBWW_NUM_TO_MODE.items()}

ADDRESSABLE_RGB_NUM_TO_WIRING = {
    0: "RGB",
    1: "RBG",
    2: "GRB",
    3: "GBR",
    4: "BRG",
    5: "BGR",
}
ADDRESSABLE_RGB_WIRING_TO_NUM = {v: k for k, v in ADDRESSABLE_RGB_NUM_TO_WIRING.items()}
ADDRESSABLE_RGBW_NUM_TO_WIRING = {
    0: "RGBW",
    1: "RBGW",
    2: "GRBW",
    3: "GBRW",
    4: "BRGW",
    5: "BGRW",
    6: "WRGB",
    7: "WRBG",
    8: "WGRB",
    9: "WGBR",
    10: "WBRG",
    11: "WBGR",
}
ADDRESSABLE_RGBW_WIRING_TO_NUM = {
    v: k for k, v in ADDRESSABLE_RGBW_NUM_TO_WIRING.items()
}


A1_NUM_TO_PROTOCOL = {
    1: "UCS1903",
    2: "SM16703",
    3: "WS2811",
    4: "WS2812B",
    5: "SK6812",
    6: "INK1003",
    7: "WS2801",
    8: "LB1914",
}
A1_PROTOCOL_TO_NUM = {v: k for k, v in A1_NUM_TO_PROTOCOL.items()}

A1_NUM_TO_OPERATING_MODE = {
    1: COLOR_MODE_RGB,
    2: COLOR_MODE_RGB,
    3: COLOR_MODE_RGB,
    4: COLOR_MODE_RGB,
    5: COLOR_MODE_RGB,
    6: COLOR_MODE_RGB,
    7: COLOR_MODE_RGB,
    8: COLOR_MODE_RGB,
}
A1_OPERATING_MODE_TO_NUM = {v: k for k, v in A1_NUM_TO_OPERATING_MODE.items()}

A2_NUM_TO_PROTOCOL = {
    1: "UCS1903",
    2: "SM16703",
    3: "WS2811",
    4: "WS2811B",
    5: "SK6812",
    6: "INK1003",
    7: "WS2801",
    8: "WS2815",
    9: "APA102",
    10: "TM1914",
    11: "UCS2904B",
}
A2_PROTOCOL_TO_NUM = {v: k for k, v in A2_NUM_TO_PROTOCOL.items()}

A2_NUM_TO_OPERATING_MODE = {
    1: COLOR_MODE_RGB,
    2: COLOR_MODE_RGB,
    3: COLOR_MODE_RGB,
    4: COLOR_MODE_RGB,
    5: COLOR_MODE_RGB,
    6: COLOR_MODE_RGB,
    7: COLOR_MODE_RGB,
    8: COLOR_MODE_RGB,
    9: COLOR_MODE_RGB,
    10: COLOR_MODE_RGB,
    11: COLOR_MODE_RGB,
}
A2_OPERATING_MODE_TO_NUM = {v: k for k, v in A2_NUM_TO_OPERATING_MODE.items()}

NEW_ADDRESSABLE_NUM_TO_PROTOCOL = {
    1: "WS2812B",
    2: "SM16703",
    3: "SM16704",
    4: "WS2811",
    5: "UCS1903",
    6: "SK6812",
    7: "SK6812RGBW",
    8: "INK1003",
    9: "UCS2904B",
}
NEW_ADDRESSABLE_PROTOCOL_TO_NUM = {
    v: k for k, v in NEW_ADDRESSABLE_NUM_TO_PROTOCOL.items()
}

NEW_ADDRESSABLE_NUM_TO_OPERATING_MODE = {
    1: COLOR_MODE_RGB,
    2: COLOR_MODE_RGB,
    3: COLOR_MODE_RGB,
    4: COLOR_MODE_RGB,
    5: COLOR_MODE_RGB,
    6: COLOR_MODE_RGB,
    7: COLOR_MODE_RGBW,
    8: COLOR_MODE_RGB,
    9: COLOR_MODE_RGB,
}
NEW_ADDRESSABLE_OPERATING_MODE_TO_NUM = {
    v: k for k, v in NEW_ADDRESSABLE_NUM_TO_OPERATING_MODE.items()
}


class ProtocolBase:
    """The base protocol."""

    power_state_response_length = MSG_LENGTHS[MSG_POWER_STATE]
    level_write_modes = LevelWriteModeData(ALL=0x00, COLORS=0xF0, WHITES=0x0F)

    def __init__(self) -> None:
        self._counter = -1
        super().__init__()

    @property
    def speed_is_delay(self) -> bool:
        """If True the speed is a delay."""
        return True

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return True

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return False

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return False

    @property
    def zones(self) -> bool:
        """If the protocol supports zones."""
        return False

    def _increment_counter(self) -> int:
        """Increment the counter byte."""
        self._counter += 1
        if self._counter == 255:
            self._counter = 0
        return self._counter

    def is_valid_power_restore_state_response(self, msg: bytes) -> bool:
        """Check if a power state response is valid."""
        return (
            _message_type_from_start_of_msg(msg) == MSG_POWER_RESTORE_STATE
            and len(msg) == LEDENET_POWER_RESTORE_RESPONSE_LEN
            and self.is_checksum_correct(msg)
        )

    def is_valid_outer_message(self, data: bytes) -> bool:
        """Check if a message is a valid outer message."""
        if not data.startswith(bytearray(OUTER_MESSAGE_WRAPPER_FIRST_BYTES)):
            return False
        return self.is_checksum_correct(data)

    def extract_inner_message(self, msg: bytes) -> bytes:
        """Extract the inner message from a wrapped message."""
        return msg[10:-1]

    def is_valid_device_config_response(self, data: bytes) -> bool:
        """Check if a message is a valid ic state response."""
        return False

    def expected_response_length(self, data: bytes) -> int:
        """
        Return the number of bytes expected in the response.

        If the response is unknown, we assume the response is
        a complete message since we have no way of knowing otherwise.
        """
        if data[0] == OUTER_MESSAGE_FIRST_BYTE:  # This is a wrapper message
            if len(data) < OUTER_MESSAGE_WRAPPER_START_LEN:
                return OUTER_MESSAGE_WRAPPER_START_LEN
            inner_msg_len = (data[8] << 8) + data[9]
            return (
                OUTER_MESSAGE_WRAPPER_START_LEN  # Includes the two bytes that are the size of the inner message
                + inner_msg_len  # The inner message itself (with checksum)
                + CHECKSUM_LEN  # The checksum of the full message
            )

        msg_type = _message_type_from_start_of_msg(data)
        if msg_type is None:
            return len(data)
        if msg_type == MSG_TIMERS:
            return self.timer_response_len
        return MSG_LENGTHS[msg_type]

    @abstractmethod
    def construct_state_query(self) -> bytearray:
        """The bytes to send for a query request."""

    @abstractmethod
    def is_valid_state_response(self, raw_state: bytes) -> bool:
        """Check if a state response is valid."""

    def is_valid_extended_state_response(self, raw_state: bytes) -> bool:
        return False

    @abstractmethod
    def extended_state_to_state(self, raw_state: bytes) -> bytes:
        """Convert an extended state response to a state response."""

    def is_checksum_correct(self, msg: bytes) -> bool:
        """Check a checksum of a message."""
        expected_sum = sum(msg[0:-1]) & 0xFF
        if expected_sum != msg[-1]:
            _LOGGER.warning(
                "Checksum mismatch: Expected %s, got %s", expected_sum, msg[-1]
            )
            return False
        return True

    @abstractmethod
    def is_valid_power_state_response(self, msg: bytes) -> bool:
        """Check if a power state response is valid."""

    @property
    def on_byte(self) -> int:
        """The on byte."""
        return 0x23

    @property
    def off_byte(self) -> int:
        """The off byte."""
        return 0x24

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return False

    @abstractmethod
    def construct_state_change(self, turn_on: int) -> bytearray:
        """The bytes to send for a state change request."""

    def construct_power_restore_state_query(self) -> bytearray:
        """The bytes to send for a query power restore state."""
        return self.construct_message(bytearray([0x32, 0x3A, 0x3B, 0x0F]))

    def construct_get_time(self) -> bytearray:
        """Construct a get time command."""
        return self.construct_message(bytearray([0x11, 0x1A, 0x1B, 0x0F]))

    def is_valid_get_time_response(self, msg: bytes) -> bool:
        """Check if the response is a valid time response."""
        return (
            _message_type_from_start_of_msg(msg) == MSG_TIME
            and len(msg) == LEDENET_TIME_RESPONSE_LEN
            and self.is_checksum_correct(msg)
        )

    def parse_get_time(self, rx: bytes) -> datetime.datetime | None:
        """Parse a get time command."""
        if self.is_valid_get_time_response(rx):
            with contextlib.suppress(Exception):
                return datetime.datetime(
                    rx[3] + 2000, rx[4], rx[5], rx[6], rx[7], rx[8]
                )
        return None

    def construct_set_time(self, time: datetime.datetime | None) -> bytearray:
        """Construct a set time command."""
        dt = time or datetime.datetime.now()
        return self.construct_message(
            bytearray(
                [
                    0x10,
                    0x14,
                    dt.year - 2000,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                    dt.isoweekday(),  # day of week
                    0x00,
                    0x0F,
                ]
            )
        )

    def construct_get_timers(self) -> bytearray:
        """The bytes to get timers."""
        return self.construct_message(bytearray([0x22, 0x2A, 0x2B, 0x0F]))

    @property
    def timer_response_len(self) -> int:
        """Return the time response len."""
        return LEDENET_TIMERS_8BYTE_RESPONSE_LEN

    @property
    def timer_len(self) -> int:
        """Return a single timer len."""
        return 14

    @property
    def timer_count(self) -> int:
        """Return the number of timers."""
        return 6

    def is_valid_timers_response(self, msg: bytes) -> bool:
        """Check if the response is a valid timers response."""
        return (
            _message_type_from_start_of_msg(msg) == MSG_TIMERS
            and len(msg) == self.timer_response_len
            and self.is_checksum_correct(msg)
        )

    def parse_get_timers(self, msg: bytes) -> list[LedTimer]:
        """Parse get timers."""
        if not self.is_valid_timers_response(msg):
            raise ValueError(f"Timers response not valid: {msg!r}")
        start = 2
        timer_list = []
        timer_bytes_len = self.timer_len
        # pass in the timer_len-byte timer structs
        for _ in range(self.timer_count):
            timer_bytes = msg[start:][:timer_bytes_len]
            timer = LedTimer(timer_bytes)
            timer_list.append(timer)
            start += timer_bytes_len
        return timer_list

    def construct_set_timers(self, timer_list: list[LedTimer]) -> bytearray:
        """Construct a set timers message."""
        # remove inactive or expired timers from list
        for t in timer_list:
            t.length = self.timer_len
            if not t.isActive() or t.isExpired():
                timer_list.remove(t)

        # truncate if more than 6
        if len(timer_list) > self.timer_count:
            _LOGGER.warning("too many timers, truncating list")
            del timer_list[self.timer_count :]

        # pad list to 6 with inactive timers
        if len(timer_list) != self.timer_count:
            for i in range(self.timer_count - len(timer_list)):
                timer_list.append(LedTimer(length=self.timer_len))

        msg = bytearray([0x21])
        for t in timer_list:
            msg.extend(t.toBytes())
        msg.extend(bytearray([0x00, 0xF0]))
        return self.construct_message(msg)

    def construct_power_restore_state_change(
        self, restore_state: PowerRestoreStates
    ) -> bytearray:
        """
        The bytes to send for a power restore state change.

        Set power on state to keep last state
        31f0f0f0f0f0e1
        Set power on state to always on
        310ff0f0f0f000
        Set power on state to always off
        31fff0f0f0f0f0
        """
        return self.construct_message(
            bytearray(
                [
                    0x31,
                    restore_state.channel1.value if restore_state.channel1 else 0x00,
                    restore_state.channel2.value if restore_state.channel2 else 0x00,
                    restore_state.channel3.value if restore_state.channel3 else 0x00,
                    restore_state.channel4.value if restore_state.channel4 else 0x00,
                    0xF0,
                ]
            )
        )

    @abstractmethod
    def construct_music_mode(
        self,
        sensitivity: int,
        brightness: int,
        mode: int | None,
        effect: int | None,
        foreground_color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> list[bytearray]:
        """The bytes to send to set music mode."""

    @abstractmethod
    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""

    @abstractmethod
    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""

    def construct_custom_effect(
        self, rgb_list: list[tuple[int, int, int]], speed: int, transition_type: str
    ) -> bytearray:
        """The bytes to send for a custom effect."""
        msg = bytearray()
        first_color = True
        for rgb in rgb_list:
            if first_color:
                lead_byte = 0x51
                first_color = False
            else:
                lead_byte = 0
            r, g, b = rgb
            msg.extend(bytearray([lead_byte, r, g, b]))
        # pad out empty slots
        if len(rgb_list) != 16:
            for i in range(16 - len(rgb_list)):
                msg.extend(bytearray([0, 1, 2, 3]))
        msg.append(0x00)
        msg.append(utils.speedToDelay(speed))
        msg.append(
            TRANSITION_BYTES.get(transition_type, TRANSITION_BYTES[TRANSITION_GRADUAL])
        )  # default to "gradual"
        msg.append(0xFF)
        msg.append(0x0F)
        return self.construct_message(msg)

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the protocol."""

    @property
    @abstractmethod
    def state_response_length(self) -> int:
        """The length of the query response."""

    @abstractmethod
    def construct_message(self, raw_bytes: bytearray) -> bytearray:
        """Original protocol uses no checksum."""

    def construct_wrapped_message(
        self,
        msg: bytearray,
        inner_pre_constructed: bool = False,
        version: int = 0x01,
    ) -> bytearray:
        """Construct a wrapped message."""
        inner_msg = msg if inner_pre_constructed else self.construct_message(msg)
        inner_msg_len = len(inner_msg)
        return self.construct_message(
            bytearray(
                [
                    *OUTER_MESSAGE_WRAPPER,
                    version,
                    self._increment_counter(),
                    inner_msg_len >> 8,
                    inner_msg_len & 0xFF,
                    *inner_msg,
                ]
            )
        )

    @abstractmethod
    def named_raw_state(
        self, raw_state: bytes
    ) -> LEDENETOriginalRawState | LEDENETRawState:
        """Convert raw_state to a namedtuple."""

    @abstractmethod
    def is_valid_remote_config_response(self, msg: bytes) -> bool:
        """Check if a remote config response is valid."""
        return _message_type_from_start_of_msg(
            msg
        ) == MSG_REMOTE_CONFIG and self.is_checksum_correct(msg)

    def construct_query_remote_config(self) -> bytearray:
        """Construct a remote config query"""
        return self.construct_wrapped_message(bytearray([0x2B, 0x2C, 0x2D]))

    def construct_remote_config(self, remote_config: RemoteConfig) -> bytearray:
        """Construct an remote config."""
        # 2a 02 ff ff ff ff ff 00 00 00 00 00 00 00 0f
        return self.construct_wrapped_message(
            bytearray(
                [
                    0x2A,
                    remote_config.value,
                    0xFF,
                    0xFF,
                    0xFF,
                    0xFF,
                    0xFF,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x0F,
                ]
            )
        )

    def construct_unpair_remotes(self) -> bytearray:
        """Construct an unpair remotes command."""
        return self.construct_wrapped_message(
            bytearray(
                [
                    0x2A,
                    0xFF,
                    0xFF,
                    0x01,
                    0xFF,
                    0xFF,
                    0xFF,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0xF0,
                ]
            )
        )

    def get_write_all_colors(self) -> set[int]:
        return {self.level_write_modes.ALL, self.level_write_modes.COLORS}

    def get_write_all_whites(self) -> set[int]:
        return {self.level_write_modes.ALL, self.level_write_modes.WHITES}


class ProtocolLEDENETOriginal(ProtocolBase):
    """The original LEDENET protocol with no checksums."""

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ORIGINAL

    @property
    def state_response_length(self) -> int:
        """The length of the query response."""
        return LEDENET_ORIGINAL_STATE_RESPONSE_LEN

    def is_valid_power_state_response(self, msg: bytes) -> bool:
        """Check if a power state response is valid."""
        return len(msg) == self.power_state_response_length and msg[0] == 0x78

    def is_valid_state_response(self, raw_state: bytes) -> bool:
        """Check if a state response is valid."""
        return len(raw_state) == self.state_response_length and raw_state[0] == 0x66

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        delay = utils.speedToDelay(speed)
        return self.construct_message(bytearray([0xBB, pattern, delay, 0x44]))

    def construct_state_query(self) -> bytearray:
        """The bytes to send for a query request."""
        return self.construct_message(bytearray([0xEF, 0x01, 0x77]))

    def construct_state_change(self, turn_on: int) -> bytearray:
        """The bytes to send for a state change request."""
        return self.construct_message(
            bytearray([0xCC, self.on_byte if turn_on else self.off_byte, 0x33])
        )

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""
        # sample message for original LEDENET protocol (w/o checksum at end)
        #  0  1  2  3  4
        # 56 90 fa 77 aa
        #  |  |  |  |  |
        #  |  |  |  |  terminator
        #  |  |  |  blue
        #  |  |  green
        #  |  red
        #  head
        return [
            self.construct_message(
                bytearray([0x56, red or 0x00, green or 0x00, blue or 0x00, 0xAA])
            )
        ]

    def construct_message(self, raw_bytes: bytearray) -> bytearray:
        """Original protocol uses no checksum."""
        return raw_bytes

    def named_raw_state(self, raw_state: bytes) -> LEDENETOriginalRawState:
        """Convert raw_state to a namedtuple."""
        raw_bytearray = bytearray([*raw_state, 0])
        return LEDENETOriginalRawState(*raw_bytearray)


class ProtocolLEDENETOriginalRGBW(ProtocolLEDENETOriginal):
    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ORIGINAL_RGBW

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""
        # sample message for original LEDENET RGBW protocol (w/o checksum at end)
        return [
            self.construct_message(
                bytearray(
                    [
                        0x56,
                        red or 0x00,
                        green or 0x00,
                        blue or 0x00,
                        warm_white or 0x00,
                        write_mode.value
                        if isinstance(write_mode, LevelWriteMode)
                        else write_mode,
                        0xAA,
                    ]
                )
            )
        ]


class ProtocolLEDENETOriginalCCT(ProtocolLEDENETOriginal):
    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ORIGINAL_CCT

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""
        # sample message for original LEDENET protocol (w/o checksum at end)
        #  0  1  2  3  4
        # 56 90 fa 77 aa
        #  |  |  |  |  |
        #  |  |  |  |  terminator
        #  |  |  |  blue
        #  |  |  green
        #  |  red
        #  head
        return [
            self.construct_message(bytearray([0x56, red or 0x00, green or 0x00, 0xAA]))
        ]


class ProtocolLEDENET8Byte(ProtocolBase):
    """The newer LEDENET protocol with checksums that uses 8 bytes to set state."""

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_8BYTE

    @property
    def state_response_length(self) -> int:
        """The length of the query response."""
        return LEDENET_STATE_RESPONSE_LEN

    def is_valid_power_state_response(self, msg: bytes) -> bool:
        """Check if a power state response is valid."""
        if (
            len(msg) != self.power_state_response_length
            or not self._is_start_of_power_state_response(msg)
            or msg[1] != 0x71
            or msg[2] not in (self.on_byte, self.off_byte)
        ):
            return False
        return True  # checksum does not always match

    def _is_start_of_power_state_response(self, data: bytes) -> bool:
        """Check if a message is the start of a state response."""
        return _message_type_from_start_of_msg(data) == MSG_POWER_STATE

    def is_valid_state_response(self, raw_state: bytes) -> bool:
        """Check if a state response is valid."""
        if len(raw_state) != self.state_response_length:
            return False
        if raw_state[0] != 129:
            return False
        return self.is_checksum_correct(raw_state)

    def construct_state_change(self, turn_on: int) -> bytearray:
        """
        The bytes to send for a state change request.

        Alternate messages

        Off 3b 24 00 00 00 00 00 00 00 32 00 00 91
        On  3b 23 00 00 00 00 00 00 00 32 00 00 90
        """
        return self.construct_message(
            bytearray([0x71, self.on_byte if turn_on else self.off_byte, 0x0F])
        )

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        delay = utils.speedToDelay(speed)
        return self.construct_message(bytearray([0x61, pattern, delay, 0x0F]))

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""
        # sample message for 8-byte protocols (w/ checksum at end)
        #  0  1  2  3  4  5  6
        # 31 90 fa 77 00 00 0f
        #  |  |  |  |  |  |  |
        #  |  |  |  |  |  |  terminator
        #  |  |  |  |  |  write mask / white2 (see below)
        #  |  |  |  |  white
        #  |  |  |  blue
        #  |  |  green
        #  |  red
        #  persistence (31 for true / 41 for false)
        #
        # byte 5 can have different values depending on the type
        # of device:
        # For devices that support 2 types of white value (warm and cold
        # white) this value is the cold white value. These use the LEDENET
        # protocol. If a second value is not given, reuse the first white value.
        #
        # For devices that cannot set both rbg and white values at the same time
        # (including devices that only support white) this value
        # specifies if this command is to set white value (0f) or the rgb
        # value (f0).
        #
        # For all other rgb and rgbw devices, the value is 00
        return [
            self.construct_message(
                bytearray(
                    [
                        0x31 if persist else 0x41,
                        red or 0x00,
                        green or 0x00,
                        blue or 0x00,
                        warm_white or 0x00,
                        write_mode.value
                        if isinstance(write_mode, LevelWriteMode)
                        else write_mode,
                        0x0F,
                    ]
                )
            )
        ]

    def construct_message(self, raw_bytes: bytearray) -> bytearray:
        """Calculate checksum of byte array and add to end."""
        csum = sum(raw_bytes) & 0xFF
        raw_bytes.append(csum)
        return raw_bytes

    def construct_state_query(self) -> bytearray:
        """The bytes to send for a query request."""
        return self.construct_message(bytearray([0x81, 0x8A, 0x8B]))

    def named_raw_state(self, raw_state: bytes) -> LEDENETRawState:
        """Convert raw_state to a namedtuple."""
        return LEDENETRawState(*raw_state)

    def construct_music_mode(
        self,
        sensitivity: int,
        brightness: int,
        mode: int | None,
        effect: int | None,
        foreground_color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> list[bytearray]:
        """
        The bytes to send for music mode.

        Known messages
        73 01 4d 0f d0
              ^^
              Likely sensitivity from 0-100 (0x64)
        73 01 64 0f e7
        73 01 4a 0f cd
        73 01 4b 0f ce
        73 01 00 0f 83
        73 01 1b 0f 9e
        73 01 05 0f 88
        73 01 02 0f 85
        73 01 06 0f 89
        73 01 05 0f 88
        73 01 10 0f 93
        73 01 4d 0f d0
        73 01 64 0f e7

        Pause music mode
        73 00 59 0f db
           ^^
           On/off byte

        Mic
        37 00 00 37  Fade In
           ^^
           Mic effect
        37 01 00 38  Gradual
        37 02 00 39  Jump
        37 03 00 3a  Strobe
        """
        # Valid modes for old protocol
        # 0x01 - Gradual
        return [self.construct_message(bytearray([0x73, 0x01, sensitivity, 0x0F]))]

    def construct_device_config(
        self,
        operating_mode: int | None,
        wiring: int | None,
        ic_type: int | None,  # ic type
        pixels_per_segment: int | None,  # pixels per segment
        segments: int | None,  # number of segments
        music_pixels_per_segment: int | None,  # music pixels per segment
        music_segments: int | None,  # number of music segments
    ) -> bytearray:
        """
        The bytes to send to change device config.

        RGBW 0x06
        62 06 02 0f 79 - RGB/W GRB W
        62 04 02 0f 77 - RGB&W GRB W
        62 04 01 0f 77 - RGB&W RGB W
        62 04 03 0f 77 - RGB&W BRG W

        RGBCW 0x07

        62 05 0f 0f 85 - RGB&CCT / WCBRG
        62 07 0f 0f 87 - RGB/CCT / WCBRG
        62 07 01 0f 79 - RGB/CCT / RGBCW
        62 07 02 0f 7a - RGB/CCT / GRBCW
        62 07 0c 0f 84 - RGB/CCT / CBRGW

        RGB 0x33 / 0x08

        62 00 01 0f 73 - RGB
        62 00 02 0f 73 - GRB
        62 00 03 0f 73 - BRG

        0x25
        62 01 0f 72 - DIM
        62 02 0f 73 - CCT
        62 03 0f 74 - RGB
        62 04 0f 74 - RGB&W
        62 05 0f 74 - RGB&CCT
        """
        msg = bytearray([0x62, operating_mode or 0x00])
        if wiring:
            msg.append(wiring)
        msg.append(0x0F)
        return self.construct_message(msg)


class ProtocolLEDENET8ByteAutoOn(ProtocolLEDENET8Byte):
    """Protocol that uses 8 bytes, and turns on by changing levels or effects."""

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_8BYTE_AUTO_ON


# This protocol also supports Candle mode but its not currently implemented here
class ProtocolLEDENET8ByteDimmableEffects(ProtocolLEDENET8ByteAutoOn):
    """Protocol that uses 8 bytes, and supports dimmable effects and auto on by changing levels or effects."""

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return True

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return True

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        delay = utils.speedToDelay(speed)
        return self.construct_message(bytearray([0x38, pattern, delay, brightness]))

    def construct_music_mode(
        self,
        sensitivity: int,
        brightness: int,
        mode: int | None,
        effect: int | None,
        foreground_color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> list[bytearray]:
        """
        The bytes to send for music mode.

        Known messages
        73 01 4d 0f d0
              ^^
              Likely sensitivity from 0-100 (0x64)
        73 01 64 0f e7
        73 01 4a 0f cd
        73 01 4b 0f ce
        73 01 00 0f 83
        73 01 1b 0f 9e
        73 01 05 0f 88
        73 01 02 0f 85
        73 01 06 0f 89
        73 01 05 0f 88
        73 01 10 0f 93
        73 01 4d 0f d0
        73 01 64 0f e7

        Pause music mode
        73 00 59 0f db
           ^^
           On/off byte

        Mic
        37 00 00 37  Fade In
           ^^
           Mic effect
        37 01 00 38  Gradual
        37 02 00 39  Jump
        37 03 00 3a  Strobe
        """
        # Valid effect
        # 0x00 - Fade In
        # 0x01 - Gradual
        # 0x02 - Jump
        # 0x03 - Strobe
        if effect and not (0x00 <= effect <= 0x03):
            raise ValueError(
                "Mode must be one of (0x00 - Fade In, 0x01 - Gradual, 0x02 - Jump, 0x03 - Strobe)"
            )
        return [
            self.construct_message(bytearray([0x73, 0x01, sensitivity, 0x0F])),
            self.construct_message(bytearray([0x37, effect or 0x00, 0x00])),
        ]


class ProtocolLEDENET9Byte(ProtocolLEDENET8Byte):
    """The newer LEDENET protocol with checksums that uses 9 bytes to set state."""

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_9BYTE

    @property
    def timer_response_len(self) -> int:
        """Return the time response len."""
        return LEDENET_TIMERS_9BYTE_RESPONSE_LEN

    @property
    def timer_len(self) -> int:
        """Return a single timer len."""
        return 15

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""
        # sample message for 9-byte LEDENET protocol (w/ checksum at end)
        #  0  1  2  3  4  5  6  7
        # 31 bc c1 ff 00 00 f0 0f
        #  |  |  |  |  |  |  |  |
        #  |  |  |  |  |  |  |  terminator
        #  |  |  |  |  |  |  write mode (f0 colors, 0f whites, 00 colors & whites)
        #  |  |  |  |  |  cold white
        #  |  |  |  |  warm white
        #  |  |  |  blue
        #  |  |  green
        #  |  red
        #  persistence (31 for true / 41 for false)
        #
        return [
            self.construct_message(
                bytearray(
                    [
                        0x31 if persist else 0x41,
                        red or 0x00,
                        green or 0x00,
                        blue or 0x00,
                        warm_white or 0x00,
                        cool_white or 0x00,
                        write_mode.value
                        if isinstance(write_mode, LevelWriteMode)
                        else write_mode,
                        0x0F,
                    ]
                )
            )
        ]


class ProtocolLEDENET9ByteAutoOn(ProtocolLEDENET9Byte):
    """Protocol that uses 9 bytes, and turns on by changing levels or effects."""

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_9BYTE_AUTO_ON


# This protocol also supports Candle mode but its not currently implemented here
class ProtocolLEDENET9ByteDimmableEffects(ProtocolLEDENET9ByteAutoOn):
    """The newer LEDENET protocol with checksums that uses 9 bytes to set state."""

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return True

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return True

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_9BYTE_DIMMABLE_EFFECTS

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        delay = utils.speedToDelay(speed)
        return self.construct_message(bytearray([0x38, pattern, delay, brightness]))


class ProtocolLEDENET25Byte(ProtocolLEDENET9Byte):
    """25 byte protocol"""

    level_write_modes = LevelWriteModeData(ALL=0x00, COLORS=0xA1, WHITES=0xB1)

    @property
    def speed_is_delay(self) -> bool:
        """If True the speed is a delay."""
        return False

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_25BYTE

    def is_valid_extended_state_response(self, raw_state: bytes) -> bool:
        """Check if a state response is valid."""
        return raw_state[0] == 0xEA and raw_state[1] == 0x81 and len(raw_state) >= 20

    def extended_state_to_state(self, raw_state: bytes) -> bytes:
        """Convert an extended state response to a state response."""
        # pos  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
        #     EA  81  01  10  35  0A  23  61  01  50  0F  3C  64  64  00  64  00  00  00  00
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ??
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ??
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ??
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   ??
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   |   White brightness
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   |   White temperature
        #      |   |   |   |   |   |   |   |   |   |   |   |   |   Value
        #      |   |   |   |   |   |   |   |   |   |   |   |   Saturation
        #      |   |   |   |   |   |   |   |   |   |   |   Hue / 2 (0-180)
        #      |   |   |   |   |   |   |   |   |   |   0f white / f0 rgb
        #      |   |   |   |   |   |   |   |   |   Speed?
        #      |   |   |   |   |   |   |   |   w 01 / rgb 00
        #      |   |   |   |   |   |   |   ??
        #      |   |   |   |   |   |   Power state (0x23 = ON, 0x24 = OFF)
        #      |   |   |   |   |   ??
        #      |   |   |   |   Version number
        #      |   |   |   Model number
        #      |   |   Unknown / reserved
        #      |   Unknown / reserved
        #   Extended message header (ea 81)

        if len(raw_state) < 20:
            return b""

        model_num = raw_state[4]
        version_number = raw_state[5]
        power_state = raw_state[6]
        preset_pattern = raw_state[7]
        speed = raw_state[9]

        hue = raw_state[11]
        saturation = raw_state[12]
        value = raw_state[13]

        white_temp = raw_state[14]
        white_brightness = raw_state[15]
        levels = scaled_color_temp_to_white_levels(white_temp, white_brightness)

        cool_white = levels.cool_white
        warm_white = levels.warm_white

        # Convert HSV to RGB
        h = (hue * 2) / 360
        s = saturation / 100
        v = value / 100
        r_f, g_f, b_f = colorsys.hsv_to_rgb(h, s, v)
        red = min(int(max(0, r_f) * 255), 255)
        green = min(int(max(0, g_f) * 255), 255)
        blue = min(int(max(0, b_f) * 255), 255)

        # Fill standard state structure
        mode = 0
        color_mode = 0
        check_sum = 0  # Set to 0; not critical

        return bytes(
            (
                raw_state[1],  # Head (second byte of EA 81)
                model_num,
                power_state,
                preset_pattern,
                mode,
                speed,
                red,
                green,
                blue,
                warm_white,
                version_number,
                cool_white,
                color_mode,
                check_sum,
            )
        )

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,  # 0-255
        green: int | None,  # 0-255
        blue: int | None,  # 0-255
        warm_white: int | None,  # 0-255
        cool_white: int | None,  # 0-255
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """The bytes to send for a level change request."""
        # sample message for 25-byte LEDENET protocol (w/ checksum at end)
        #  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
        # b0 b1 b2 b3 00 01 02 79 00 0e e0 01 00 a1 78 64 64 00 00 00 00 14 00 00
        #                    |  |              |  |  |  |  |  |  |              |
        #                    |  |              |  |  |  |  |  |  |              footer
        #                    |  |              |  |  |  |  |  |  white brightness
        #                    |  |              |  |  |  |  |  white temperature (00 warm - 64 cool)
        #                    |  |              |  |  |  |  value
        #                    |  |              |  |  |  saturation
        #                    |  |              |  |  hue / 2
        #                    |  |              |  write mode (a1 color, b1 white)
        #                    |  |              unknown filler
        #                    |  increment counter
        #                    header

        if red is not None and green is not None and blue is not None:
            h, s, v = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            h = int((h * 360) / 2)  # Hue needs to be halved
            s = int(s * 100)
            v = int(v * 100)
        else:
            h = s = v = 0x00

        if (
            cool_white is None
            or warm_white is None
            or (cool_white == 0 and warm_white == 0)
        ):
            white_temp = white_brightness = 0
        else:
            total = warm_white + cool_white
            # temperature: ratio of cool to total, scaled to 0-100
            white_temp = round((cool_white / float(total)) * 100)
            # brightness: clamp sum at 255, then scale to 0-100
            clamped_sum = min(total, 255)
            white_brightness = round((clamped_sum / 255.0) * 100)

        return [
            self.construct_wrapped_message(
                bytearray(
                    [
                        0xE0,
                        0x01,
                        0x00,
                        write_mode.value
                        if isinstance(write_mode, LevelWriteMode)
                        else write_mode,
                        h,
                        s,
                        v,
                        white_temp,
                        white_brightness,
                        0x00,
                        0x00,
                        0x14,
                        0x00,
                        0x00,
                    ]
                ),
                inner_pre_constructed=True,
                version=0x02,
            )
        ]


class ProtocolLEDENETAddressableBase(ProtocolLEDENET9Byte):
    """Base class for addressable protocols."""

    @property
    def timer_response_len(self) -> int:
        """Return the time response len."""
        return LEDENET_TIMERS_8BYTE_RESPONSE_LEN

    @property
    def timer_len(self) -> int:
        """Return a single timer len."""
        return 14

    @property
    def speed_is_delay(self) -> bool:
        """If True the speed is a delay."""
        return False


class ProtocolLEDENETAddressableA1(ProtocolLEDENETAddressableBase):
    def construct_request_strip_setting(self) -> bytearray:
        return bytearray([0x63, 0x12, 0x21, 0x36])

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ADDRESSABLE_A1

    def is_valid_device_config_response(self, data: bytes) -> bool:
        """Check if a message is a valid ic state response."""
        return (
            len(data) == LEDENET_A1_DEVICE_CONFIG_RESPONSE_LEN
            and _message_type_from_start_of_msg(data) == MSG_A1_DEVICE_CONFIG
            and self.is_checksum_correct(data)
        )

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return False

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        effect = pattern + 99
        return self.construct_message(
            bytearray([0x61, effect >> 8, effect & 0xFF, speed, 0x0F])
        )

    def parse_strip_setting(self, msg: bytes) -> LEDENETAddressableDeviceConfiguration:
        """Parse a strip settings message."""
        # pos  0  1  2  3  4  5  6  7  8  9 10 11
        #    63 00 32 05 00 00 00 00 00 00 02 9c
        #     |  |  |  |  |  |  |  |  |  |  |  |
        #     |  |  |  |  |  |  |  |  |  |  |  checksum
        #     |  |  |  |  |  |  |  |  |  |  wiring type (0 indexed, RGB or RGBW)
        #     |  |  |  |  |  |  |  |  |  ?? always 00
        #     |  |  |  |  |  |  |  |  ?? always 00
        #     |  |  |  |  |  |  |  n?? always 00
        #     |  |  |  |  |  |  ?? always 00
        #     |  |  |  |  |  ?? always 00
        #     |  |  |  |  ?? always 00
        #     |  |  |  ic type (01=UCS1903, 02=SM16703, 03=WS2811, 04=WS2812B, 05=SK6812, 06=INK1003, 07=WS2801, 08=LB1914)
        #     |  |  num pixels (16 bit, low byte)
        #     |  num pixels (16 bit, high byte)
        #     msg head
        #
        high_byte = msg[1]
        low_byte = msg[2]
        pixels_per_segment = (high_byte << 8) + low_byte
        _LOGGER.debug(
            "Pixel count (high: %s, low: %s) is: %s",
            hex(high_byte),
            hex(low_byte),
            pixels_per_segment,
        )
        return LEDENETAddressableDeviceConfiguration(
            pixels_per_segment=pixels_per_segment,
            segments=None,
            music_pixels_per_segment=None,
            music_segments=None,
            wirings=list(ADDRESSABLE_RGB_WIRING_TO_NUM),
            wiring_num=msg[10],
            wiring=ADDRESSABLE_RGB_NUM_TO_WIRING.get(msg[10]),
            num_to_wiring=ADDRESSABLE_RGB_NUM_TO_WIRING,
            wiring_to_num=ADDRESSABLE_RGB_WIRING_TO_NUM,
            ic_type=A1_NUM_TO_PROTOCOL.get(msg[3]),
            ic_type_num=msg[3],
            operating_mode=A1_NUM_TO_OPERATING_MODE.get(msg[3]),
        )

    def construct_device_config(
        self,
        operating_mode: int | None,
        wiring: int | None,
        ic_type: int | None,  # ic type
        pixels_per_segment: int | None,  # pixels per segment
        segments: int | None,  # number of segments
        music_pixels_per_segment: int | None,  # music pixels per segment
        music_segments: int | None,  # number of music segments
    ) -> bytearray:
        """
        The bytes to send to change device config.
        pos  0  1  2  3  4  5  6  7  8  9 10 11 12
            62 04 00 04 00 00 00 00 00 00 02 f0 5c <- checksum
             |  |  |  |  |  |  |  |  |  |  |  |
             |  |  |  |  |  |  |  |  |  |  |  always 0xf0
             |  |  |  |  |  |  |  |  |  |  wiring type (0 indexed, RGB or RGBW)
             |  |  |  |  |  |  |  |  |  ?? always 00
             |  |  |  |  |  |  |  |  ?? always 00
             |  |  |  |  |  |  |  n?? always 00
             |  |  |  |  |  |  ?? always 00
             |  |  |  |  |  ?? always 00
             |  |  |  |  ?? always 00
             |  |  |  ic type (01=UCS1903, 02=SM16703, 03=WS2811, 04=WS2812B, 05=SK6812, 06=INK1003, 07=WS2801, 08=LB1914)
             |  |  num pixels (16 bit, low byte)
             |  num pixels (16 bit, high byte)
             msg head

        """
        assert ic_type is not None
        assert pixels_per_segment is not None
        assert wiring is not None
        return self.construct_message(
            bytearray(
                [
                    0x62,
                    pixels_per_segment >> 8,
                    pixels_per_segment & 0xFF,
                    ic_type,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    0x00,
                    wiring,
                    0xF0,
                ]
            )
        )


class ProtocolLEDENETAddressableA2(ProtocolLEDENETAddressableBase):
    # ic response
    # 0x96 0x63 0x00 0x32 0x00 0x01 0x01 0x04 0x32 0x01 0x64 (11)
    def construct_request_strip_setting(self) -> bytearray:
        return self.construct_message(bytearray([0x63, 0x12, 0x21, 0x0F]))

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ADDRESSABLE_A2

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        # This is likely due to buggy firmware
        return False

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return True

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    def is_valid_device_config_response(self, data: bytes) -> bool:
        """Check if a message is a valid ic state response."""
        return (
            len(data) == LEDENET_DEVICE_CONFIG_RESPONSE_LEN
            and _message_type_from_start_of_msg(data) == MSG_DEVICE_CONFIG
            and self.is_checksum_correct(data)
        )

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        return self.construct_message(bytearray([0x42, pattern, speed, brightness]))

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """
        The bytes to send for a level change request.

        white  41 01 ff ff ff 00 00 00 60 ff 00 00 9e
        """
        preset_number = 0x01  # aka fixed color
        msgs = []
        if red is not None or green is not None or blue is not None:
            msgs.append(
                self.construct_message(
                    bytearray(
                        [
                            0x41,
                            preset_number,
                            red or 0x00,
                            green or 0x00,
                            blue or 0x00,
                            0x00,
                            0x00,
                            0x00,
                            0x60,
                            0xFF,
                            0x00,
                            0x00,
                        ]
                    )
                )
            )
        if warm_white is not None:
            msgs.append(self.construct_message(bytearray([0x47, warm_white or 0x00])))
        return msgs

    def construct_music_mode(
        self,
        sensitivity: int,
        brightness: int,
        mode: int | None,
        effect: int | None,
        foreground_color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> list[bytearray]:
        """
        The bytes to send for music mode.

        Known messages
        73 01 27 01 00 00 00 00 ff ff 64 64 62 - lowest brightness music
        73 01 27 01 ff ff ff 00 ff ff 64 64 5f - highest brightness music
                    ^R ^G ^B <-- failling color
        73 01 27 01 ff 00 00 00 ff ff 64 64 61
                    ^R ^G ^B <-- failling color


        73 01 27 01 ff ff ff 00 ff ff 00 64 fb - lowest sensitivity
        73 01 27 01 ff ff ff 00 ff ff 64 64 5f - highest sensitivity
                                      ^ sensitivity


        73 01 27 13 00 ff 19 ff 00 00 64 64 8d
                    ^R ^G ^B <-- failling color (light screen mode)
        73 01 27 13 00 ff 19 ff 00 00 64 64 8d
                             ^R ^G ^B <-- column color (light screen mode)

        73 01 27 14 00 ff 19 ff 00 00 64 64 8e
                 ^ effect
        73 01 27 15 00 ff 19 ff 00 00 64 64 8f
                 ^ effect

        73 01 27 15 00 ff 19 ff 00 00 64 64 8f
              ^ mode - light screen mode
        73 01 26 01 00 00 00 00 ff ff 64 64 61
              ^ mode - led strip mode

        73 01 26 0e 00 00 00 ff 00 00 64 64 6f
                             ^R ^G ^B <-- led strip mode color

        73 01 26 0e 00 00 00 ff 00 00 64 06 11
                                         ^brightness <-- led strip mode color

        """
        if foreground_color is None:
            foreground_color = (0xFF, 0xFF, 0xFF)
        if background_color is None:
            background_color = (0x00, 0x00, 0x00)
        if effect and not (1 <= effect <= 16):
            raise ValueError("Effect must be between 1 and 16")
        if mode and not (0x26 <= mode <= 0x27):
            raise ValueError("Mode must be between 0x26 and 0x27")

        return [
            self.construct_message(
                bytearray(
                    [
                        0x73,
                        0x01,
                        mode
                        or MusicMode.STRIP.value,  # strip mode 0x26, light screen mode 0x27
                        effect or 0x01,
                        *foreground_color,
                        *background_color,
                        sensitivity,
                        brightness,
                    ]
                )
            )
        ]

    def parse_strip_setting(self, msg: bytes) -> LEDENETAddressableDeviceConfiguration:
        """Parse a strip settings message."""
        # pos  0  1  2  3  4  5  6  7  8  9 10
        #    00 63 01 2c 00 01 07 08 96 01 45
        #     |  |  |  |  |  |  |  |  |  |  |
        #     |  |  |  |  |  |  |  |  |  |  checksum
        #     |  |  |  |  |  |  |  |  |  |
        #     |  |  |  |  |  |  |  |  |  segments (music mode)
        #     |  |  |  |  |  |  |  |  num pixels (music mode)
        #     |  |  |  |  |  |  |  wiring type (0 indexed, RGB or RGBW)
        #     |  |  |  |  |  |  ic type (01=UCS1903, 02=SM16703, 03=WS2811, 04=WS2811B, 05=SK6812, 06=INK1003, 07=WS2801, 08=WS2815, 09=APA102, 10=TM1914, 11=UCS2904B)
        #     |  |  |  |  |  segments
        #     |  |  |  |  ?? (always 0x00)
        #     |  |  |  num pixels (16 bit, low byte)
        #     |  |  num pixels (16 bit, high byte)
        #     |  msg head
        #     msg head
        #
        high_byte = msg[2]
        low_byte = msg[3]
        pixels_per_segment = (high_byte << 8) + low_byte
        _LOGGER.debug("bytes: %s", msg)
        _LOGGER.debug(
            "Pixel count (high: %s, low: %s) is: %s",
            hex(high_byte),
            hex(low_byte),
            pixels_per_segment,
        )
        segments = msg[5]
        _LOGGER.debug(
            "Segment count (%s) is: %s",
            hex(segments),
            segments,
        )
        return LEDENETAddressableDeviceConfiguration(
            pixels_per_segment=pixels_per_segment,
            segments=segments,
            music_pixels_per_segment=msg[8],
            music_segments=msg[9],
            wirings=list(ADDRESSABLE_RGB_NUM_TO_WIRING.values()),
            wiring_num=msg[7],
            wiring=ADDRESSABLE_RGB_NUM_TO_WIRING.get(msg[7]),
            num_to_wiring=ADDRESSABLE_RGB_NUM_TO_WIRING,
            wiring_to_num=ADDRESSABLE_RGB_WIRING_TO_NUM,
            ic_type=A2_NUM_TO_PROTOCOL.get(msg[6]),
            ic_type_num=msg[6],
            operating_mode=A2_NUM_TO_OPERATING_MODE.get(msg[6]),
        )

    def construct_device_config(
        self,
        operating_mode: int | None,
        wiring: int | None,
        ic_type: int | None,  # ic type
        pixels_per_segment: int | None,  # pixels per segment
        segments: int | None,  # number of segments
        music_pixels_per_segment: int | None,  # music pixels per segment
        music_segments: int | None,  # number of music segments
    ) -> bytearray:
        """
        The bytes to send to change device config.
        pos  0  1  2  3  4  5  6  7  8  9 10
            62 01 2c 00 06 01 04 32 01 0f dc
             |  |  |  |  |  |  |  |  |  |  |
             |  |  |  |  |  |  |  |  |  |  |
             |  |  |  |  |  |  |  |  |  |  checksum
             |  |  |  |  |  |  |  |  |  ?? always 0x0f
             |  |  |  |  |  |  |  |  segments (music mode)
             |  |  |  |  |  |  |  num pixels (music mode)
             |  |  |  |  |  |  wiring type (0 indexed, RGB or RGBW)
             |  |  |  |  |  ic type (01=WS2812B, 02=SM16703, 03=SM16704, 04=WS2811, 05=UCS1903, 06=SK6812, 07=SK6812RGBW, 08=INK1003, 09=UCS2904B)
             |  |  |  |  segments
             |  |  |  ?? always 00
             |  |  num pixels (16 bit, low byte)
             |  num pixels (16 bit, high byte)
             msg head


        """
        assert ic_type is not None
        assert pixels_per_segment is not None
        assert segments is not None
        assert music_pixels_per_segment is not None
        assert music_segments is not None
        assert wiring is not None
        pixels_per_segment = max(1, min(pixels_per_segment, PIXELS_PER_SEGMENT_MAX))
        segments = max(1, min(segments, SEGMENTS_MAX))
        if pixels_per_segment * segments > PIXELS_MAX:
            segments = int(PIXELS_MAX / pixels_per_segment)
        music_pixels_per_segment = max(
            1, min(music_pixels_per_segment, MUSIC_PIXELS_PER_SEGMENT_MAX)
        )
        music_segments = max(1, min(music_segments, MUSIC_SEGMENTS_MAX))
        if music_pixels_per_segment * music_segments > MUSIC_PIXELS_MAX:
            music_segments = int(MUSIC_PIXELS_MAX / music_pixels_per_segment)
        if (
            pixels_per_segment <= MUSIC_PIXELS_PER_SEGMENT_MAX
            and segments <= MUSIC_SEGMENTS_MAX
            and pixels_per_segment * segments <= MUSIC_PIXELS_MAX
        ):
            # If the pixels_per_segment and segments can accomate music
            # mode then we sync them
            music_pixels_per_segment = pixels_per_segment
            music_segments = segments
        return self.construct_message(
            bytearray(
                [
                    0x62,
                    pixels_per_segment >> 8,
                    pixels_per_segment & 0xFF,
                    0x00,
                    segments,
                    ic_type,
                    wiring,
                    music_pixels_per_segment,
                    music_segments,
                    0xF0,
                ]
            )
        )


class ProtocolLEDENETAddressableA3(ProtocolLEDENETAddressableA2):
    def construct_request_strip_setting(self) -> bytearray:
        return self.construct_wrapped_message(
            super().construct_request_strip_setting(),
            inner_pre_constructed=True,
        )

    def construct_state_query(self) -> bytearray:
        """The bytes to send for a query request."""
        return self.construct_wrapped_message(
            super().construct_state_query(),
            inner_pre_constructed=True,
        )

    def construct_state_change(self, turn_on: int) -> bytearray:
        """The bytes to send for a state change request."""
        return self.construct_wrapped_message(
            super().construct_state_change(turn_on),
            inner_pre_constructed=True,
        )

    # ic response
    # 0x00 0x63 0x00 0x32 0x00 0x01 0x04 0x03 0x32 0x01 0xD0 (11)
    # b0 b1 b2 b3 00 01 01 37 00 0b 00 63 00 32 00 01 04 03 32 01 d0 aa
    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return True

    @property
    def zones(self) -> bool:
        """If the protocol supports zones."""
        return True

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ADDRESSABLE_A3

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return True

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """The bytes to send for a preset pattern."""
        return self.construct_wrapped_message(
            super().construct_preset_pattern(pattern, speed, brightness),
            inner_pre_constructed=True,
        )

    def parse_strip_setting(self, msg: bytes) -> LEDENETAddressableDeviceConfiguration:
        """Parse a strip settings message."""
        # pos  0  1  2  3  4  5  6  7  8  9 10
        #    00 63 01 2c 00 01 07 08 96 01 45
        #     |  |  |  |  |  |  |  |  |  |  |
        #     |  |  |  |  |  |  |  |  |  |  checksum
        #     |  |  |  |  |  |  |  |  |  |
        #     |  |  |  |  |  |  |  |  |  segments (music mode)
        #     |  |  |  |  |  |  |  |  num pixels (music mode)
        #     |  |  |  |  |  |  |  wiring type (0 indexed, RGB or RGBW)
        #     |  |  |  |  |  |  ic type (01=WS2812B, 02=SM16703, 03=SM16704, 04=WS2811, 05=UCS1903, 06=SK6812, 07=SK6812RGBW, 08=INK1003, 09=UCS2904B)
        #     |  |  |  |  |  segments
        #     |  |  |  |  ?? (always 0x00)
        #     |  |  |  num pixels (16 bit, low byte)
        #     |  |  num pixels (16 bit, high byte)
        #     |  msg head
        #     msg head
        #
        high_byte = msg[2]
        low_byte = msg[3]
        pixels_per_segment = (high_byte << 8) + low_byte
        _LOGGER.debug("bytes: %s", msg)
        _LOGGER.debug(
            "Pixel count (high: %s, low: %s) is: %s",
            hex(high_byte),
            hex(low_byte),
            pixels_per_segment,
        )
        segments = msg[5]
        _LOGGER.debug(
            "Segment count (%s) is: %s",
            hex(segments),
            segments,
        )
        if NEW_ADDRESSABLE_NUM_TO_OPERATING_MODE.get(msg[6]) == COLOR_MODE_RGBW:
            wirings = ADDRESSABLE_RGBW_NUM_TO_WIRING
            num_to_wiring = ADDRESSABLE_RGBW_NUM_TO_WIRING
            wiring_to_num = ADDRESSABLE_RGBW_WIRING_TO_NUM
        else:
            wirings = ADDRESSABLE_RGB_NUM_TO_WIRING
            num_to_wiring = ADDRESSABLE_RGB_NUM_TO_WIRING
            wiring_to_num = ADDRESSABLE_RGB_WIRING_TO_NUM
        return LEDENETAddressableDeviceConfiguration(
            pixels_per_segment=pixels_per_segment,
            segments=segments,
            music_pixels_per_segment=msg[8],
            music_segments=msg[9],
            wirings=list(wirings.values()),
            wiring_num=msg[7],
            wiring=wirings.get(msg[7]),
            ic_type=NEW_ADDRESSABLE_NUM_TO_PROTOCOL.get(msg[6]),
            num_to_wiring=num_to_wiring,
            wiring_to_num=wiring_to_num,
            ic_type_num=msg[6],
            operating_mode=NEW_ADDRESSABLE_NUM_TO_OPERATING_MODE.get(msg[6]),
        )

    # To query music mode
    # Send     -> b0 b1 b2 b3 00 01 01 1c 00 03 72 00 72 cb
    # Responds <- b0 b1 b2 b3 00 01 01 1c 00 0d 72 01 26 01 00 00 00 00 00 00 64 64 62 b5

    def construct_music_mode(
        self,
        sensitivity: int,
        brightness: int,
        mode: int | None,
        effect: int | None,
        foreground_color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> list[bytearray]:
        """
        The bytes to send for music mode.

        Known messages
        b0 b1 b2 b3 00 01 01 1f 00 0d 73 01 27 01 ff 00 00 ff 00 00 64 64 62 b8 - Music mode
        b0 b1 b2 b3 00 01 01 20 00 0d 73 01 27 01 00 ff 44 ff 00 00 64 64 a6 41 - Music mode
        b0 b1 b2 b3 00 01 01 21 00 0d 73 01 27 01 ff a6 00 ff 00 00 64 64 08 06 - Music mode
        b0 b1 b2 b3 00 01 01 22 00 0d 73 01 27 01 ff a6 00 ff 00 00 2e 64 d2 9b - Music mode

        b0 b1 b2 b3 00 01 01 2d 00 0d 73 01 27 01 ff a6 00 ff 00 00 4e 64 f2 e6 - Music mode (various sensitivity)
        b0 b1 b2 b3 00 01 01 2e 00 0d 73 01 27 01 ff a6 00 ff 00 00 5f 64 03 09 - Music mode (various sensitivity)
        b0 b1 b2 b3 00 01 01 2f 00 0d 73 01 27 01 ff a6 00 ff 00 00 64 64 08 14 - Music mode (various sensitivity)
        b0 b1 b2 b3 00 01 01 30 00 0d 73 01 27 01 ff a6 00 ff 00 00 37 64 db bb - Music mode (various sensitivity)
                                                                    ^^
                                                                    Likely sensitivity from 0-100 (0x64)
        b0 b1 b2 b3 00 01 01 60 00 0d 73 01 27 01 ff a6 00 ff 00 00 64 64 08 45 - Music mode (various sensitivity)
        b0 b1 b2 b3 00 01 01 5f 00 0d 73 01 27 01 ff a6 00 ff 00 00 64 64 08 44 - Music mode (various sensitivity)
        b0 b1 b2 b3 00 01 01 69 00 0d 73 01 26 01 ff 00 00 ff 00 00 00 64 fd 38 - Music mode (various sensitivity)
        b0 b1 b2 b3 00 01 01 68 00 0d 73 01 26 01 ff 00 00 ff 00 00 64 64 61 ff - Music mode (various sensitivity)


        b0 b1 b2 b3 00 01 01 08 00 0d 73 01 26 02 00 00 00 00 ff ff 64 60 5e 99 -- red lines
        b0 b1 b2 b3 00 01 01 16 00 0d 73 01 26 02 00 00 00 00 ff ff 64 64 62 af -- red lines
        b0 b1 b2 b3 00 01 01 17 00 0d 73 01 26 01 00 00 00 00 ff ff 64 64 61 ae -- rainbow lines
                                                                       ^^
                                                                       Likely brightness from 0-100 (0x64)
        """
        return [
            self.construct_wrapped_message(msg, inner_pre_constructed=True)
            for msg in super().construct_music_mode(
                sensitivity,
                brightness,
                mode,
                effect,
                foreground_color,
                background_color,
            )
        ]

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """
        The bytes to send for a level change request.

        b0 [unknown static?] b1 [unknown static?] b2 [unknown static?] b3 [unknown static?] 00 [unknown static?] 01 [unknown static?] 01 [unknown static?] 6a [incrementing sequence number] 00 [unknown static?] 0d [unknown, sometimes 0c] 41 [unknown static?] 02 [preset number] ff [foreground r] 00 [foreground g] 00 [foreground b] 00 [background red] ff [background green] 00 [background blue] 06 [speed or direction?] 00 [unknown static?] 00 [unknown static?] 00 [unknown static?] 47 [speed or direction?] cd [check sum]

        Known messages


        b0 b1 b2 b3 00 01 01 01 00 0c 10 14 15 0a 0b 0e 12 06 01 00 0f 84 dd - preset 1
        b0 b1 b2 b3 00 01 01 03 00 0d 41 02 00 ff ff 00 00 00 06 00 00 00 47 66 - preset 2
        b0 b1 b2 b3 00 01 01 04 00 0d 41 03 00 ff ff 00 00 00 06 00 00 00 48 69 - preset 3
        b0 b1 b2 b3 00 01 01 02 00 0d 41 01 00 ff ff 00 00 00 06 ff 00 00 45 61 - preset 4
        b0 b1 b2 b3 00 01 01 1f 00 0d 41 01 ff 00 00 00 00 00 06 ff 00 00 46 80 - preset 1 red or green
        b0 b1 b2 b3 00 01 01 27 00 0d 41 01 00 ff 00 00 00 00 06 ff 00 00 46 88 - preset 1 red or green
        b0 b1 b2 b3 00 01 01 2e 00 0d 41 01 ff 00 00 00 00 00 06 ff 00 00 46 8f - preset 1 red (foreground)
        b0 b1 b2 b3 00 01 01 27 00 0d 41 01 00 ff 00 00 00 00 06 ff 00 00 46 88 - preset 1 green (foreground)
        b0 b1 b2 b3 00 01 01 3e 00 0d 41 01 00 00 ff 00 00 00 06 ff 00 00 46 9f - preset 1 blue (foreground)
        b0 b1 b2 b3 00 01 01 54 00 0d 41 02 00 ff 00 00 00 00 06 00 00 00 48 b9 - preset 2 green (foreground)
        b0 b1 b2 b3 00 01 01 55 00 0d 41 02 ff 00 00 00 00 00 06 00 00 00 48 ba - preset 2 red (foreground)
        b0 b1 b2 b3 00 01 01 67 00 0d 41 02 ff 00 00 ff 00 00 06 00 00 00 47 ca - preset 2 red (foreground), red (background)
        b0 b1 b2 b3 00 01 01 67 00 0d 41 02 ff 00 00 ff 00 00 06 00 00 00 47 ca - preset 2 red (foreground), red (background)
        b0 b1 b2 b3 00 01 01 69 00 0d 41 02 ff 00 00 ff 00 00 06 00 00 00 47 cc - preset 2 red (foreground), red (background)
        b0 b1 b2 b3 00 01 01 6a 00 0d 41 02 ff 00 00 00 ff 00 06 00 00 00 47 cd - preset 2 red (foreground), green (background)
        b0 b1 b2 b3 00 01 01 77 00 0d 41 02 ff 00 00 00 ff 00 06 00 00 00 47 da - preset 2 red (foreground), green (background) - direction RTL
        b0 b1 b2 b3 00 01 01 7d 00 0d 41 02 ff 00 00 00 ff 00 06 00 00 00 47 e0 - preset 2 red (foreground), green (background) - direction RTL
        b0 b1 b2 b3 00 01 01 7d 00 0d 41 02 ff 00 00 00 ff 00 06 00 00 00 47 e0 - preset 2 red (foreground), green (background) - direction RTL
        b0 b1 b2 b3 00 01 01 7c 00 0d 41 02 ff 00 00 00 ff 00 06 01 00 00 48 e1 - preset 2 red (foreground), green (background) - direction LTR
        b0 b1 b2 b3 00 01 01 89 00 0d 41 02 ff 00 00 00 ff 00 00 00 00 00 41 e0 - preset 2 red (foreground), green (background) - direction LTR - speed 0
        b0 b1 b2 b3 00 01 01 8a 00 0d 41 02 ff 00 00 00 ff 00 64 00 00 00 a5 a9 - preset 2 red (foreground), green (background) - direction LTR - speed 64
        b0 b1 b2 b3 00 01 01 8b 00 0d 41 02 ff 00 00 00 ff 00 00 00 00 00 41 e2 - preset 2 red (foreground), green (background) - direction LTR - speed 0?
        b0 b1 b2 b3 00 01 01 8c 00 0d 41 02 ff 00 00 00 ff 00 64 00 00 00 a5 ab - preset 2 red (foreground), green (background) - direction LTR - speed 64?

        Set Blue
        b0b1b2b30001010b0034a0000600010000ff0000ff0002ffff000000ff00030000ff0000ff0004ffff000000ff00050000ff0000ff0006ffff000000ffac5f

        Query
        b0b1b2b30001010c0004818a8b9604
        b0b1b2b30001010c000e811a23280000640f000001000660a2

        Set Red
        b0b1b2b30001010d0034a0000600010000ff0000ff0002ff00000000ff00030000ff0000ff0004ff00000000ff00050000ff0000ff0006ff00000000ffaf67
        """
        return [
            self.construct_wrapped_message(msg, inner_pre_constructed=True)
            for msg in super().construct_levels_change(
                persist, red, green, blue, warm_white, cool_white, write_mode
            )
        ]

    def construct_zone_change(
        self,
        points: int,  # the number of points on the strip
        rgb_list: list[tuple[int, int, int]],
        speed: int,
        effect: MultiColorEffects,
    ) -> bytearray:
        """
        The bytes to send for multiple zones.

        Blue/Green - Static
        590063ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff00000000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff001e04640024

        Red/Blue - Jump
        5900630000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff00ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff00001e01640021

        White/Green - Static
        590063ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff00ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff00001e01640003
            11111 22222 33333 44444 55555 66666 77777 88888 99999 00000 11111 22222 33333 44444 55555

        White - Static
        590063ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff001e016400e5

        White - Running Water - Full speed
        590063ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff001e026400e6

        White - Running Water - 50% speed
        590063ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff001e023200b4

        Red - Blue - Gradient
        590063ff0000f60008ed0011e4001adb0023d3002bca0034c1003db80046af004fa700579e00609500698c007283007b7b008372008c69009560009e5700a74f00af4600b83d00c13400ca2b00d32300db1a00e41100ed0800f60000ff001e01640005

        Red - Brething
        590063ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000ff0000001e05640025
        """
        sent_zones = len(rgb_list)
        if sent_zones > points:
            raise ValueError(f"Device supports a maximum of {points} zones")
        pixel_bits = 9 + (points * 3)
        pixels = bytearray([pixel_bits >> 8, pixel_bits & 0xFF])
        msg = bytearray([0x59])
        msg.extend(pixels)
        zone_size = points // sent_zones
        remaining = points
        for rgb in rgb_list:
            for _ in range(zone_size):
                remaining -= 1
                msg.extend(bytearray([*rgb]))
        while remaining:
            remaining -= 1
            msg.extend(bytearray([*rgb]))
        msg.extend(bytearray([0x00, 0x1E]))
        msg.extend(bytearray([effect.value, speed]))
        msg.append(0x00)
        return self.construct_wrapped_message(msg)

    def construct_device_config(
        self,
        operating_mode: int | None,
        wiring: int | None,
        ic_type: int | None,  # ic type
        pixels_per_segment: int | None,  # pixels per segment
        segments: int | None,  # number of segments
        music_pixels_per_segment: int | None,  # music pixels per segment
        music_segments: int | None,  # number of music segments
    ) -> bytearray:
        """The bytes to send to change device config."""
        return self.construct_wrapped_message(
            super().construct_device_config(
                operating_mode,
                wiring,
                ic_type,
                pixels_per_segment,
                segments,
                music_pixels_per_segment,
                segments,
            ),
            inner_pre_constructed=True,
        )


class ProtocolLEDENETSocket(ProtocolLEDENET8Byte):
    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return True

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_SOCKET

    @property
    def timer_response_len(self) -> int:
        """Return the time response len."""
        return LEDENET_TIMERS_SOCKET_RESPONSE_LEN

    @property
    def timer_len(self) -> int:
        """Return a single timer len."""
        return 12

    @property
    def timer_count(self) -> int:
        """Return the number of timers."""
        return 8


class ProtocolLEDENETCCT(ProtocolLEDENET9Byte):
    MIN_BRIGHTNESS = 2

    @property
    def timer_response_len(self) -> int:
        """Return the time response len."""
        return LEDENET_TIMERS_8BYTE_RESPONSE_LEN

    @property
    def timer_len(self) -> int:
        """Return a single timer len."""
        return 14

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return False

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_CCT

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """
        The bytes to send for a level change request.

        b0 b1 b2 b3 00 01 01 52 00 09 35 b1 00 64 00 00 00 03 4d bd - 100% warm
        b0 b1 b2 b3 00 01 01 72 00 09 35 b1 64 64 00 00 00 03 b1 a5 - 100% cool
        b0 b1 b2 b3 00 01 01 9f 00 09 35 b1 64 32 00 00 00 03 7f 6e - 100% cool - dim 50%
        """
        assert warm_white is not None, "CCT devices must set a warm white value"
        assert cool_white is not None, "CCT devices must set a cool white value"
        scaled_temp, brightness = white_levels_to_scaled_color_temp(
            warm_white, cool_white
        )
        return [
            self.construct_message(
                bytearray(
                    [
                        0x35,
                        0xB1,
                        scaled_temp,
                        # If the brightness goes below the precision the device
                        # will flip from cold to warm
                        max(self.MIN_BRIGHTNESS, brightness),
                        0x00,
                        0x00,
                        0x00,
                        0x03,
                    ]
                )
            )
        ]


class ProtocolLEDENETCCTWrapped(ProtocolLEDENETCCT):
    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_CCT_WRAPPED

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return True

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    def construct_state_query(self) -> bytearray:
        """The bytes to send for a query request."""
        return self.construct_wrapped_message(
            super().construct_state_query(),
            inner_pre_constructed=True,
        )

    def construct_state_change(self, turn_on: int) -> bytearray:
        """The bytes to send for a state change request."""
        return self.construct_wrapped_message(
            super().construct_state_change(turn_on),
            inner_pre_constructed=True,
        )

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """
        The bytes to send for a level change request.

        b0 b1 b2 b3 00 01 01 52 00 09 35 b1 00 64 00 00 00 03 4d bd - 100% warm
        b0 b1 b2 b3 00 01 01 72 00 09 35 b1 64 64 00 00 00 03 b1 a5 - 100% cool
        b0 b1 b2 b3 00 01 01 9f 00 09 35 b1 64 32 00 00 00 03 7f 6e - 100% cool - dim 50%
        """
        return [
            self.construct_wrapped_message(
                super().construct_levels_change(
                    persist, red, green, blue, warm_white, cool_white, write_mode
                )[0],
                inner_pre_constructed=True,
            )
        ]


class ProtocolLEDENETAddressableChristmas(ProtocolLEDENETAddressableBase):
    @property
    def speed_is_delay(self) -> bool:
        """If True the speed is a delay in ms."""
        return True

    def construct_state_query(self) -> bytearray:
        """The bytes to send for a query request."""
        return self.construct_wrapped_message(
            super().construct_state_query(),
            inner_pre_constructed=True,
        )

    def construct_state_change(self, turn_on: int) -> bytearray:
        """The bytes to send for a state change request."""
        return self.construct_wrapped_message(
            super().construct_state_change(turn_on),
            inner_pre_constructed=True,
        )

    @property
    def name(self) -> str:
        """The name of the protocol."""
        return PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS

    @property
    def zones(self) -> bool:
        """If the protocol supports zones."""
        return True

    @property
    def power_push_updates(self) -> bool:
        """If True the protocol pushes power state updates when controlled via ir/rf/app."""
        return True

    @property
    def state_push_updates(self) -> bool:
        """If True the protocol pushes state updates when controlled via ir/rf/app."""
        return True

    @property
    def dimmable_effects(self) -> bool:
        """Protocol supports dimmable effects."""
        return False

    @property
    def requires_turn_on(self) -> bool:
        """If True the device must be turned on before setting level/patterns/modes."""
        return False

    def construct_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """
        The bytes to send for a preset pattern.
        0xB0 0xB1 0xB2 0xB3 0x00 0x01 0x01 0x2A 0x00 0x04 0x38 0x01 0x10 0x00 0x3F (15)
        0xB0 0xB1 0xB2 0xB3 0x00 0x01 0x01 0x2B 0x00 0x04 0x38 0x02 0x10 0x00 0x41 (15)
        0xB0 0xB1 0xB2 0xB3 0x00 0x01 0x01 0x2C 0x00 0x04 0x38 0x03 0x10 0x00 0x43 (15)
        0xB0 0xB1 0xB2 0xB3 0x00 0x01 0x01 0x2D 0x00 0x04 0x38 0x04 0x10 0x00 0x45 (15)
        0xB0 0xB1 0xB2 0xB3 0x00 0x01 0x01 0x2E 0x00 0x04 0x38 0x05 0x10 0x00 0x47 (15)
        0xB0 0xB1 0xB2 0xB3 0x00 0x01 0x01 0x2F 0x00 0x04 0x38 0x06 0x10 0x00 0x49 (15)
        """
        return self.construct_wrapped_message(
            bytearray(
                [
                    0x38,
                    pattern,
                    utils.speedToDelay(speed),
                ]
            )
        )

    def construct_levels_change(
        self,
        persist: int,
        red: int | None,
        green: int | None,
        blue: int | None,
        warm_white: int | None,
        cool_white: int | None,
        write_mode: LevelWriteMode | int,
    ) -> list[bytearray]:
        """
        The bytes to send for a level change request.

        Green 100%:
        b0b1b2b300010180000d3ba100646400000000000000a49d

        Blue 50%
        b0b1b2b300010110000d3ba176e4320000000000000068b5

        Red & green 255 and 25% bright
        b0b1b2b300010133000d3ba11e64190000000000000077f6

        Red & Blue 255 and 40%
        b0b1b2b30001014e000d3ba196642800000000000000fe1f

        Inner messages

        Single - Green - Brightness 100%
        3b a1 3c 64 64 00 00 00 00 00 00 00 e0

        Single - Green - Brightness 50%
        3b a1 3c 64 32 00 00 00 00 00 00 00 ae

        Single - Blue - Brightness 100%
        3b a1 78 64 64 00 00 00 00 00 00 00 1c

        Single - Red - Brightness 100%
        3b a1 00 64 64 00 00 00 00 00 00 00 a4

        Single - Pink (100% Red, 100% Blue) - Brightness 100%
        3b a1 96 64 64 00 00 00 00 00 00 00 3a

        Single - White (100% Red, 100% Green, 100% Blue) - Brightness 100%
        3b a1 00 00 64 00 00 00 00 00 00 00 40

        Single - Yellow (100% Red, 100% Green) - Brightness 100%
        3b a1 1e 64 64 00 00 00 00 00 00 00 c2

        Single - Light Blue (100% Blue, 100% Green) - Brightness 100%
        3b a1 5a 64 64 00 00 00 00 00 00 00 fe

        Single - Red - Brightness 0%
        3b a1 00 64 00 00 00 00 00 00 00 00 40

        Single - Red - Brightness 50%
        3b a1 00 64 32 00 00 00 00 00 00 00 72

        Single - Blue - Brightness 50%
        3b a1 78 64 32 00 00 00 00 00 00 00 ea
        """
        assert red is not None
        assert green is not None
        assert blue is not None
        h, s, v = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
        return [
            self.construct_wrapped_message(
                bytearray(
                    [
                        0x3B,
                        0xA1,
                        int(h * 180),
                        int(s * 100),
                        int(v * 100),
                        0x00,
                        0x00,
                        0x00,
                        0x00,
                        0x00,
                        0x00,
                        0x00,
                    ]
                )
            )
        ]

    def construct_zone_change(
        self,
        points: int,  # the number of points on the strip
        rgb_list: list[tuple[int, int, int]],
        speed: int,
        effect: MultiColorEffects,
    ) -> bytearray:
        """
        The bytes to send for multiple zones.

        6 Zone All red
        a000060001ff00000000ff0002ff00000000ff0003ff00000000ff0004ff00000000ff0005ff00000000ff0006ff00000000ffaf
        6 Zone All Yellow
        a000060001ffff000000ff0002ffff000000ff0003ffff000000ff0004ffff000000ff0005ffff000000ff0006ffff000000ffa9
        6 Zone All Green
        a00006000100ff000000ff000200ff000000ff000300ff000000ff000400ff000000ff000500ff000000ff000600ff000000ffaf
        6 Zone All Green
        a00006000100ff000000ff000200ff000000ff000300ff000000ff000400ff000000ff000500ff000000ff000600ff000000ffaf
        6 Zone All Cyan
        a00006000100ffff0000ff000200ffff0000ff000300ffff0000ff000400ffff0000ff000500ffff0000ff000600ffff0000ffa9
        6 Zone All White
        a000060001ffffff0000ff0002ffffff0000ff0003ffffff0000ff0004ffffff0000ff0005ffffff0000ff0006ffffff0000ffa3
        """
        sent_zones = len(rgb_list)
        if sent_zones > points:
            raise ValueError(f"Device supports a maximum of {points} zones")
        msg = bytearray([0xA0, 0x00, 0x06])
        zone_size = points // sent_zones
        remaining = points
        for rgb in rgb_list:
            for _ in range(zone_size):
                remaining -= 1
                msg.extend(
                    bytearray([0x00, points - remaining, *rgb, 0x00, 0x00, 0xFF])
                )
        while remaining:
            remaining -= 1
            msg.extend(
                bytearray([0x00, points - remaining, *rgb_list[-1], 0x00, 0x00, 0xFF])
            )
        return self.construct_wrapped_message(msg)

    def parse_strip_setting(self, msg: bytes) -> LEDENETAddressableDeviceConfiguration:
        """Parse a strip settings message."""
        return LEDENETAddressableDeviceConfiguration(
            pixels_per_segment=6,
            segments=None,
            music_pixels_per_segment=None,
            music_segments=None,
            wirings=[],
            wiring_num=None,
            wiring=None,
            num_to_wiring={},
            wiring_to_num={},
            ic_type=None,
            ic_type_num=None,
            operating_mode=COLOR_MODE_RGB,
        )
