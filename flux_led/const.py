"""FluxLED Models Database."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final, NamedTuple  # pylint: disable=no-name-in-module

MIN_TEMP: Final = 2700
MAX_TEMP: Final = 6500


class WhiteChannelType(Enum):
    WARM = MIN_TEMP
    NATURAL = MAX_TEMP - ((MAX_TEMP - MIN_TEMP) / 2)
    COLD = MAX_TEMP


# Legacy LevelWriteModel left for backwards compatibility
class LevelWriteMode(Enum):
    ALL = 0x00
    COLORS = 0xF0
    WHITES = 0x0F


@dataclass(frozen=True)
class LevelWriteModeData:
    ALL: int
    COLORS: int
    WHITES: int


class MultiColorEffects(Enum):
    STATIC = 0x01
    RUNNING_WATER = 0x02
    STROBE = 0x03
    JUMP = 0x04
    BREATHING = 0x05


class ExtendedCustomEffectPattern(Enum):
    """Pattern IDs for extended custom effects (e.g., 0xB6 device)."""

    WAVE = 0x01
    METEOR = 0x02
    STREAMER = 0x03
    BUILDING_BLOCKS = 0x04
    FLOWING_WATER = 0x05
    CHASE = 0x06
    HORSE_RACING = 0x07
    CYCLE = 0x08
    BREATHE = 0x09
    JUMP = 0x0A
    STROBE = 0x0B
    TWINKLING_STARS = 0x0C
    STARS_WINK = 0x0D
    WARNING = 0x0E
    COLLISION = 0x0F
    FIREWORKS = 0x10
    COMET = 0x11
    GRADIENT_METEOR = 0x12
    VOLCANO = 0x13
    SUPERLUMINAL = 0x14
    RAINBOW_BRIDGE = 0x15
    GRADIENT_OVERLAY = 0x16
    STATIC_GRADIENT = 0x65
    STATIC_FILL = 0x66


class ExtendedCustomEffectDirection(Enum):
    """Direction for extended custom effect animations."""

    LEFT_TO_RIGHT = 0x01
    RIGHT_TO_LEFT = 0x02


class ExtendedCustomEffectOption(Enum):
    """Option values for extended custom effects.

    The meaning varies by pattern:
    - Some patterns: DEFAULT=static color, VARIANT_1=color change
    - Rainbow patterns: DEFAULT=strobe, VARIANT_1=twinkle, VARIANT_2=breathe
    """

    DEFAULT = 0x00
    VARIANT_1 = 0x01
    VARIANT_2 = 0x02


class ScribbleEffect(Enum):
    """Global effect id for the scribble feature (E1 26 byte 2).

    Distinct from ExtendedCustomEffectPattern (E1 21). Byte-2 gaps (0x04,
    0x06, 0x07, ...) imply more effects/variants than the UI exposes.
    """

    STATIC = 0x00
    FLOWING = 0x01
    TWINKLING_STARS = 0x02
    TWINKLING_STARS_VARIANT = 0x03
    STARS_WINK = 0x05
    ACCUMULATE = 0x08


class ScribbleBlinkMode(Enum):
    """Blink mode for the scribble global command (E1 26 byte 6)."""

    NONE = 0x00
    SLOW = 0x08
    FAST = 0x10


class ScribbleLED(NamedTuple):
    """One LED's setting in a scribble (per-LED) configuration.

    Color and white are mutually exclusive:
      * rgb set, white None  -> color mode; brightness is the RGB value itself
        (the E1 26 V byte is derived from the RGB tuple)
      * white set, rgb None  -> white mode; the white level (0-100) is the
        brightness (the E1 26 WLVL byte)
      * both None            -> LED off (painted as color (0,0,0))

    Per-LED blink is achieved by grouping: LEDs sharing a
    (blink_mode, blink_speed) are painted by one E1 26 and blink independently
    per group. These are real, rendered fields that flow into the group key
    and the E1 26 paint (the E1 23 init carries no color/blink and does not
    render, so they never go into an E1 23 record).
    """

    rgb: tuple[int, int, int] | None = None  # (R,G,B) 0-255, or None
    white: int | None = None  # warm-white level 0-100, or None
    blink_mode: ScribbleBlinkMode = ScribbleBlinkMode.NONE  # E1 26 byte 6
    blink_speed: int = 100  # 0-100, only meaningful when blinking


DEFAULT_WHITE_CHANNEL_TYPE: Final = WhiteChannelType.WARM

PRESET_MUSIC_MODE: Final = 0x62
PRESET_MUSIC_MODE_LEGACY: Final = 0x5D

PRESET_MUSIC_MODES: Final = {PRESET_MUSIC_MODE, PRESET_MUSIC_MODE_LEGACY}

ATTR_IPADDR: Final = "ipaddr"
ATTR_ID: Final = "id"
ATTR_MODEL: Final = "model"
ATTR_MODEL_NUM: Final = "model_num"
ATTR_VERSION_NUM: Final = "version_num"
ATTR_FIRMWARE_DATE: Final = "firmware_date"
ATTR_MODEL_INFO: Final = "model_info"
ATTR_MODEL_DESCRIPTION: Final = "model_description"
ATTR_REMOTE_ACCESS_ENABLED: Final = "remote_access_enabled"
ATTR_REMOTE_ACCESS_HOST: Final = "remote_access_host"
ATTR_REMOTE_ACCESS_PORT: Final = "remote_access_port"


# Color modes
COLOR_MODE_DIM: Final = "DIM"
COLOR_MODE_CCT: Final = "CCT"
COLOR_MODE_RGB: Final = "RGB"
COLOR_MODE_RGBW: Final = "RGBW"
COLOR_MODE_RGBWW: Final = "RGBWW"
COLOR_MODE_ADDRESSABLE: Final = "ADDRESSABLE"

POWER_STATE_CHANGE_LATENCY: Final = 3
STATE_CHANGE_LATENCY: Final = 2
ADDRESSABLE_STATE_CHANGE_LATENCY: Final = 5
PRESET_PATTERN_CHANGE_LATENCY: Final = 40  # Time to switch to music mode

# WRITE_ALL_COLORS and WRITE_ALL_WHITES are used in external
# libraries, leaving them for backwards compatibility
WRITE_ALL_COLORS = (LevelWriteMode.ALL, LevelWriteMode.COLORS)
WRITE_ALL_WHITES = (LevelWriteMode.ALL, LevelWriteMode.WHITES)

DEFAULT_RETRIES: Final = 2

# Modes
MODE_SWITCH: Final = "switch"
MODE_COLOR: Final = "color"
MODE_WW: Final = "ww"
MODE_CUSTOM: Final = "custom"
MODE_MUSIC: Final = "music"
MODE_PRESET: Final = "preset"

# Transitions
TRANSITION_JUMP: Final = "jump"
TRANSITION_STROBE: Final = "strobe"
TRANSITION_GRADUAL: Final = "gradual"

STATIC_MODES = {MODE_COLOR, MODE_WW}

# Non light device models
MODEL_NUMS_SWITCHS = {0x19, 0x93, 0x0B, 0x94, 0x95, 0x96, 0x97}

COLOR_MODES_RGB = {COLOR_MODE_RGB, COLOR_MODE_RGBW, COLOR_MODE_RGBWW}
COLOR_MODES_RGB_CCT = {  # AKA Split RGB & CCT modes used for bulbs/lamps
    COLOR_MODE_RGB,
    COLOR_MODE_CCT,
}
COLOR_MODES_RGB_W = {  # AKA RGB/W in the Magic Home Pro app
    COLOR_MODE_RGB,
    COLOR_MODE_DIM,
}
COLOR_MODES_ADDRESSABLE = {COLOR_MODE_RGB}


DEFAULT_MODE: Final = COLOR_MODE_RGB


# States
STATE_HEAD: Final = "head"
STATE_MODEL_NUM: Final = "model_num"
STATE_POWER_STATE: Final = "power_state"
STATE_PRESET_PATTERN: Final = "preset_pattern"
STATE_MODE: Final = "mode"
STATE_SPEED: Final = "speed"
STATE_RED: Final = "red"
STATE_GREEN: Final = "green"
STATE_BLUE: Final = "blue"
STATE_WARM_WHITE: Final = "warm_white"
STATE_VERSION_NUMBER: Final = "version_number"
STATE_COOL_WHITE: Final = "cool_white"
STATE_COLOR_MODE: Final = "color_mode"
STATE_CHECK_SUM: Final = "check_sum"

CHANNEL_STATES = {
    STATE_RED,
    STATE_GREEN,
    STATE_BLUE,
    STATE_WARM_WHITE,
    STATE_COOL_WHITE,
}


EFFECT_RANDOM = "random"
EFFECT_MUSIC = "music"

# Addressable limits
SEGMENTS_MAX: Final = 2048
PIXELS_MAX: Final = 2048
PIXELS_PER_SEGMENT_MAX: Final = 300

MUSIC_SEGMENTS_MAX: Final = 64
MUSIC_PIXELS_MAX: Final = 960
MUSIC_PIXELS_PER_SEGMENT_MAX: Final = 150


#
# PUSH_UPDATE_INTERVAL reduces polling the device for state when its off
# since we do not care about the state when its off. When it turns on
# the device will push its new state to us anyways (except for buggy firmwares
# are identified in protocol.py)
#
# The downside to a longer polling interval for OFF is the
# time to declare the device offline is MAX_UPDATES_WITHOUT_RESPONSE*PUSH_UPDATE_INTERVAL
#
PUSH_UPDATE_INTERVAL = 90  # seconds

NEVER_TIME = -PUSH_UPDATE_INTERVAL
