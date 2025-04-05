from __future__ import annotations

import colorsys
import logging
import random
import time
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

from .const import (  # imported for back compat, remove once Home Assistant no longer uses
    ADDRESSABLE_STATE_CHANGE_LATENCY,
    ATTR_MODEL,
    ATTR_MODEL_DESCRIPTION,
    ATTR_MODEL_INFO,
    CHANNEL_STATES,
    COLOR_MODE_CCT,
    COLOR_MODE_DIM,
    COLOR_MODE_RGB,
    COLOR_MODE_RGBW,
    COLOR_MODE_RGBWW,
    COLOR_MODES_RGB,
    COLOR_MODES_RGB_CCT,
    COLOR_MODES_RGB_W,
    DEFAULT_MODE,
    DEFAULT_WHITE_CHANNEL_TYPE,
    EFFECT_MUSIC,
    EFFECT_RANDOM,
    MAX_TEMP,
    MODE_COLOR,
    MODE_CUSTOM,
    MODE_MUSIC,
    MODE_PRESET,
    MODE_SWITCH,
    MODE_WW,
    MODEL_NUMS_SWITCHS,
    NEVER_TIME,
    POWER_STATE_CHANGE_LATENCY,
    PRESET_MUSIC_MODE,
    PRESET_MUSIC_MODE_LEGACY,
    PRESET_MUSIC_MODES,
    PRESET_PATTERN_CHANGE_LATENCY,
    STATE_BLUE,
    STATE_CHANGE_LATENCY,
    STATE_COOL_WHITE,
    STATE_GREEN,
    STATE_POWER_STATE,
    STATE_RED,
    STATE_WARM_WHITE,
    STATIC_MODES,
    WhiteChannelType,
)
from .models_db import (
    BASE_MODE_MAP,
    HARDWARE_MAP,
    LEDENETHardware,
    LEDENETModel,
    get_model,
    is_known_model,
)
from .pattern import (
    ADDRESSABLE_EFFECT_ID_NAME,
    ADDRESSABLE_EFFECT_NAME_ID,
    ASSESSABLE_MULTI_COLOR_ID_NAME,
    CHRISTMAS_ADDRESSABLE_EFFECT_ID_NAME,
    CHRISTMAS_ADDRESSABLE_EFFECT_NAME_ID,
    EFFECT_CUSTOM,
    EFFECT_CUSTOM_CODE,
    EFFECT_ID_NAME,
    EFFECT_ID_NAME_LEGACY_CCT,
    EFFECT_LIST,
    EFFECT_LIST_DIMMABLE,
    EFFECT_LIST_LEGACY_CCT,
    ORIGINAL_ADDRESSABLE_EFFECT_ID_NAME,
    ORIGINAL_ADDRESSABLE_EFFECT_NAME_ID,
    PresetPattern,
)
from .protocol import (
    PROTOCOL_LEDENET_8BYTE,
    PROTOCOL_LEDENET_8BYTE_AUTO_ON,
    PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS,
    PROTOCOL_LEDENET_9BYTE,
    PROTOCOL_LEDENET_9BYTE_AUTO_ON,
    PROTOCOL_LEDENET_9BYTE_DIMMABLE_EFFECTS,
    PROTOCOL_LEDENET_25BYTE,
    PROTOCOL_LEDENET_ADDRESSABLE_A1,
    PROTOCOL_LEDENET_ADDRESSABLE_A2,
    PROTOCOL_LEDENET_ADDRESSABLE_A3,
    PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS,
    PROTOCOL_LEDENET_CCT,
    PROTOCOL_LEDENET_CCT_WRAPPED,
    PROTOCOL_LEDENET_ORIGINAL,
    PROTOCOL_LEDENET_ORIGINAL_CCT,
    PROTOCOL_LEDENET_ORIGINAL_RGBW,
    PROTOCOL_LEDENET_SOCKET,
    LEDENETAddressableDeviceConfiguration,
    LEDENETOriginalRawState,
    LEDENETRawState,
    ProtocolLEDENET8Byte,
    ProtocolLEDENET8ByteAutoOn,
    ProtocolLEDENET8ByteDimmableEffects,
    ProtocolLEDENET9Byte,
    ProtocolLEDENET9ByteAutoOn,
    ProtocolLEDENET9ByteDimmableEffects,
    ProtocolLEDENET25Byte,
    ProtocolLEDENETAddressableA1,
    ProtocolLEDENETAddressableA2,
    ProtocolLEDENETAddressableA3,
    ProtocolLEDENETAddressableChristmas,
    ProtocolLEDENETCCT,
    ProtocolLEDENETCCTWrapped,
    ProtocolLEDENETOriginal,
    ProtocolLEDENETOriginalCCT,
    ProtocolLEDENETOriginalRGBW,
    ProtocolLEDENETSocket,
    RemoteConfig,
)
from .scanner import FluxLEDDiscovery, is_legacy_device
from .timer import BuiltInTimer
from .utils import scaled_color_temp_to_white_levels, utils, white_levels_to_color_temp

_LOGGER = logging.getLogger(__name__)


class DeviceUnavailableException(RuntimeError):
    """Exception to indicate a device is not available."""


PROTOCOL_PROBES: tuple[type[ProtocolLEDENET8Byte], type[ProtocolLEDENETOriginal]] = (
    ProtocolLEDENET8Byte,
    ProtocolLEDENETOriginal,
)
PROTOCOL_PROBES_LEGACY: tuple[
    type[ProtocolLEDENETOriginal], type[ProtocolLEDENET8Byte]
] = (ProtocolLEDENETOriginal, ProtocolLEDENET8Byte)

PROTOCOL_TYPES = Union[
    ProtocolLEDENET8Byte,
    ProtocolLEDENET8ByteAutoOn,
    ProtocolLEDENET8ByteDimmableEffects,
    ProtocolLEDENET9Byte,
    ProtocolLEDENET9ByteAutoOn,
    ProtocolLEDENET9ByteDimmableEffects,
    ProtocolLEDENET25Byte,
    ProtocolLEDENETAddressableA1,
    ProtocolLEDENETAddressableA2,
    ProtocolLEDENETAddressableA3,
    ProtocolLEDENETOriginal,
    ProtocolLEDENETOriginalCCT,
    ProtocolLEDENETOriginalRGBW,
    ProtocolLEDENETCCT,
    ProtocolLEDENETCCTWrapped,
    ProtocolLEDENETSocket,
    ProtocolLEDENETAddressableChristmas,
]

ADDRESSABLE_PROTOCOLS = {
    PROTOCOL_LEDENET_ADDRESSABLE_A1,
    PROTOCOL_LEDENET_ADDRESSABLE_A2,
    PROTOCOL_LEDENET_ADDRESSABLE_A3,
}
ALL_ADDRESSABLE_PROTOCOLS = (
    ProtocolLEDENETAddressableA1,
    ProtocolLEDENETAddressableA2,
    ProtocolLEDENETAddressableA3,
)
ALL_IC_PROTOCOLS = (ProtocolLEDENETAddressableChristmas, *ALL_ADDRESSABLE_PROTOCOLS)
CHRISTMAS_EFFECTS_PROTOCOLS = {PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS}
OLD_EFFECTS_PROTOCOLS = {PROTOCOL_LEDENET_ADDRESSABLE_A1}
NEW_EFFECTS_PROTOCOLS = {
    PROTOCOL_LEDENET_ADDRESSABLE_A2,
    PROTOCOL_LEDENET_ADDRESSABLE_A3,
}
SPEED_ADJUST_WILL_TURN_ON = {
    PROTOCOL_LEDENET_ADDRESSABLE_A1,
    PROTOCOL_LEDENET_ADDRESSABLE_A2,
}
PROTOCOL_NAME_TO_CLS = {
    PROTOCOL_LEDENET_ORIGINAL: ProtocolLEDENETOriginal,
    PROTOCOL_LEDENET_ORIGINAL_CCT: ProtocolLEDENETOriginalCCT,
    PROTOCOL_LEDENET_ORIGINAL_RGBW: ProtocolLEDENETOriginalRGBW,
    PROTOCOL_LEDENET_8BYTE: ProtocolLEDENET8Byte,
    PROTOCOL_LEDENET_8BYTE_AUTO_ON: ProtocolLEDENET8ByteAutoOn,
    PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS: ProtocolLEDENET8ByteDimmableEffects,
    PROTOCOL_LEDENET_9BYTE: ProtocolLEDENET9Byte,
    PROTOCOL_LEDENET_9BYTE_AUTO_ON: ProtocolLEDENET9ByteAutoOn,
    PROTOCOL_LEDENET_9BYTE_DIMMABLE_EFFECTS: ProtocolLEDENET9ByteDimmableEffects,
    PROTOCOL_LEDENET_25BYTE: ProtocolLEDENET25Byte,
    PROTOCOL_LEDENET_ADDRESSABLE_A3: ProtocolLEDENETAddressableA3,
    PROTOCOL_LEDENET_ADDRESSABLE_A2: ProtocolLEDENETAddressableA2,
    PROTOCOL_LEDENET_ADDRESSABLE_A1: ProtocolLEDENETAddressableA1,
    PROTOCOL_LEDENET_CCT: ProtocolLEDENETCCT,
    PROTOCOL_LEDENET_CCT_WRAPPED: ProtocolLEDENETCCTWrapped,
    PROTOCOL_LEDENET_SOCKET: ProtocolLEDENETSocket,
    PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS: ProtocolLEDENETAddressableChristmas,
}


PATTERN_CODE_TO_EFFECT = {
    PRESET_MUSIC_MODE: MODE_MUSIC,
    PRESET_MUSIC_MODE_LEGACY: MODE_MUSIC,
    EFFECT_CUSTOM_CODE: EFFECT_CUSTOM,
}

SERIALIZABLE_TYPES = (str, bool, dict, int, float, list, tuple, set)


DEFAULT_PORT = 5577
ARMACOST_PORT = 34001


class DeviceType(Enum):
    Bulb = 0
    Switch = 1


class LEDENETDevice:
    """An LEDENET Device."""

    def __init__(
        self,
        ipaddr: str,
        port: int = 0,
        timeout: float = 5,
        discovery: FluxLEDDiscovery | None = None,
    ) -> None:
        """Init the LEDENEt Device."""
        self.ipaddr: str = ipaddr
        self._port: int = port
        self.timeout: float = timeout
        self.raw_state: LEDENETOriginalRawState | LEDENETRawState | None = None
        self.available: bool | None = None
        self._model_num: int | None = None
        self._model_data: LEDENETModel | None = None
        self._paired_remotes: int | None = None
        self._remote_config: RemoteConfig | None = None
        self._white_channel_channel_type: WhiteChannelType = DEFAULT_WHITE_CHANNEL_TYPE
        self._discovery = discovery
        self._protocol: PROTOCOL_TYPES | None = None
        self._mode: str | None = None
        self._transition_complete_time: float = 0
        self._preset_pattern_transition_complete_time: float = 0
        self._power_state_transition_complete_time: float = 0
        self._last_effect_brightness: int = 100
        self._device_config: LEDENETAddressableDeviceConfiguration | None = None
        self._last_message: dict[str, bytes] = {}
        self._unavailable_reason: str | None = None

    def _protocol_probes(
        self,
    ) -> (
        tuple[type[ProtocolLEDENETOriginal], type[ProtocolLEDENET8Byte]]
        | tuple[type[ProtocolLEDENET8Byte], type[ProtocolLEDENETOriginal]]
    ):
        """Determine the probe order based on device type."""
        discovery = self.discovery
        return (
            PROTOCOL_PROBES_LEGACY if is_legacy_device(discovery) else PROTOCOL_PROBES
        )

    @property
    def model_num(self) -> int:
        """Return the model number."""
        assert self._model_num is not None
        return self._model_num

    @property
    def model_data(self) -> LEDENETModel:
        """Return the model data."""
        assert self._model_data is not None
        return self._model_data

    @property
    def discovery(self) -> FluxLEDDiscovery | None:
        """Return the discovery data."""
        return self._discovery

    @discovery.setter
    def discovery(self, value: FluxLEDDiscovery | None) -> None:
        """Set the discovery data."""
        self._discovery = value

    @property
    def white_channel_channel_type(self) -> WhiteChannelType:
        """Return the type of the white channel."""
        return self._white_channel_channel_type

    @white_channel_channel_type.setter
    def white_channel_channel_type(self, value: WhiteChannelType) -> None:
        """Set the type of the white channel."""
        self._white_channel_channel_type = value

    @property
    def hardware(self) -> LEDENETHardware | None:
        """Retrurn the hardware mapping for the device."""
        if not self._discovery or ATTR_MODEL not in self._discovery:
            return None
        model = self._discovery.get(ATTR_MODEL)
        if model is None:
            return None
        assert isinstance(model, str)
        return HARDWARE_MAP.get(model)

    @property
    def paired_remotes(self) -> int | None:
        """Return the number of paired remotes or None if not supported."""
        return self._paired_remotes

    @property
    def remote_config(self) -> RemoteConfig | None:
        """Return the number of remote config or None if not supported."""
        return self._remote_config

    @property
    def speed_adjust_off(self) -> int:
        """Return true if the speed of an effect can be adjusted while off."""
        return self.protocol not in SPEED_ADJUST_WILL_TURN_ON

    @property
    def _whites_are_temp_brightness(self) -> bool:
        """Return true if warm_white and cool_white are scaled temp values and not raw 0-255."""
        return self.protocol in (PROTOCOL_LEDENET_CCT, PROTOCOL_LEDENET_CCT_WRAPPED)

    @property
    def model(self) -> str:
        """Return the human readable model description."""
        if self._discovery and self._discovery.get(ATTR_MODEL_DESCRIPTION):
            return f"{self._discovery[ATTR_MODEL_DESCRIPTION]} (0x{self.model_num:02X})"
        return f"{self.model_data.description} (0x{self.model_num:02X})"

    @property
    def version_num(self) -> int:
        """Return the version number."""
        assert self.raw_state is not None
        raw_state = self.raw_state
        if hasattr(raw_state, "version_number"):
            assert isinstance(raw_state, LEDENETRawState)
            return raw_state.version_number
        return 0  # old devices report as 0

    @property
    def preset_pattern_num(self) -> int:
        """Return the preset pattern number."""
        assert self.raw_state is not None
        return self.raw_state.preset_pattern

    @property
    def rgbwprotocol(self) -> bool:
        """Devices that don't require a separate rgb/w bit."""
        return self.rgbwcapable or self.model_data.always_writes_white_and_colors

    @property
    def microphone(self) -> bool:
        """Devices that have a microphone built in."""
        return self.model_data.microphone

    @property
    def rgbwcapable(self) -> bool:
        """Devices that actually support rgbw."""
        color_modes = self.color_modes
        return COLOR_MODE_RGBW in color_modes or COLOR_MODE_RGBWW in color_modes

    @property
    def device_type(self) -> DeviceType:
        """Return the device type."""
        is_switch = self.model_num in MODEL_NUMS_SWITCHS
        return DeviceType.Switch if is_switch else DeviceType.Bulb

    @property
    def color_temp(self) -> int:
        """Return the current color temp in kelvin."""
        return (self.getWhiteTemperature())[0]

    @property
    def min_temp(self) -> int:
        """Returns the minimum color temp in kelvin."""
        return int(self._white_channel_channel_type.value)

    @property
    def max_temp(self) -> int:
        """Returns the maximum color temp in kelvin."""
        return MAX_TEMP

    @property
    def _rgbwwprotocol(self) -> bool:
        """Device that uses the 9-byte protocol."""
        return self.protocol in (
            PROTOCOL_LEDENET_9BYTE,
            PROTOCOL_LEDENET_9BYTE_DIMMABLE_EFFECTS,
        )

    @property
    def white_active(self) -> bool:
        """Any white channel is active."""
        assert self.raw_state is not None
        raw_state = self.raw_state
        return bool(raw_state.warm_white or raw_state.cool_white)

    @property
    def color_active(self) -> bool:
        """Any color channel is active."""
        assert self.raw_state is not None
        raw_state = self.raw_state
        return bool(raw_state.red or raw_state.green or raw_state.blue)

    def rgbw_color_temp_support(self, color_modes: set[str]) -> bool:
        """RGBW color temp support."""
        return COLOR_MODE_RGBW in color_modes and self.max_temp != self.min_temp

    @property
    def color_is_white_only(self) -> bool:
        """Return if the curent color is active and white."""
        assert self.raw_state is not None
        raw_state = self.raw_state
        return bool(
            # At least one channel is on
            (raw_state.red or raw_state.green or raw_state.blue or raw_state.warm_white)
            # The color channels are white
            and raw_state.red == raw_state.green == raw_state.blue
        )

    @property
    def multi_color_mode(self) -> bool:
        """The device supports multiple color modes."""
        return len(self.color_modes) > 1

    @property
    def color_modes(self) -> set[str]:
        """The available color modes."""
        color_modes = self._internal_color_modes
        # We support CCT mode if the device supports RGBWW
        # but we do not add it to internal color modes as
        # we need to distingush between devices that are RGB/CCT
        # and ones that are RGB&CCT
        if (
            COLOR_MODE_CCT not in color_modes and COLOR_MODE_RGBWW in color_modes
        ) or self.rgbw_color_temp_support(color_modes):
            return {COLOR_MODE_CCT, *color_modes}
        return color_modes

    @property
    def _internal_color_modes(self) -> set[str]:
        """The internal available color modes."""
        assert self.raw_state is not None
        if (
            self._device_config is not None
            # Currently this is only the SK6812RGBW strips on 0xA3
            and self._device_config.operating_mode == COLOR_MODE_RGBW
        ):
            return {COLOR_MODE_RGBW}
        if not is_known_model(self.model_num):
            # Default mode is RGB
            return BASE_MODE_MAP.get(self.raw_state.mode & 0x0F, {DEFAULT_MODE})
        model_data = self.model_data
        return model_data.mode_to_color_mode.get(
            self.raw_state.mode, model_data.color_modes
        )

    @property
    def pixels_per_segment(self) -> int | None:
        """Return the pixels per segment."""
        if self._device_config is None:
            return None
        return self._device_config.pixels_per_segment

    @property
    def segments(self) -> int | None:
        """Return the number of segments."""
        if self._device_config is None:
            return None
        return self._device_config.segments

    @property
    def music_pixels_per_segment(self) -> int | None:
        """Return the music pixels per segment."""
        if self._device_config is None:
            return None
        return self._device_config.music_pixels_per_segment

    @property
    def music_segments(self) -> int | None:
        """Return the number of music segments."""
        if self._device_config is None:
            return None
        return self._device_config.music_segments

    @property
    def wiring(self) -> str | None:
        """Return the sort order as a string."""
        device_config = self.model_data.device_config
        if not device_config.wiring:
            return None
        if self._device_config:
            return self._device_config.wiring
        assert self.raw_state is not None
        return device_config.num_to_wiring.get(int((self.raw_state.mode & 0xF0) / 16))

    @property
    def wiring_num(self) -> int | None:
        """Return the wiring number."""
        if not self.model_data.device_config.wiring:
            return None
        if self._device_config:
            return self._device_config.wiring_num
        assert self.raw_state is not None
        return int((self.raw_state.mode & 0xF0) / 16)

    @property
    def wirings(self) -> list[str] | None:
        """Return available wirings for the device."""
        device_config = self.model_data.device_config
        if not device_config.wiring:
            return None
        if self._device_config:
            return list(self._device_config.wirings)
        return list(device_config.wiring_to_num)

    @property
    def operating_mode(self) -> str | None:
        """Return the strip mode as a string."""
        device_config = self.model_data.device_config
        if not device_config.operating_modes:
            return None
        if self._device_config:
            return self._device_config.operating_mode
        assert self.raw_state is not None
        return device_config.num_to_operating_mode.get(self.raw_state.mode & 0x0F)

    @property
    def operating_mode_num(self) -> int | None:
        """Return the strip mode as a string."""
        if not self.model_data.device_config.operating_modes:
            return None
        assert self.raw_state is not None
        return self.raw_state.mode & 0x0F

    @property
    def operating_modes(self) -> list[str] | None:
        """Return available operating modes for the device."""
        if not self.model_data.device_config.operating_modes:
            return None
        return list(self.model_data.device_config.operating_mode_to_num)

    @property
    def ic_type(self) -> str | None:
        """Return the strip ictype as a string."""
        if not self.model_data.device_config.ic_type:
            return None
        assert self._device_config is not None
        return self._device_config.ic_type

    @property
    def ic_type_num(self) -> int | None:
        """Return the strip ictype as an int."""
        if not self.model_data.device_config.ic_type:
            return None
        assert self._device_config is not None
        return self._device_config.ic_type_num

    @property
    def ic_types(self) -> list[str] | None:
        """Return the ic types."""
        if not self.model_data.device_config.ic_type:
            return None
        return list(self.model_data.device_config.ic_type_to_num)

    @property
    def color_mode(self) -> str | None:
        """The current color mode."""
        color_modes = self._internal_color_modes
        if COLOR_MODE_RGBWW in color_modes:
            # We support CCT mode if the device supports RGBWW
            return COLOR_MODE_RGBWW if self.color_active else COLOR_MODE_CCT
        if self.rgbw_color_temp_support(color_modes):
            # We support CCT mode if the device supports RGB&W
            return COLOR_MODE_CCT if self.color_is_white_only else COLOR_MODE_RGBW
        if (
            color_modes == COLOR_MODES_RGB_CCT
        ):  # RGB/CCT split, only one active at a time
            return COLOR_MODE_CCT if self.white_active else COLOR_MODE_RGB
        if color_modes == COLOR_MODES_RGB_W:  # RGB/W split, only one active at a time
            return COLOR_MODE_DIM if self.white_active else COLOR_MODE_RGB
        if color_modes:
            return next(iter(color_modes))
        return None  # Usually a switch or non-light device

    @property
    def protocol(self) -> str | None:
        """Returns the name of the protocol in use."""
        if self._protocol is None:
            return None
        return self._protocol.name

    @property
    def dimmable_effects(self) -> bool:
        """Return true of the device supports dimmable effects."""
        assert self._protocol is not None
        return self._protocol.dimmable_effects

    @property
    def requires_turn_on(self) -> bool:
        """Return true of the device requires a power on command before setting levels/effects."""
        assert self._protocol is not None
        return self._protocol.requires_turn_on

    @property
    def is_on(self) -> bool:
        assert self.raw_state is not None
        assert self._protocol is not None
        return self.raw_state.power_state == self._protocol.on_byte

    @property
    def mode(self) -> str | None:
        return self._mode

    @property
    def warm_white(self) -> int:
        assert self.raw_state is not None
        return self.raw_state.warm_white if self._rgbwwprotocol else 0

    @property
    def effect_list(self) -> list[str]:
        """Return the list of available effects."""
        effects: Iterable[str] = []
        protocol = self.protocol
        if protocol in OLD_EFFECTS_PROTOCOLS:
            effects = ORIGINAL_ADDRESSABLE_EFFECT_ID_NAME.values()
        elif protocol in NEW_EFFECTS_PROTOCOLS:
            effects = ADDRESSABLE_EFFECT_ID_NAME.values()
        elif protocol in CHRISTMAS_EFFECTS_PROTOCOLS:
            effects = CHRISTMAS_ADDRESSABLE_EFFECT_ID_NAME.values()
        elif COLOR_MODES_RGB.intersection(self.color_modes):
            effects = EFFECT_LIST_DIMMABLE if self.dimmable_effects else EFFECT_LIST
        elif protocol == PROTOCOL_LEDENET_ORIGINAL_CCT:
            effects = EFFECT_LIST_LEGACY_CCT
        if self.microphone:
            return [*effects, EFFECT_RANDOM, EFFECT_MUSIC]
        return [*effects, EFFECT_RANDOM]

    @property
    def effect(self) -> str | None:
        """Return the current effect."""
        if self.protocol in CHRISTMAS_EFFECTS_PROTOCOLS:
            return self._named_effect
        return PATTERN_CODE_TO_EFFECT.get(self.preset_pattern_num, self._named_effect)

    @property
    def _named_effect(self) -> str | None:
        """Returns the named effect."""
        assert self.raw_state is not None
        mode = self.raw_state.mode
        pattern_code = self.preset_pattern_num
        protocol = self.protocol
        if protocol in OLD_EFFECTS_PROTOCOLS:
            effect_id = (pattern_code << 8) + mode - 99
            return ORIGINAL_ADDRESSABLE_EFFECT_ID_NAME.get(effect_id)
        if protocol in NEW_EFFECTS_PROTOCOLS:
            if pattern_code == 0x25:
                return ADDRESSABLE_EFFECT_ID_NAME.get(mode)
            if pattern_code == 0x24:
                return ASSESSABLE_MULTI_COLOR_ID_NAME.get(mode)
            return None
        if protocol in CHRISTMAS_EFFECTS_PROTOCOLS:
            if pattern_code == 0x25:
                return CHRISTMAS_ADDRESSABLE_EFFECT_ID_NAME.get(mode)
            return None
        if protocol == PROTOCOL_LEDENET_ORIGINAL_CCT:
            return EFFECT_ID_NAME_LEGACY_CCT.get(pattern_code)
        return EFFECT_ID_NAME.get(pattern_code)

    @property
    def cool_white(self) -> int:
        assert self.raw_state is not None
        if self._rgbwwprotocol:
            return self.raw_state.cool_white
        return 0

    # Old name is deprecated
    @property
    def cold_white(self) -> int:
        return self.cool_white

    @property
    def brightness(self) -> int:
        """Return current brightness 0-255.

        For warm white return current led level. For RGB
        calculate the HSV and return the 'value'.
        for CCT calculate the brightness.
        for ww send led level
        """
        color_mode = self.color_mode
        raw_state = self.raw_state

        assert raw_state is not None

        if self._named_effect:
            if self.dimmable_effects:
                if (
                    self.protocol in NEW_EFFECTS_PROTOCOLS
                    and time.monotonic() > self._transition_complete_time
                ):
                    # the red byte holds the brightness during an effect
                    return min(255, round(raw_state.red * 255 / 100))
                return round(self._last_effect_brightness * 255 / 100)
            return 255
        if raw_state.preset_pattern in PRESET_MUSIC_MODES and not self.dimmable_effects:
            return 255
        if color_mode == COLOR_MODE_DIM:
            return int(raw_state.warm_white)
        if color_mode == COLOR_MODE_CCT:
            _, b = self.getWhiteTemperature()
            return b

        r, g, b = self.getRgb()
        _, _, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        v_255 = v * 255
        if color_mode == COLOR_MODE_RGBW:
            return round((v_255 + raw_state.warm_white) / 2)
        if color_mode == COLOR_MODE_RGBWW:
            return round((v_255 + raw_state.warm_white + raw_state.cool_white) / 3)

        # Default color mode (RGB)
        return int(v_255)

    def _determineMode(self) -> str | None:
        assert self.raw_state is not None
        pattern_code = self.raw_state.preset_pattern
        if self.device_type == DeviceType.Switch:
            return MODE_SWITCH
        if pattern_code in (0x41, 0x61):
            if self.color_mode in {COLOR_MODE_DIM, COLOR_MODE_CCT}:
                return MODE_WW
            return MODE_COLOR
        if pattern_code == EFFECT_CUSTOM_CODE:
            return (
                MODE_PRESET
                if self.protocol in CHRISTMAS_EFFECTS_PROTOCOLS
                else MODE_CUSTOM
            )
        if pattern_code in (PRESET_MUSIC_MODE, PRESET_MUSIC_MODE_LEGACY):
            return MODE_MUSIC
        if PresetPattern.valid(pattern_code):
            return MODE_PRESET
        if BuiltInTimer.valid(pattern_code):
            return BuiltInTimer.valtostr(pattern_code)
        if self.protocol in ADDRESSABLE_PROTOCOLS:
            return MODE_PRESET
        return None

    @property
    def port(self) -> int:
        """Return the discovered port."""
        if self._port:
            return self._port
        if (
            self._discovery
            and "armacost" in (self._discovery.get(ATTR_MODEL_INFO) or "").lower()
        ):
            return ARMACOST_PORT
        return DEFAULT_PORT

    def set_unavailable(self, reason: str) -> None:
        _LOGGER.debug("%s: set_unavailable: %s", self.ipaddr, reason)
        self._unavailable_reason = reason
        self.available = False

    def set_available(self, reason: str) -> None:
        _LOGGER.debug("%s: set_available: %s", self.ipaddr, reason)
        self._unavailable_reason = None
        self.available = True

    def process_device_config_response(self, msg: bytes) -> None:
        """Process an IC (strip config) response."""
        assert isinstance(self._protocol, ALL_IC_PROTOCOLS)
        self._device_config = self._protocol.parse_strip_setting(msg)
        _LOGGER.debug("%s: device_config: %s", self.ipaddr, self._device_config)

    def process_state_response(self, rx: bytes) -> bool:
        """Process a state change message."""
        assert self._protocol is not None
        if not self._protocol.is_valid_state_response(rx):
            _LOGGER.warning(
                "%s: Invalid response: %s",
                self.ipaddr,
                utils.raw_state_to_dec(rx),
            )
            return False
        return self._process_valid_state_response(rx)

    def _process_valid_state_response(self, rx: bytes) -> bool:
        assert self._protocol is not None
        raw_state: LEDENETOriginalRawState | LEDENETRawState = (
            self._protocol.named_raw_state(rx)
        )
        _LOGGER.debug("%s: State: %s", self.ipaddr, raw_state)

        if raw_state != self.raw_state:
            _LOGGER.debug(
                "%s: unmapped raw state: %s",
                self.ipaddr,
                utils.raw_state_to_dec(raw_state),
            )

        now_time = time.monotonic()
        transition_states = set()
        if now_time < self._power_state_transition_complete_time:
            transition_states.add(STATE_POWER_STATE)
        if now_time < self._transition_complete_time:
            # Do not update the channel states if a transition is
            # in progress as the state will not be correct
            # until the transition is completed since devices
            # "FADE" into the state requested.
            transition_states |= CHANNEL_STATES
        if now_time < self._preset_pattern_transition_complete_time:
            transition_states.add("preset_pattern")

        if transition_states:
            self._replace_raw_state(
                {
                    name: value
                    for name, value in raw_state._asdict().items()
                    if name not in transition_states
                }
            )
        else:
            self._set_raw_state(raw_state)

        _LOGGER.debug("%s: Mapped State: %s", self.ipaddr, self.raw_state)

        mode = self._determineMode()

        if mode is None:
            _LOGGER.debug(
                "%s: Unable to determine mode from raw state: %s",
                self.ipaddr,
                utils.raw_state_to_dec(rx),
            )
            return False

        self._mode = mode
        return True

    def process_power_state_response(self, msg: bytes) -> bool:
        """Process a power state change message."""
        assert self._protocol is not None
        if not self._protocol.is_valid_power_state_response(msg):
            _LOGGER.warning(
                "%s: Recieved invalid power state response: %s",
                self.ipaddr,
                utils.raw_state_to_dec(msg),
            )
            return False
        _LOGGER.debug("%s: Setting power state to: %s", self.ipaddr, f"0x{msg[2]:02X}")
        self._set_power_state(msg[2])
        return True

    def process_extended_state_response(self, msg: bytes) -> bool:
        """Process and extended state response."""
        assert self._protocol is not None
        self._process_valid_state_response(self._protocol.extended_state_to_state(msg))
        return True

    def _set_raw_state(
        self,
        raw_state: LEDENETOriginalRawState | LEDENETRawState,
        updated: set[str] | None = None,
    ) -> None:
        """Set the raw state remapping channels as needed.

        The goal is to normalize the data so the raw state
        is always in the same format reguardless of the protocol

        Some devices need to have channels remapped

        Other devices uses color_temp/brightness format
        which needs to be converted back to 0-255 values for
        warm_white and cool_white
        """
        channel_map = self.model_data.channel_map
        # Only remap updated states as we do not want to switch any
        # state that have not changed since they will already be in
        # the correct slot
        #
        # If updated is None than all raw_state values have been sent
        #
        if self._whites_are_temp_brightness:
            assert isinstance(raw_state, LEDENETRawState)
            # Only convert on a full update since we still use 0-255 internally
            if updated is not None:
                self.raw_state = raw_state
                return
            # warm_white is the color temp from 1-100
            temp = raw_state.warm_white
            # cold_white is the brightness from 1-100
            brightness = raw_state.cool_white
            warm_white, cool_white = scaled_color_temp_to_white_levels(temp, brightness)
            self.raw_state = raw_state._replace(
                warm_white=warm_white, cool_white=cool_white
            )
            return

        if channel_map:
            if updated is None:
                updated = set(channel_map.keys())
            self.raw_state = raw_state._replace(
                **{
                    name: getattr(raw_state, source)
                    if source in updated
                    else getattr(raw_state, name)
                    for name, source in channel_map.items()
                }
            )
            return

        if isinstance(self._protocol, ProtocolLEDENETAddressableA3):
            if updated is not None:
                self.raw_state = raw_state
                return
            # A3 uses a unique scale for warm white
            self.raw_state = raw_state._replace(
                warm_white=utils.A3WarmWhiteToByte(raw_state.warm_white)
            )
            return

        self.raw_state = raw_state

    def __str__(self) -> str:
        assert self.raw_state is not None
        assert self._protocol is not None

        rx = self.raw_state
        if not rx:
            return "No state data"
        mode = self.mode
        color_mode = self.color_mode
        power_str = "Unknown power state"
        if rx.power_state == self._protocol.on_byte:
            power_str = "ON "
        elif rx.power_state == self._protocol.off_byte:
            power_str = "OFF "

        if mode in STATIC_MODES:
            if color_mode in COLOR_MODES_RGB:
                mode_str = f"Color: {(rx.red, rx.green, rx.blue)}"
                # Should add ability to get CCT from rgbwcapable*
                if self.rgbwcapable:
                    mode_str += f" White: {rx.warm_white}"
                else:
                    mode_str += f" Brightness: {round(self.brightness * 100 / 255)}%"
            elif color_mode == COLOR_MODE_DIM:
                mode_str = f"Warm White: {utils.byteToPercent(rx.warm_white)}%"
            elif color_mode == COLOR_MODE_CCT:
                cct_value = self.getWhiteTemperature()
                mode_str = f"CCT: {cct_value[0]}K Brightness: {round(cct_value[1] * 100 / 255)}%"
        elif mode == MODE_PRESET:
            mode_str = f"Pattern: {self.effect} (Speed {self.speed}%)"
        elif mode == MODE_CUSTOM:
            mode_str = f"Custom pattern (Speed {self.speed}%)"
        elif BuiltInTimer.valid(rx.preset_pattern):
            mode_str = BuiltInTimer.valtostr(rx.preset_pattern)
        elif mode == MODE_MUSIC:
            mode_str = "Music"
        elif mode == MODE_SWITCH:
            mode_str = "Switch"
        else:
            mode_str = f"Unknown mode 0x{rx.preset_pattern:x}"
        mode_str += " raw state: "
        mode_str += utils.raw_state_to_dec(rx)
        return f"{power_str} [{mode_str}]"

    def _set_power_state(self, new_power_state: int) -> None:
        """Set the power state in the raw state."""
        self._replace_raw_state({"power_state": new_power_state})
        self._set_transition_complete_time()

    def _replace_raw_state(self, new_states: dict[str, int]) -> None:
        assert self.raw_state is not None
        _LOGGER.debug("%s: _replace_raw_state: %s", self.ipaddr, new_states)
        self._set_raw_state(
            self.raw_state._replace(**new_states), set(new_states.keys())
        )

    def isOn(self) -> bool:
        return self.is_on

    def getWarmWhite255(self) -> int:
        if self.color_mode not in {COLOR_MODE_CCT, COLOR_MODE_DIM}:
            return 255
        return self.brightness

    def getWhiteTemperature(self) -> tuple[int, int]:
        """Returns the color temp and brightness"""
        # Assume input temperature of between 2700 and 6500 Kelvin, and scale
        # the warm and cold LEDs linearly to provide that
        assert self.raw_state is not None
        raw_state = self.raw_state
        warm_white = raw_state.warm_white
        if self.rgbw_color_temp_support(self.color_modes):
            cool_white = raw_state.red if self.color_is_white_only else 0
        else:
            cool_white = raw_state.cool_white
        temp, brightness = white_levels_to_color_temp(
            warm_white, cool_white, self.min_temp, self.max_temp
        )
        return temp, brightness

    def getRgbw(self) -> tuple[int, int, int, int]:
        """Returns red,green,blue,white (usually warm)."""
        if self.color_mode not in COLOR_MODES_RGB:
            return (255, 255, 255, 255)
        return self.rgbw

    @property
    def rgbw(self) -> tuple[int, int, int, int]:
        """Returns red,green,blue,white (usually warm)."""
        assert self.raw_state is not None
        raw_state = self.raw_state
        return (
            raw_state.red,
            raw_state.green,
            raw_state.blue,
            raw_state.warm_white,
        )

    def getRgbww(self) -> tuple[int, int, int, int, int]:
        """Returns red,green,blue,warm,cool."""
        if self.color_mode not in COLOR_MODES_RGB:
            return (255, 255, 255, 255, 255)
        return self.rgbww

    @property
    def rgbww(self) -> tuple[int, int, int, int, int]:
        """Returns red,green,blue,warm,cool."""
        raw_state = self.raw_state
        assert raw_state is not None
        return (
            raw_state.red,
            raw_state.green,
            raw_state.blue,
            raw_state.warm_white,
            raw_state.cool_white,
        )

    def getRgbcw(self) -> tuple[int, int, int, int, int]:
        """Returns red,green,blue,cool,warm."""
        if self.color_mode not in COLOR_MODES_RGB:
            return (255, 255, 255, 255, 255)
        return self.rgbcw

    @property
    def rgbcw(self) -> tuple[int, int, int, int, int]:
        """Returns red,green,blue,cool,warm."""
        raw_state = self.raw_state
        assert raw_state is not None
        return (
            raw_state.red,
            raw_state.green,
            raw_state.blue,
            raw_state.cool_white,
            raw_state.warm_white,
        )

    def getCCT(self) -> tuple[int, int]:
        if self.color_mode != COLOR_MODE_CCT:
            return (255, 255)
        raw_state = self.raw_state
        assert raw_state is not None
        return (raw_state.warm_white, raw_state.cool_white)

    @property
    def speed(self) -> int:
        assert self.raw_state is not None
        if self._protocol is not None and not self._protocol.speed_is_delay:
            return self.raw_state.speed
        if self.protocol in CHRISTMAS_EFFECTS_PROTOCOLS:
            return utils.delayToSpeed(self.raw_state.green)
        return utils.delayToSpeed(self.raw_state.speed)

    def getSpeed(self) -> int:
        return self.speed

    def _generate_random_levels_change(self) -> tuple[list[bytearray], dict[str, int]]:
        """Generate a random levels change."""
        channels = {STATE_WARM_WHITE}
        if COLOR_MODES_RGB.intersection(self.color_modes):
            channels = {STATE_RED, STATE_GREEN, STATE_BLUE}
        elif COLOR_MODE_CCT in self.color_modes:
            channels = {STATE_WARM_WHITE, STATE_COOL_WHITE}
        return self._generate_levels_change(
            {
                channel: random.randint(0, 255) if channel in channels else None
                for channel in CHANNEL_STATES
            }
        )

    def _generate_levels_change(
        self,
        channels: dict[str, int | None],
        persist: bool = True,
        brightness: int | None = None,
    ) -> tuple[list[bytearray], dict[str, int]]:
        """Generate the levels change request."""
        channel_map = self.model_data.channel_map
        if channel_map:
            mapped_channels = {
                channel: channels[channel_map.get(channel, channel)]
                for channel in channels
            }
        else:
            mapped_channels = channels

        r = mapped_channels[STATE_RED]
        g = mapped_channels[STATE_GREEN]
        b = mapped_channels[STATE_BLUE]
        w = mapped_channels[STATE_WARM_WHITE]
        w2 = mapped_channels[STATE_COOL_WHITE]

        if (r or g or b) and (w or w2) and not self.rgbwcapable:
            raise ValueError("RGB&CW command sent to non-RGB&CW device")

        if brightness is not None and r is not None and g is not None and b is not None:
            (r, g, b) = self._calculateBrightness((r, g, b), brightness)

        r_value = None if r is None else int(r)
        g_value = None if g is None else int(g)
        b_value = None if b is None else int(b)
        w_value = None if w is None else int(w)
        # ProtocolLEDENET9Byte devices support two white outputs for cold and warm.
        if w2 is None:
            if w is not None and self.color_mode in {COLOR_MODE_CCT, COLOR_MODE_RGBWW}:
                # If we're only setting a single white value, we preserve the cold white value
                w2_value: int | None = self.cold_white
            else:
                # If we're only setting a single white value, we set the second output to be the same as the first
                w2_value = w_value
        else:
            w2_value = int(w2)

        # color / white write mode changed in Firmware 11 (25 byte)
        if TYPE_CHECKING:
            assert self._protocol is not None, "Protocol should not be None"
        level_write_mode = self._protocol.level_write_modes

        write_mode = level_write_mode.ALL
        # rgbwprotocol always overwrite both color & whites
        if not self.rgbwprotocol:
            if w is None and w2 is None:
                write_mode = level_write_mode.COLORS
            elif r is None and g is None and b is None:
                write_mode = level_write_mode.WHITES

        assert self._protocol is not None
        msgs = self._protocol.construct_levels_change(
            persist, r_value, g_value, b_value, w_value, w2_value, write_mode
        )
        updates = {}
        multi_mode = self.multi_color_mode
        if multi_mode or write_mode in self._protocol.get_write_all_colors():
            updates.update(
                {"red": r_value or 0, "green": g_value or 0, "blue": b_value or 0}
            )
        if multi_mode or write_mode in self._protocol.get_write_all_whites():
            updates.update({"warm_white": w_value or 0, "cool_white": w2_value or 0})
        return msgs, updates

    def _set_transition_complete_time(self) -> None:
        """Set the time we expect the transition will be completed.

        Devices fade to a specific state so we want to avoid
        consuming state updates into self.raw_state while a transition
        is in progress as this will provide unexpected results
        and the brightness values will be wrong until
        the transition completes.
        """
        assert self.raw_state is not None
        latency = STATE_CHANGE_LATENCY
        if self.protocol in ADDRESSABLE_PROTOCOLS:
            latency = ADDRESSABLE_STATE_CHANGE_LATENCY
        transition_time = latency + utils.speedToDelay(self.raw_state.speed) / 100
        self._transition_complete_time = time.monotonic() + transition_time
        _LOGGER.debug(
            "%s: Transition time is %s, set _transition_complete_time to %s",
            self.ipaddr,
            transition_time,
            self._transition_complete_time,
        )
        # If we are doing a state transition cancel and preset pattern transition
        self._preset_pattern_transition_complete_time = NEVER_TIME

    def _set_preset_pattern_transition_complete_time(self) -> None:
        """Set the time we expect the preset_pattern transition will be completed."""
        assert self.raw_state is not None
        self._preset_pattern_transition_complete_time = (
            time.monotonic() + PRESET_PATTERN_CHANGE_LATENCY
        )
        _LOGGER.debug(
            "%s: Mode transition time is %s, set _preset_pattern_transition_complete_time to %s",
            self.ipaddr,
            PRESET_PATTERN_CHANGE_LATENCY,
            self._preset_pattern_transition_complete_time,
        )

    def _set_power_transition_complete_time(self) -> None:
        """Set the time we expect the power transition will be completed."""
        assert self.raw_state is not None
        self._power_state_transition_complete_time = (
            time.monotonic() + POWER_STATE_CHANGE_LATENCY
        )
        _LOGGER.debug(
            "%s: Mode transition time is %s, set _power_state_transition_complete_time to %s",
            self.ipaddr,
            POWER_STATE_CHANGE_LATENCY,
            self._power_state_transition_complete_time,
        )

    def getRgb(self) -> tuple[int, int, int]:
        if self.color_mode not in COLOR_MODES_RGB:
            return (255, 255, 255)
        return self.rgb

    @property
    def rgb(self) -> tuple[int, int, int]:
        assert self.raw_state is not None
        raw_state = self.raw_state
        return (raw_state.red, raw_state.green, raw_state.blue)

    @property
    def rgb_unscaled(self) -> tuple[int, int, int]:
        """Return the unscaled RGB."""
        r, g, b = self.rgb
        hsv = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        r_p, g_p, b_p = colorsys.hsv_to_rgb(hsv[0], hsv[1], 1)
        return round(r_p * 255), round(g_p * 255), round(b_p * 255)

    def _calculateBrightness(
        self, rgb: tuple[int, int, int], level: int
    ) -> tuple[int, int, int]:
        hsv = colorsys.rgb_to_hsv(*rgb)
        r, g, b = colorsys.hsv_to_rgb(hsv[0], hsv[1], level)
        return int(r), int(g), int(b)

    def setProtocol(self, protocol: str) -> None:
        cls = PROTOCOL_NAME_TO_CLS.get(protocol)
        if cls is None:
            raise ValueError(f"Invalid protocol: {protocol}")
        self._protocol = cls()  # type: ignore

    def _set_protocol_from_msg(
        self,
        full_msg: bytes,
        fallback_protocol: str,
    ) -> None:
        self._model_num = full_msg[1]
        self._model_data = get_model(self._model_num, fallback_protocol)
        version_num = full_msg[10] if len(full_msg) > 10 else 1
        self.setProtocol(self._model_data.protocol_for_version_num(version_num))

    def _generate_preset_pattern(
        self, pattern: int, speed: int, brightness: int
    ) -> bytearray:
        """Generate the preset pattern protocol bytes."""
        protocol = self.protocol
        if protocol in OLD_EFFECTS_PROTOCOLS:
            if pattern not in ORIGINAL_ADDRESSABLE_EFFECT_ID_NAME:
                raise ValueError("Pattern must be between 1 and 302")
        elif protocol in NEW_EFFECTS_PROTOCOLS:
            if pattern not in ADDRESSABLE_EFFECT_ID_NAME:
                raise ValueError("Pattern must be between 1 and 100")
        elif protocol in CHRISTMAS_EFFECTS_PROTOCOLS:
            if pattern not in CHRISTMAS_ADDRESSABLE_EFFECT_ID_NAME:
                raise ValueError("Pattern must be between 1 and 100")
        else:
            PresetPattern.valid_or_raise(pattern)
        if not (1 <= brightness <= 100):
            raise ValueError("Brightness must be between 1 and 100")
        self._last_effect_brightness = brightness
        assert self._protocol is not None
        return self._protocol.construct_preset_pattern(pattern, speed, brightness)

    def _generate_custom_patterm(
        self, rgb_list: list[tuple[int, int, int]], speed: int, transition_type: str
    ) -> bytearray:
        """Generate the custom pattern protocol bytes."""
        # truncate if more than 16
        if len(rgb_list) > 16:
            _LOGGER.warning(
                "Too many colors in %s, truncating list to %s", len(rgb_list), 16
            )
            del rgb_list[16:]
        # quit if too few
        if len(rgb_list) == 0:
            raise ValueError("setCustomPattern requires at least one color tuples")

        assert self._protocol is not None
        return self._protocol.construct_custom_effect(rgb_list, speed, transition_type)

    def _effect_to_pattern(self, effect: str) -> int:
        """Convert an effect to a pattern code."""
        protocol = self.protocol
        if protocol in CHRISTMAS_EFFECTS_PROTOCOLS:
            return CHRISTMAS_ADDRESSABLE_EFFECT_NAME_ID[effect]
        if protocol in NEW_EFFECTS_PROTOCOLS:
            return ADDRESSABLE_EFFECT_NAME_ID[effect]
        if protocol in OLD_EFFECTS_PROTOCOLS:
            return ORIGINAL_ADDRESSABLE_EFFECT_NAME_ID[effect]
        return PresetPattern.str_to_val(effect)

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return diagnostics for the device."""
        data: dict[str, Any] = {"device_state": {}, "last_messages": {}}
        last_messages = data["last_messages"]
        for name, msg in self._last_message.items():
            last_messages[name] = " ".join(f"0x{x:02X}" for x in msg)
        device_state = data["device_state"]
        for name in dir(self):
            if name.startswith("_") or name == "diagnostics" or not hasattr(self, name):
                continue
            value: Any = getattr(self, name)
            if is_dataclass(value):
                value = asdict(value)  # type: ignore[arg-type]
            if hasattr(value, "value"):
                value = value.value
            if value is None or isinstance(value, SERIALIZABLE_TYPES):
                device_state[name] = value
        return data
