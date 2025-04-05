from __future__ import annotations

import asyncio
import logging
import time
from asyncio import ALL_COMPLETED, FIRST_COMPLETED
from datetime import datetime
from typing import Any, Callable

from .aioprotocol import AIOLEDENETProtocol
from .aioscanner import AIOBulbScanner
from .aioutils import asyncio_timeout
from .base_device import (
    ALL_ADDRESSABLE_PROTOCOLS,
    ALL_IC_PROTOCOLS,
    DeviceType,
    DeviceUnavailableException,
    LEDENETDevice,
)
from .const import (
    COLOR_MODE_CCT,
    COLOR_MODE_DIM,
    COLOR_MODE_RGB,
    COLOR_MODE_RGBW,
    COLOR_MODE_RGBWW,
    EFFECT_MUSIC,
    EFFECT_RANDOM,
    NEVER_TIME,
    PRESET_MUSIC_MODE,
    PUSH_UPDATE_INTERVAL,
    STATE_BLUE,
    STATE_COOL_WHITE,
    STATE_GREEN,
    STATE_RED,
    STATE_WARM_WHITE,
    MultiColorEffects,
)
from .protocol import (
    POWER_RESTORE_BYTES_TO_POWER_RESTORE,
    REMOTE_CONFIG_BYTES_TO_REMOTE_CONFIG,
    LEDENETOriginalRawState,
    LEDENETRawState,
    PowerRestoreState,
    PowerRestoreStates,
    ProtocolLEDENET8Byte,
    ProtocolLEDENETAddressableA3,
    ProtocolLEDENETAddressableChristmas,
    ProtocolLEDENETOriginal,
    RemoteConfig,
)
from .scanner import FluxLEDDiscovery
from .timer import LedTimer
from .utils import color_temp_to_white_levels, rgbw_brightness, rgbww_brightness

_LOGGER = logging.getLogger(__name__)


COMMAND_SPACING_DELAY = 1
MAX_UPDATES_WITHOUT_RESPONSE = 4
DEVICE_CONFIG_WAIT_SECONDS = (
    3.5  # time it takes for the device to respond after a config change
)
POWER_STATE_TIMEOUT = 1.2
POWER_CHANGE_ATTEMPTS = 6


class AIOWifiLedBulb(LEDENETDevice):
    """A LEDENET Wifi bulb device."""

    def __init__(
        self,
        ipaddr: str,
        port: int = 0,
        timeout: float = 7.5,
        discovery: FluxLEDDiscovery | None = None,
    ) -> None:
        """Init and setup the bulb."""
        super().__init__(ipaddr, port, timeout, discovery)
        loop = asyncio.get_running_loop()
        self._connect_lock = asyncio.Lock()
        self._aio_protocol: AIOLEDENETProtocol | None = None
        self._get_time_lock: asyncio.Lock = asyncio.Lock()
        self._get_time_future: asyncio.Future[bool] | None = None
        self._get_timers_lock: asyncio.Lock = asyncio.Lock()
        self._get_timers_future: asyncio.Future[bool] | None = None
        self._timers: list[LedTimer] | None = None
        self._power_restore_future: asyncio.Future[bool] = loop.create_future()
        self._device_config_lock: asyncio.Lock = asyncio.Lock()
        self._device_config_future: asyncio.Future[bool] = loop.create_future()
        self._remote_config_future: asyncio.Future[bool] = loop.create_future()
        self._device_config_setup = False
        self._power_state_lock = asyncio.Lock()
        self._power_state_futures: list[asyncio.Future[bool]] = []
        self._state_futures: list[
            asyncio.Future[LEDENETRawState | LEDENETOriginalRawState]
        ] = []
        self._determine_protocol_future: asyncio.Future[bool] | None = None
        self._updated_callback: Callable[[], None] | None = None
        self._updates_without_response = 0
        self._last_update_time: float = NEVER_TIME
        self._power_restore_state: PowerRestoreStates | None = None
        self._buffer = b""
        self.loop = loop

    @property
    def power_restore_states(self) -> PowerRestoreStates | None:
        """Returns the power restore states for all channels."""
        return self._power_restore_state

    async def async_setup(self, updated_callback: Callable[[], None]) -> None:
        """Setup the connection and fetch initial state."""
        self._updated_callback = updated_callback
        try:
            await self._async_setup()
        except Exception:  # pylint: disable=broad-except
            self._async_stop()
            raise

    async def _async_setup(self) -> None:
        await self._async_determine_protocol()
        assert self._protocol is not None
        if isinstance(self._protocol, ALL_IC_PROTOCOLS):
            await self._async_device_config_setup()
        hardware = self.hardware
        if hardware is not None and hardware.remote_24g_controls:
            await self._async_remote_config_setup()
        if self.device_type == DeviceType.Switch:
            await self._async_switch_setup()
        _LOGGER.debug(
            "%s: device_config: wiring=%s operating_mode=%s",
            self.ipaddr,
            self.wiring,
            self.operating_mode,
        )

    async def _async_remote_config_setup(self) -> None:
        """Setup remote config."""
        assert self._protocol is not None
        await self._async_send_msg(self._protocol.construct_query_remote_config())
        try:
            async with asyncio_timeout(self.timeout):
                await self._remote_config_future
        except asyncio.TimeoutError:
            _LOGGER.warning("%s: Could not determine 2.4ghz remote config", self.ipaddr)

    async def _async_switch_setup(self) -> None:
        """Setup a switch."""
        assert self._protocol is not None
        await self._async_send_msg(self._protocol.construct_power_restore_state_query())
        try:
            async with asyncio_timeout(self.timeout):
                await self._power_restore_future
        except asyncio.TimeoutError:
            self.set_unavailable("Could not determine power restore state")
            raise DeviceUnavailableException(
                f"{self.ipaddr}: Could not determine power restore state"
            )

    async def _async_device_config_setup(self) -> None:
        """Setup an addressable light."""
        assert self._protocol is not None
        if isinstance(self._protocol, ProtocolLEDENETAddressableChristmas):
            self._device_config = self._protocol.parse_strip_setting(b"")
            return

        if self._device_config_setup:
            self._device_config_future = self.loop.create_future()
        self._device_config_setup = True

        assert isinstance(self._protocol, ALL_ADDRESSABLE_PROTOCOLS)
        await self._async_send_msg(self._protocol.construct_request_strip_setting())
        try:
            async with asyncio_timeout(self.timeout):
                await self._device_config_future
        except asyncio.TimeoutError:
            self.set_unavailable("Could not determine number pixels")
            raise DeviceUnavailableException(
                f"{self.ipaddr}: Could not determine number pixels"
            )

    async def async_stop(self) -> None:
        """Shutdown the connection."""
        self._async_stop()

    def _async_stop(self) -> None:
        """Shutdown the connection and mark unavailable."""
        self.set_unavailable("Connection closed")
        self._async_close()
        self._last_update_time = NEVER_TIME

    def _async_close(self) -> None:
        """Close the connection."""
        if self._aio_protocol:
            self._aio_protocol.close()
            self._aio_protocol = None

    async def _async_send_state_query(self) -> None:
        assert self._protocol is not None
        await self._async_send_msg(self._protocol.construct_state_query())

    async def _async_wait_state_change(
        self, futures: list[asyncio.Future[Any]], state: bool, timeout: float
    ) -> bool:
        # If the device requires the two step turn on, we need to wait for the
        # power state and the state update. If its the single step turn on, we
        # only need to wait for any future.
        return_when = ALL_COMPLETED if self.requires_turn_on else FIRST_COMPLETED
        done, _ = await asyncio.wait(futures, timeout=timeout, return_when=return_when)
        return bool(done and self.is_on == state)

    async def _async_set_power_state(
        self, state: bool, accept_any_power_state_response: bool
    ) -> bool:
        assert self._protocol is not None
        power_state_future: asyncio.Future[bool] = self.loop.create_future()
        state_future: asyncio.Future[LEDENETRawState | LEDENETOriginalRawState] = (
            self.loop.create_future()
        )
        self._power_state_futures.append(power_state_future)
        self._state_futures.append(state_future)
        await self._async_send_msg(self._protocol.construct_state_change(state))
        _LOGGER.debug("%s: Waiting for power state response", self.ipaddr)
        if await self._async_wait_state_change(
            [state_future, power_state_future], state, POWER_STATE_TIMEOUT * (3 / 8)
        ):
            return True
        if power_state_future.done() and accept_any_power_state_response:
            # The magic home app will accept any response as success
            # so after a few tries, we do as well.
            return True
        if power_state_future.done() or state_future.done():
            _LOGGER.debug(
                "%s: Bulb power state change taking longer than expected to %s, sending state query",
                self.ipaddr,
                state,
            )
        else:
            _LOGGER.debug(
                "%s: Bulb failed to respond, sending state query", self.ipaddr
            )
        if state_future.done():
            state_future = self.loop.create_future()
            self._state_futures.append(state_future)
        pending: list[asyncio.Future[Any]] = [state_future]
        if not power_state_future.done():
            # If the power state still hasn't responded
            # we want to stop waiting as soon as it does
            pending.append(power_state_future)
        await self._async_send_state_query()
        if await self._async_wait_state_change(
            pending, state, POWER_STATE_TIMEOUT * (5 / 8)
        ):
            return True
        _LOGGER.debug(
            "%s: State query did not return expected power state of %s",
            self.ipaddr,
            state,
        )
        return False

    async def async_turn_on(self) -> bool:
        """Turn on the device."""
        return await self._async_set_power_locked(True)

    async def async_turn_off(self) -> bool:
        """Turn off the device."""
        return await self._async_set_power_locked(False)

    async def _async_set_power_locked(self, state: bool) -> bool:
        async with self._power_state_lock:
            self._power_state_transition_complete_time = NEVER_TIME
            return await self._async_set_power_state_with_retry(state)

    async def _async_set_power_state_with_retry(self, state: bool) -> bool:
        for idx in range(POWER_CHANGE_ATTEMPTS):
            accept_any_power_state_response = idx > 2
            if await self._async_set_power_state(
                state, accept_any_power_state_response
            ):
                _LOGGER.debug(
                    "%s: Completed power state change to %s (%s/%s)",
                    self.ipaddr,
                    state,
                    1 + idx,
                    POWER_CHANGE_ATTEMPTS,
                )
                if accept_any_power_state_response and self.is_on != state:
                    # Sometimes these devices respond with "I turned off" and
                    # they actually turn on when we are requesting to turn on.
                    assert self._protocol is not None
                    byte = self._protocol.on_byte if state else self._protocol.off_byte
                    self._set_power_state(byte)
                    self._set_power_transition_complete_time()
                return True
            _LOGGER.debug(
                "%s: Failed to set power state to %s (%s/%s)",
                self.ipaddr,
                state,
                1 + idx,
                POWER_CHANGE_ATTEMPTS,
            )
        _LOGGER.error(
            "%s: Failed to change power state to %s after %s attempts; Try rebooting the device",
            self.ipaddr,
            state,
            POWER_CHANGE_ATTEMPTS,
        )
        return False

    async def async_set_white_temp(
        self, temperature: int, brightness: int, persist: bool = True
    ) -> None:
        """Set the white tempature."""
        warm, cold = color_temp_to_white_levels(
            temperature, brightness, self.min_temp, self.max_temp
        )
        if self.rgbw_color_temp_support(self.color_modes):
            await self.async_set_levels(cold, cold, cold, warm, 0, persist=persist)
        else:
            await self.async_set_levels(None, None, None, warm, cold, persist=persist)

    async def async_update(self, force: bool = False) -> None:
        """Request an update.

        The callback will be triggered when the state is recieved.
        """
        now = time.monotonic()
        assert self._protocol is not None
        if (
            # If the device is not available from a previous disconnect
            # fall through and try to reconnect send we do the
            # _async_send_state_query
            self.available
            and not force
            and (self._last_update_time + PUSH_UPDATE_INTERVAL) > now
        ):
            if self.is_on and self._protocol.state_push_updates:
                # If the device pushes state updates when on
                # then no need to poll except for the interval
                # to make sure the device is still responding
                return
            if self._protocol.power_push_updates:
                # If the device pushes power updates
                # then no need to poll except for the interval
                # to make sure the device is still responding
                return
        self._last_update_time = now
        if (
            self._aio_protocol
            and self._updates_without_response >= MAX_UPDATES_WITHOUT_RESPONSE
        ):
            self._async_close()
            self.set_unavailable(
                f"device stopped responding after {MAX_UPDATES_WITHOUT_RESPONSE} requests to send state"
            )
            raise DeviceUnavailableException(
                f"{self.ipaddr}: device stopped responding after {MAX_UPDATES_WITHOUT_RESPONSE} requests to send state"
            )
        try:
            await self._async_send_state_query()
        except asyncio.TimeoutError as ex:
            raise DeviceUnavailableException(
                f"{self.ipaddr}: timed out trying to connect after {self.timeout} seconds"
            ) from ex

        self._updates_without_response += 1

    async def async_set_levels(
        self,
        r: int | None = None,
        g: int | None = None,
        b: int | None = None,
        w: int | None = None,
        w2: int | None = None,
        persist: bool = True,
        brightness: int | None = None,
    ) -> None:
        """Set any of the levels."""
        await self._async_process_levels_change(
            *self._generate_levels_change(
                {
                    STATE_RED: r,
                    STATE_GREEN: g,
                    STATE_BLUE: b,
                    STATE_WARM_WHITE: w,
                    STATE_COOL_WHITE: w2,
                },
                persist,
                brightness,
            )
        )

    async def _async_process_levels_change(
        self, msgs: list[bytearray], updates: dict[str, int]
    ) -> None:
        """Process and send a levels change."""
        self._set_transition_complete_time()
        if updates:
            self._replace_raw_state(updates)
        for idx, msg in enumerate(msgs):
            await self._async_send_msg(msg)
            if idx > 0:
                self._process_callbacks()
                await asyncio.sleep(COMMAND_SPACING_DELAY)
                self._set_transition_complete_time()

    async def async_set_preset_pattern(
        self, effect: int, speed: int, brightness: int = 100
    ) -> None:
        """Set a preset pattern on the device."""
        self._set_transition_complete_time()
        await self._async_send_msg(
            self._generate_preset_pattern(effect, speed, brightness)
        )

    async def async_set_custom_pattern(
        self, rgb_list: list[tuple[int, int, int]], speed: int, transition_type: str
    ) -> None:
        """Set a custom pattern on the device."""
        await self._async_send_msg(
            self._generate_custom_patterm(rgb_list, speed, transition_type)
        )

    async def async_set_effect(
        self, effect: str, speed: int, brightness: int = 100
    ) -> None:
        """Set an effect."""
        if effect == EFFECT_RANDOM:
            await self.async_set_random()
            return
        brightness = min(100, max(0, brightness))
        if effect == EFFECT_MUSIC:
            await self.async_set_music_mode(brightness=brightness)
            return
        await self.async_set_preset_pattern(
            self._effect_to_pattern(effect), speed, brightness
        )

    async def async_set_zones(
        self,
        rgb_list: list[tuple[int, int, int]],
        speed: int = 100,
        effect: MultiColorEffects = MultiColorEffects.STATIC,
    ) -> None:
        """Set zones."""
        assert self._protocol is not None
        if not self._protocol.zones:
            raise ValueError("{self.model} does not support zones")
        assert self._device_config is not None
        assert isinstance(
            self._protocol,
            (ProtocolLEDENETAddressableA3, ProtocolLEDENETAddressableChristmas),
        )
        await self._async_send_msg(
            self._protocol.construct_zone_change(
                self._device_config.pixels_per_segment, rgb_list, speed, effect
            )
        )

    async def async_set_music_mode(
        self,
        sensitivity: int | None = 100,
        brightness: int | None = 100,
        mode: int | None = None,
        effect: int | None = None,
        foreground_color: tuple[int, int, int] | None = None,
        background_color: tuple[int, int, int] | None = None,
    ) -> None:
        """Set music mode."""
        assert self._protocol is not None
        if not self.microphone:
            raise ValueError("{self.model} does not have a built-in microphone")
        self._set_preset_pattern_transition_complete_time()
        self._replace_raw_state({"preset_pattern": PRESET_MUSIC_MODE})
        for idx, bytes_send in enumerate(
            self._protocol.construct_music_mode(
                sensitivity or 100,
                brightness or 100,
                mode,
                effect,
                foreground_color or self.rgb,
                background_color,
            )
        ):
            if idx > 0:
                await asyncio.sleep(COMMAND_SPACING_DELAY)
            await self._async_send_msg(bytes_send)

    async def async_set_random(self) -> None:
        """Set levels randomly."""
        await self._async_process_levels_change(*self._generate_random_levels_change())

    async def async_set_brightness(self, brightness: int) -> None:
        """Adjust brightness."""
        effect = self.effect
        if effect:
            effect_brightness = round(brightness / 255 * 100)
            await self.async_set_effect(effect, self.speed, effect_brightness)
            return
        if self.color_mode == COLOR_MODE_CCT:
            await self.async_set_white_temp(self.color_temp, brightness)
            return
        if self.color_mode == COLOR_MODE_RGB:
            await self.async_set_levels(*self.rgb_unscaled, brightness=brightness)
            return
        if self.color_mode == COLOR_MODE_RGBW:
            await self.async_set_levels(*rgbw_brightness(self.rgbw, brightness))
            return
        if self.color_mode == COLOR_MODE_RGBWW:
            await self.async_set_levels(*rgbww_brightness(self.rgbww, brightness))
            return
        if self.color_mode == COLOR_MODE_DIM:
            await self.async_set_levels(w=brightness)
            return

    async def async_enable_remote_access(
        self, remote_access_host: str, remote_access_port: int
    ) -> None:
        """Enable remote access."""
        await AIOBulbScanner().async_enable_remote_access(
            self.ipaddr, remote_access_host, remote_access_port
        )
        self._async_stop()

    async def async_disable_remote_access(self) -> None:
        """Disable remote access."""
        await AIOBulbScanner().async_disable_remote_access(self.ipaddr)
        self._async_stop()

    async def async_reboot(self) -> None:
        """Reboot a device."""
        await AIOBulbScanner().async_reboot(self.ipaddr)
        self._async_stop()

    async def async_set_power_restore(
        self,
        channel1: PowerRestoreState | None = None,
        channel2: PowerRestoreState | None = None,
        channel3: PowerRestoreState | None = None,
        channel4: PowerRestoreState | None = None,
    ) -> None:
        new_power_restore_state = self._power_restore_state
        assert new_power_restore_state is not None
        if channel1 is not None:
            new_power_restore_state.channel1 = channel1
        if channel2 is not None:
            new_power_restore_state.channel2 = channel2
        if channel3 is not None:
            new_power_restore_state.channel3 = channel3
        if channel4 is not None:
            new_power_restore_state.channel4 = channel4
        assert self._protocol is not None
        await self._async_send_msg(
            self._protocol.construct_power_restore_state_change(new_power_restore_state)
        )

    async def async_set_device_config(
        self,
        operating_mode: str | None = None,
        wiring: str | None = None,
        ic_type: str | None = None,  # ic type
        pixels_per_segment: int | None = None,  # pixels per segment
        segments: int | None = None,  # number of segments
        music_pixels_per_segment: int | None = None,  # music pixels per segment
        music_segments: int | None = None,  # number of music segments
    ) -> None:
        """Set device configuration."""
        # Since Home Assistant will modify one value at a time,
        # we need to lock, and then update so the previous value
        # modification does not get trampled in the event they
        # change two values before the first one has been updated
        async with self._device_config_lock:
            device_config = self.model_data.device_config
            ic_type_to_num = device_config.ic_type_to_num
            operating_mode_to_num = device_config.operating_mode_to_num
            if self._device_config is not None:
                wiring_to_num = self._device_config.wiring_to_num
            else:
                wiring_to_num = device_config.wiring_to_num
            operating_mode_num = (
                self.operating_mode_num
                if operating_mode is None
                else operating_mode_to_num[operating_mode]
            )
            wiring_num = self.wiring_num if wiring is None else wiring_to_num[wiring]
            ic_type_num = (
                self.ic_type_num if ic_type is None else ic_type_to_num[ic_type]
            )
            assert self._protocol is not None
            assert not isinstance(self._protocol, ProtocolLEDENETOriginal)
            await self._async_send_msg(
                self._protocol.construct_device_config(
                    operating_mode_num,
                    wiring_num,
                    ic_type_num,
                    pixels_per_segment or self.pixels_per_segment,
                    segments or self.segments,
                    music_pixels_per_segment or self.music_pixels_per_segment,
                    music_segments or self.music_segments,
                )
            )
            if isinstance(self._protocol, ALL_IC_PROTOCOLS):
                await self._async_device_config_resync()

    async def async_unpair_remotes(self) -> None:
        """Unpair 2.4ghz remotes."""
        assert self._protocol is not None
        if self.paired_remotes is None:
            raise ValueError("{self.model} does support unpairing remotes")
        await self._async_send_msg(self._protocol.construct_unpair_remotes())
        await self._async_send_msg(self._protocol.construct_query_remote_config())

    async def async_config_remotes(self, remote_config: RemoteConfig) -> None:
        """Change remote config."""
        assert self._protocol is not None
        if self.paired_remotes is None:
            raise ValueError("{self.model} does support unpairing remotes")
        await self._async_send_msg(
            self._protocol.construct_remote_config(remote_config)
        )
        await self._async_send_msg(self._protocol.construct_query_remote_config())

    async def async_get_time(self) -> datetime | None:
        """Get the current time."""
        assert self._protocol is not None
        await self._async_send_msg(self._protocol.construct_get_time())
        async with self._get_time_lock:
            self._get_time_future = self.loop.create_future()
            try:
                async with asyncio_timeout(self.timeout):
                    await self._get_time_future
            except asyncio.TimeoutError:
                _LOGGER.warning("%s: Could not get time from the device", self.ipaddr)
                return None
            return self._last_time

    async def async_get_timers(self) -> list[LedTimer] | None:
        """Get the timers."""
        assert self._protocol is not None
        if isinstance(self._protocol, ProtocolLEDENETOriginal):
            led_timers: list[LedTimer] = []
            return led_timers
        await self._async_send_msg(self._protocol.construct_get_timers())
        async with self._get_timers_lock:
            self._get_timers_future = self.loop.create_future()
            try:
                async with asyncio_timeout(self.timeout):
                    await self._get_timers_future
            except asyncio.TimeoutError:
                _LOGGER.warning("%s: Could not get timers from the device", self.ipaddr)
                return None
            return self._timers

    async def async_set_timers(self, timer_list: list[LedTimer]) -> None:
        """Set the timers."""
        assert self._protocol is not None
        await self._async_send_msg(self._protocol.construct_set_timers(timer_list))

    async def async_set_time(self, time: datetime | None = None) -> None:
        """Set the current time."""
        assert self._protocol is not None
        await self._async_send_msg(self._protocol.construct_set_time(time))

    async def _async_device_config_resync(self) -> None:
        await asyncio.sleep(DEVICE_CONFIG_WAIT_SECONDS)
        await self._async_device_config_setup()

    async def _async_connect(self) -> None:
        """Create connection."""
        async with asyncio_timeout(self.timeout):
            _, self._aio_protocol = await self.loop.create_connection(
                lambda: AIOLEDENETProtocol(
                    self._async_data_recieved, self._async_connection_lost
                ),
                self.ipaddr,
                self.port,
            )

    def _async_connection_lost(self, exc: Exception | None) -> None:
        """Called when the connection is lost."""
        self._aio_protocol = None
        self.set_unavailable("Connection lost")

    def _async_data_recieved(self, data: bytes) -> None:
        """New data on the socket."""
        assert self._protocol is not None
        assert self._aio_protocol is not None
        start_empty_buffer = not self._buffer
        self._buffer += data
        self._updates_without_response = 0
        msg_length = len(self._buffer)
        while msg_length:
            expected_length = self._protocol.expected_response_length(self._buffer)
            if msg_length < expected_length:
                # need more bytes
                return
            msg = self._buffer[:expected_length]
            self._buffer = self._buffer[expected_length:]
            msg_length = len(self._buffer)
            if not start_empty_buffer:
                _LOGGER.debug(
                    "%s <= Reassembled (%s) (%d)",
                    self._aio_protocol.peername,
                    " ".join(f"0x{x:02X}" for x in msg),
                    len(msg),
                )
            self._async_process_message(msg)

    def _async_process_state_response(self, msg: bytes) -> bool:
        if (
            self._determine_protocol_future
            and not self._determine_protocol_future.done()
        ):
            assert self._protocol is not None
            self._set_protocol_from_msg(msg, self._protocol.name)
            self._determine_protocol_future.set_result(True)
        return self.process_state_response(msg)

    def _async_process_message(self, msg: bytes) -> None:
        """Process a full message (maybe reassembled)."""
        assert self._protocol is not None
        self.set_available(f"Received message {msg.hex()}")
        prev_state = self.raw_state
        changed_state = False
        if self._protocol.is_valid_outer_message(msg):
            msg = self._protocol.extract_inner_message(msg)

        if self._protocol.is_valid_state_response(msg):
            self._last_message["state"] = msg
            self._async_process_state_response(msg)
            self._process_state_futures()
        elif self._protocol.is_valid_extended_state_response(msg):
            self._last_message["extended_state"] = msg
            self.process_extended_state_response(msg)
        elif self._protocol.is_valid_power_state_response(msg):
            self._last_message["power_state"] = msg
            self.process_power_state_response(msg)
            self._process_power_futures()
        elif self._protocol.is_valid_get_time_response(msg):
            self._last_message["get_time"] = msg
            self.process_time_response(msg)
        elif self._protocol.is_valid_timers_response(msg):
            self._last_message["timers"] = msg
            self.process_timers_response(msg)
            changed_state = True
        elif self._protocol.is_valid_device_config_response(msg):
            self._last_message["device_config"] = msg
            self.process_device_config_response(msg)
            changed_state = True
        elif self._protocol.is_valid_power_restore_state_response(msg):
            self._last_message["power_restore_state"] = msg
            self.process_power_restore_state_response(msg)
        elif self._protocol.is_valid_remote_config_response(msg):
            self._last_message["remote_config"] = msg
            self.process_remote_config_response(msg)
            changed_state = True
        else:
            self._last_message["unknown"] = msg
            _LOGGER.debug(
                "%s: Ignoring unknown message: %s",
                self.ipaddr,
                " ".join(f"0x{x:02X}" for x in msg),
            )
            return
        if not changed_state and self.raw_state == prev_state:
            return
        self._process_callbacks()

    def _process_state_futures(self) -> None:
        """Process power future responses."""
        assert self.raw_state is not None
        for future in self._state_futures:
            if not future.done():
                future.set_result(self.raw_state)
        self._state_futures.clear()

    def _process_power_futures(self) -> None:
        """Process power future responses."""
        for future in self._power_state_futures:
            if not future.done():
                future.set_result(self.is_on)
        self._power_state_futures.clear()

    def _process_callbacks(self) -> None:
        """Called when state changes."""
        assert self._updated_callback is not None
        try:
            self._updated_callback()
        except Exception as ex:  # pylint: disable=broad-except
            _LOGGER.exception("Error while calling callback: %s", ex)

    def process_power_restore_state_response(self, msg: bytes) -> None:
        """Process a power restore state response.
        Power on state always off
        f0 32 ff f0 f0 f0 f1
        Power on state always on
        f0 32 0f f0 f0 f0 01
        Power on state keep last state
        f0 32 f0 f0 f0 f0 e2
        """
        self._power_restore_state = PowerRestoreStates(
            channel1=POWER_RESTORE_BYTES_TO_POWER_RESTORE.get(msg[2]),
            channel2=POWER_RESTORE_BYTES_TO_POWER_RESTORE.get(msg[3]),
            channel3=POWER_RESTORE_BYTES_TO_POWER_RESTORE.get(msg[4]),
            channel4=POWER_RESTORE_BYTES_TO_POWER_RESTORE.get(msg[5]),
        )
        if not self._power_restore_future.done():
            self._power_restore_future.set_result(True)

    def process_device_config_response(self, msg: bytes) -> None:
        """Process an IC (strip config) response."""
        super().process_device_config_response(msg)
        if not self._device_config_future.done():
            self._device_config_future.set_result(True)

    def process_time_response(self, msg: bytes) -> None:
        """Process an time response."""
        assert self._protocol is not None
        self._last_time = self._protocol.parse_get_time(msg)
        if self._get_time_future and not self._get_time_future.done():
            self._get_time_future.set_result(True)

    def process_timers_response(self, msg: bytes) -> None:
        """Process an timers response."""
        assert self._protocol is not None
        self._timers = self._protocol.parse_get_timers(msg)
        if self._get_timers_future and not self._get_timers_future.done():
            self._get_timers_future.set_result(True)

    def process_remote_config_response(self, msg: bytes) -> None:
        """Process a 2.4ghz remote config response."""
        # 2b 03 00 02 00 00 00 00 00 00 00 00 00 30
        #  0  1  2  3
        self._paired_remotes = msg[3]
        self._remote_config = REMOTE_CONFIG_BYTES_TO_REMOTE_CONFIG.get(msg[1])
        _LOGGER.debug(
            "%s: remote_config: config=%s paired_remotes=%s",
            self.ipaddr,
            self._remote_config,
            self._paired_remotes,
        )
        if not self._remote_config_future.done():
            self._remote_config_future.set_result(True)

    async def _async_send_msg(self, msg: bytearray) -> None:
        """Write a message on the socket."""
        if not self._aio_protocol:
            async with self._connect_lock:
                # Check again under the lock
                if not self._aio_protocol:
                    await self._async_connect()
        assert self._aio_protocol is not None
        self._aio_protocol.write(msg)

    async def _async_determine_protocol(self) -> None:
        # determine the type of protocol based of first 2 bytes.
        for protocol_cls in self._protocol_probes():
            protocol = protocol_cls()
            assert isinstance(protocol, (ProtocolLEDENET8Byte, ProtocolLEDENETOriginal))
            self._protocol = protocol
            async with self._connect_lock:
                await self._async_connect()
                assert self._aio_protocol is not None
                self._determine_protocol_future = self.loop.create_future()
                self._aio_protocol.write(protocol.construct_state_query())
                try:
                    async with asyncio_timeout(self.timeout):
                        await self._determine_protocol_future
                except asyncio.TimeoutError:
                    self._async_close()
                    continue
                else:
                    return
        self.set_unavailable("Cannot determine protocol")
        raise DeviceUnavailableException(f"{self.ipaddr}: Cannot determine protocol")
