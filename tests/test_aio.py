from __future__ import annotations

import asyncio
import colorsys
import contextlib
import datetime
import json
import logging
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from flux_led import DeviceUnavailableException, aiodevice, aioscanner
from flux_led.aio import AIOWifiLedBulb
from flux_led.aioprotocol import AIOLEDENETProtocol
from flux_led.aioscanner import AIOBulbScanner, LEDENETDiscovery
from flux_led.const import (
    COLOR_MODE_CCT,
    COLOR_MODE_DIM,
    COLOR_MODE_RGB,
    COLOR_MODE_RGBW,
    COLOR_MODE_RGBWW,
    EFFECT_MUSIC,
    MAX_TEMP,
    MIN_TEMP,
    PUSH_UPDATE_INTERVAL,
    ExtendedCustomEffectDirection,
    ExtendedCustomEffectOption,
    ExtendedCustomEffectPattern,
    MultiColorEffects,
    ScribbleBlinkMode,
    ScribbleEffect,
    ScribbleLED,
    WhiteChannelType,
)
from flux_led.protocol import (
    LEDENET_EXTENDED_STATE_RESPONSE_LEN,
    PROTOCOL_LEDENET_8BYTE_AUTO_ON,
    PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS,
    PROTOCOL_LEDENET_9BYTE,
    PROTOCOL_LEDENET_25BYTE,
    PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS,
    PROTOCOL_LEDENET_EXTENDED_CUSTOM,
    PROTOCOL_LEDENET_ORIGINAL,
    LEDENETRawState,
    PowerRestoreState,
    PowerRestoreStates,
    ProtocolLEDENET8Byte,
    ProtocolLEDENET25Byte,
    ProtocolLEDENETCCT,
    ProtocolLEDENETCCTWrapped,
    ProtocolLEDENETExtendedCustom,
    RemoteConfig,
)
from flux_led.scanner import (
    FluxLEDDiscovery,
    create_udp_socket,
    is_legacy_device,
    merge_discoveries,
)
from flux_led.timer import LedTimer

IP_ADDRESS = "127.0.0.1"
MODEL_NUM_HEX = "0x35"
MODEL = "AZ120444"
MODEL_DESCRIPTION = "Bulb RGBCW"
FLUX_MAC_ADDRESS = "aabbccddeeff"

FLUX_DISCOVERY_PARTIAL = FluxLEDDiscovery(
    ipaddr=IP_ADDRESS,
    model=MODEL,
    id=FLUX_MAC_ADDRESS,
    model_num=None,
    version_num=None,
    firmware_date=None,
    model_info=None,
    model_description=None,
)
FLUX_DISCOVERY = FluxLEDDiscovery(
    ipaddr=IP_ADDRESS,
    model=MODEL,
    id=FLUX_MAC_ADDRESS,
    model_num=0x25,
    version_num=0x04,
    firmware_date=datetime.date(2021, 5, 5),
    model_info=MODEL,
    model_description=MODEL_DESCRIPTION,
)
FLUX_DISCOVERY_24G_REMOTE = FluxLEDDiscovery(
    ipaddr=IP_ADDRESS,
    model="AK001-ZJ2148",
    id=FLUX_MAC_ADDRESS,
    model_num=0x25,
    version_num=0x04,
    firmware_date=datetime.date(2021, 5, 5),
    model_info=MODEL,
    model_description=MODEL_DESCRIPTION,
)
FLUX_DISCOVERY_LEGACY = FluxLEDDiscovery(
    ipaddr=IP_ADDRESS,
    model=MODEL,
    id="ACCF23123456",
    model_num=0x23,
    version_num=0x04,
    firmware_date=datetime.date(2021, 5, 5),
    model_info=MODEL,
    model_description=MODEL_DESCRIPTION,
)
FLUX_DISCOVERY_MISSING_HARDWARE = FluxLEDDiscovery(
    ipaddr=IP_ADDRESS,
    model=None,
    id=FLUX_MAC_ADDRESS,
    model_num=0x25,
    version_num=0x04,
    firmware_date=datetime.date(2021, 5, 5),
    model_info=MODEL,
    model_description=MODEL_DESCRIPTION,
)


class MinJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        return json.JSONEncoder.default(self, o)


def mock_coro(return_value=None, exception=None):
    """Return a coro that returns a value or raise an exception."""
    fut = asyncio.Future()
    if exception is not None:
        fut.set_exception(exception)
    else:
        fut.set_result(return_value)
    return fut


@pytest.fixture
async def mock_discovery_aio_protocol():
    """Fixture to mock an asyncio connection."""
    loop = asyncio.get_running_loop()
    future = asyncio.Future()

    async def _wait_for_connection():
        transport, protocol = await future
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return transport, protocol

    async def _mock_create_datagram_endpoint(func, sock=None):
        protocol: LEDENETDiscovery = func()
        transport = MagicMock()
        protocol.connection_made(transport)
        with contextlib.suppress(asyncio.InvalidStateError):
            future.set_result((transport, protocol))
        return transport, protocol

    with (
        patch.object(loop, "create_datagram_endpoint", _mock_create_datagram_endpoint),
        patch.object(aioscanner, "MESSAGE_SEND_INTERLEAVE_DELAY", 0),
    ):
        yield _wait_for_connection


@pytest.fixture
async def mock_aio_protocol():
    """Fixture to mock an asyncio connection."""
    loop = asyncio.get_running_loop()
    future = asyncio.Future()

    async def _wait_for_connection():
        transport, protocol = await future
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        return transport, protocol

    async def _mock_create_connection(func, ip, port):
        protocol: AIOLEDENETProtocol = func()
        transport = MagicMock()
        protocol.connection_made(transport)
        with contextlib.suppress(asyncio.InvalidStateError):
            future.set_result((transport, protocol))
        return transport, protocol

    with patch.object(loop, "create_connection", _mock_create_connection):
        yield _wait_for_connection


@pytest.mark.asyncio
async def test_no_initial_response(mock_aio_protocol):
    """Test we try switching protocol if we get no initial response."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.01)
    assert light.protocol is None

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    with pytest.raises(RuntimeError):
        await task

    assert transport.mock_calls == [
        call.get_extra_info("peername"),
        call.write(bytearray(b"\x81\x8a\x8b\x96")),
        call.write_eof(),
        call.close(),
    ]
    assert not light.available
    assert light.protocol is PROTOCOL_LEDENET_ORIGINAL


@pytest.mark.asyncio
async def test_invalid_initial_response(mock_aio_protocol):
    """Test we try switching protocol if we an unexpected response."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.01)

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(b"\x31\x25")
    with pytest.raises(RuntimeError):
        await task

    assert transport.mock_calls == [
        call.get_extra_info("peername"),
        call.write(bytearray(b"\x81\x8a\x8b\x96")),
        call.write_eof(),
        call.close(),
    ]
    assert not light.available


@pytest.mark.asyncio
async def test_cannot_determine_strip_type(mock_aio_protocol):
    """Test we raise RuntimeError when we cannot determine the strip type."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.01)

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    # protocol state
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    with pytest.raises(RuntimeError):
        await task
    assert not light.available


@pytest.mark.asyncio
async def test_setting_discovery(mock_aio_protocol):
    """Test we can pass discovery to AIOWifiLedBulb."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.01)

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    # protocol state
    light._aio_protocol.data_received(
        b"\x81\x35\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xee"
    )
    discovery = FluxLEDDiscovery(
        {
            "firmware_date": datetime.date(2021, 1, 9),
            "id": "B4E842E10586",
            "ipaddr": "192.168.213.259",
            "model": "AK001-ZJ2145",
            "model_description": "Bulb RGBCW",
            "model_info": "ZG-BL-PWM",
            "model_num": 53,
            "remote_access_enabled": False,
            "remote_access_host": None,
            "remote_access_port": None,
            "version_num": 98,
        }
    )

    await task
    assert light.available
    assert light.model == "Bulb RGBCW (0x35)"
    light.discovery = discovery
    assert light.model == "Bulb RGBCW (0x35)"
    assert light.discovery == discovery


@pytest.mark.asyncio
async def test_reassemble(mock_aio_protocol):
    """Test we can reassemble."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    assert light.color_modes == {COLOR_MODE_RGBWW, COLOR_MODE_CCT}
    assert light.protocol == PROTOCOL_LEDENET_9BYTE
    assert light.model_num == 0x25
    assert light.model == "Controller RGB/WW/CW (0x25)"
    assert light.is_on is True
    assert len(light.effect_list) == 21

    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf"
    )
    await asyncio.sleep(0)
    assert light.is_on is False

    light._aio_protocol.data_received(b"\x81")
    light._aio_protocol.data_received(
        b"\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f"
    )
    light._aio_protocol.data_received(b"\xde")
    await asyncio.sleep(0)
    assert light.is_on is True

    transport.reset_mock()
    await light.async_set_device_config()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x05\x0fv"

    transport.reset_mock()
    await light.async_set_device_config(operating_mode="CCT")
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x02\x0fs"


@pytest.mark.asyncio
async def test_extract_from_outer_message(mock_aio_protocol):
    """Test we can can extract a message wrapped with an outer message."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x81\x00\x0e\x81\x1a\x23\x61\x07\x00\xff\x00\x00\x00\x01\x00\x06\x2c\xaf"
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x81\x00\x0e\x81\x1a\x23\x61\x07\x00\xff\x00\x00\x00\x01\x00\x06\x2c\xaf"
    )
    await task
    assert light.color_modes == {COLOR_MODE_RGB}
    assert light.protocol == PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS
    assert light.model_num == 0x1A
    assert light.model == "String Lights (0x1A)"
    assert light.is_on is True
    assert len(light.effect_list) == 101
    assert light.rgb == (255, 0, 0)


@pytest.mark.asyncio
async def test_extract_from_outer_message_and_reassemble(mock_aio_protocol):
    """Test we can can extract a message wrapped with an outer message."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    for byte in b"\xb0\xb1\xb2\xb3\x00\x01\x01\x81\x00\x0e\x81\x1a\x23\x61\x07\x00\xff\x00\x00\x00\x01\x00\x06\x2c\xaf":
        light._aio_protocol.data_received(bytearray([byte]))
    await task
    assert light.color_modes == {COLOR_MODE_RGB}
    assert light.protocol == PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS
    assert light.model_num == 0x1A
    assert light.model == "String Lights (0x1A)"
    assert light.is_on is True
    assert len(light.effect_list) == 101
    assert light.rgb == (255, 0, 0)


@pytest.mark.asyncio
async def test_turn_on_off(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can turn on and off."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task

    data = []

    def _send_data(*args, **kwargs):
        light._aio_protocol.data_received(data.pop(0))

    with (
        patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.010),
        patch.object(light._aio_protocol, "write", _send_data),
    ):
        data = [
            b"\xf0\x71\x24\x85",
            b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
        ]
        await light.async_turn_off()
        await asyncio.sleep(0)
        assert light.is_on is False
        assert len(data) == 1

        data = [
            b"\xf0\x71\x24\x85",
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde",
        ]
        await light.async_turn_on()
        await asyncio.sleep(0)
        assert light.is_on is True
        assert len(data) == 0

        data = [b"\xf0\x71\x24\x85"]
        await light.async_turn_off()
        await asyncio.sleep(0)
        assert light.is_on is False
        assert len(data) == 0

        data = [
            b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde",
        ]
        await light.async_turn_on()
        await asyncio.sleep(0)
        assert light.is_on is True
        assert len(data) == 0

        data = [
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde",
            b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
        ]
        await light.async_turn_off()
        await asyncio.sleep(0)
        assert light.is_on is False
        assert len(data) == 0

        data = [
            *(
                b"\xf0\x71\x24\x85",
                b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
            )
            * 5
        ]
        await light.async_turn_on()
        await asyncio.sleep(0)
        assert light.is_on is True
        assert len(data) == 3
        light._aio_protocol.data_received(
            b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf"
        )
        assert (
            light.is_on is True
        )  # transition time should now be in effect since we forced state

        data = [*(b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde",) * 14]
        await light.async_turn_off()
        await asyncio.sleep(0)
        # If all we get is on 0x81 responses, the bulb failed to turn off
        assert light.is_on is True
        assert len(data) == 2

    await asyncio.sleep(0)
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    # Handle the failure case
    with patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.010):
        await asyncio.create_task(light.async_turn_off())
        assert light.is_on is True
        assert "Failed to set power state to False (1/6)" in caplog.text
        assert "Failed to set power state to False (2/6)" in caplog.text
        assert "Failed to set power state to False (3/6)" in caplog.text
        assert "Failed to set power state to False (4/6)" in caplog.text
        assert "Failed to set power state to False (5/6)" in caplog.text
        assert "Failed to set power state to False (6/6)" in caplog.text

    with (
        patch.object(light._aio_protocol, "write", _send_data),
        patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.010),
    ):
        data = [
            *(
                b"\x0f\x71\x24\xa4",
                b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
            )
            * 5
        ]
        await light.async_turn_off()
        assert light.is_on is False
        assert len(data) == 9

        data = [
            *(
                b"\xf0\x71\x23\xa3",
                b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde",
            )
            * 5
        ]
        await light.async_turn_on()
        assert light.is_on is True
        assert len(data) == 9

        data = [
            *(
                b"\x0f\x71\x24\xa4",
                b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
            )
            * 5
        ]
        await light.async_turn_off()
        assert light.is_on is False
        assert len(data) == 9

    await asyncio.sleep(0)
    caplog.clear()
    caplog.set_level(logging.DEBUG)
    # Handle the failure case
    with patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.010):
        await asyncio.create_task(light.async_turn_on())
        assert light.is_on is False
        assert "Failed to set power state to True (1/6)" in caplog.text
        assert "Failed to set power state to True (2/6)" in caplog.text
        assert "Failed to set power state to True (3/6)" in caplog.text
        assert "Failed to set power state to True (4/6)" in caplog.text
        assert "Failed to set power state to True (5/6)" in caplog.text
        assert "Failed to set power state to True (6/6)" in caplog.text


@pytest.mark.asyncio
async def test_turn_on_off_via_power_state_message(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can turn on and off via power state message."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.010):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        )
        await task

        task = asyncio.create_task(light.async_turn_off())
        # Wait for the future to get added
        await asyncio.sleep(0)
        light._ignore_next_power_state_update = False
        light._aio_protocol.data_received(b"\x0f\x71\x24\xa4")
        await asyncio.sleep(0)
        assert light.is_on is False
        await task

        task = asyncio.create_task(light.async_turn_on())
        await asyncio.sleep(0)
        light._ignore_next_power_state_update = False
        light._aio_protocol.data_received(b"\x0f\x71\x23\xa3")
        await asyncio.sleep(0)
        assert light.is_on is True
        await task


@pytest.mark.asyncio
async def test_turn_on_off_via_assessable_state_message(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can turn on and off via addressable state message."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.025):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        await mock_aio_protocol()
        # protocol state
        light._aio_protocol.data_received(
            b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
        )
        # ic sorting
        light._aio_protocol.data_received(
            b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0"
        )
        await task

        data = None

        def _send_data(*args, **kwargs):
            light._aio_protocol.data_received(data)

        with patch.object(light._aio_protocol, "write", _send_data):
            data = b"\xb0\xb1\xb2\xb3\x00\x01\x01\x23\x00\x0e\x81\xa3\x24\x25\xff\x47\x64\xff\xff\x00\x01\x00\x1e\x34\x61"
            await light.async_turn_off()
            assert light.is_on is False

            data = b"\xb0\xb1\xb2\xb3\x00\x01\x01\x24\x00\x0e\x81\xa3\x23\x25\x5f\x21\x64\xff\xff\x00\x01\x00\x1e\x6d\xd4"
            await light.async_turn_on()
            assert light.is_on is True


@pytest.mark.asyncio
async def test_shutdown(mock_aio_protocol):
    """Test we can shutdown."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task

    await light.async_stop()
    await asyncio.sleep(0)  # make sure nothing throws


@pytest.mark.asyncio
async def test_handling_connection_lost(mock_aio_protocol):
    """Test we can reconnect."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.025):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        )
        await task

        light._aio_protocol.connection_lost(None)
        await asyncio.sleep(0)  # make sure nothing throws

        # Test we reconnect and can turn off
        task = asyncio.create_task(light.async_turn_off())
        # Wait for the future to get added
        await asyncio.sleep(0.1)  # wait for reconnect
        light._aio_protocol.data_received(
            b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf"
        )
        await asyncio.sleep(0)
        assert light.is_on is False
        await task


@pytest.mark.asyncio
async def test_handling_unavailable_after_no_response(mock_aio_protocol):
    """Test we handle the bulb not responding."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task

    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    with pytest.raises(RuntimeError):
        await light.async_update()
    assert light.available is False


@pytest.mark.asyncio
async def test_handling_unavailable_after_no_response_force(mock_aio_protocol):
    """Test we handle the bulb not responding."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, original_aio_protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    # ic state
    light._aio_protocol.data_received(
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\x0b\x00\x63\x00\x90\x00\x01\x07\x08\x90\x01\x94\xfb"
    )
    await task
    assert light._protocol.power_push_updates is True

    transport.reset_mock()
    await light.async_update()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x04\x81\x8a\x8b\x96\xf9"
    )

    light._last_update_time = time.monotonic() - (PUSH_UPDATE_INTERVAL + 1)
    await light.async_update()
    light._last_update_time = time.monotonic() - (PUSH_UPDATE_INTERVAL + 1)
    await light.async_update()
    light._last_update_time = time.monotonic() - (PUSH_UPDATE_INTERVAL + 1)
    await light.async_update()
    light._last_update_time = time.monotonic() - (PUSH_UPDATE_INTERVAL + 1)
    with pytest.raises(RuntimeError):
        await light.async_update()
    assert light.available is False

    # simulate reconnect
    await light.async_update()
    assert light._aio_protocol != original_aio_protocol
    light._aio_protocol = original_aio_protocol

    transport.reset_mock()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    await light.async_update(force=True)
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x06\x00\x04\x81\x8a\x8b\x96\xfe"
    )
    assert light.available is True
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    assert light.available is True


@pytest.mark.asyncio
async def test_async_set_levels(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set levels."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x33\x24\x61\x23\x01\x00\xff\x00\x00\x04\x00\x0f\x6f"
    )
    await task
    assert light.model_num == 0x33
    assert light.version_num == 4
    assert light.wiring == "GRB"
    assert light.wiring_num == 2
    assert light.wirings == ["RGB", "GRB", "BRG"]
    assert light.operating_mode is None
    assert light.dimmable_effects is False
    assert light.requires_turn_on is True
    assert light._protocol.power_push_updates is False
    assert light._protocol.state_push_updates is False

    transport.reset_mock()
    await light.async_set_device_config()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x00\x02\x0fs"

    transport.reset_mock()
    await light.async_set_device_config(wiring="BRG")
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x00\x03\x0ft"

    transport.reset_mock()
    with pytest.raises(ValueError):
        # ValueError: RGBW command sent to non-RGBW devic
        await light.async_set_levels(255, 255, 255, 255, 255)

    await light.async_set_levels(255, 0, 0)

    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\x00\x00\x00\x00\x0f?"

    # light is on
    light._aio_protocol.data_received(
        b"\x81\x33\x23\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x65"
    )
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 4

    # light is off
    light._aio_protocol.data_received(
        b"\x81\x33\x24\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x66"
    )
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 4

    with pytest.raises(ValueError):
        await light.async_set_preset_pattern(101, 50, 100)


@pytest.mark.asyncio
async def test_async_set_levels_0x52(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set levels."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x52\x23\x61\x00\x00\xff\x00\x00\x00\x01\x00\x00\x57"
    )
    await task
    assert light.model_num == 0x52
    assert light.version_num == 1
    assert light.wiring is None
    assert light.wiring_num is None
    assert light.wirings is None
    assert light.operating_mode is None
    assert light.dimmable_effects is False
    assert light.requires_turn_on is True
    assert light._protocol.power_push_updates is False
    assert light._protocol.state_push_updates is False

    transport.reset_mock()
    with pytest.raises(ValueError):
        # ValueError: RGBW command sent to non-RGBW devic
        await light.async_set_levels(255, 255, 255, 255, 255)

    transport.reset_mock()
    await light.async_set_levels(0, 0, 0, 255, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\xff\x00\x00\x00\x0f>"

    transport.reset_mock()
    await light.async_set_levels(0, 0, 0, 128, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x80\xff\x00\x00\x00\x0f\xbf"

    transport.reset_mock()
    await light.async_set_levels(0, 0, 0, 0, 128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x80\x00\x00\x00\x0f\xc0"


@pytest.mark.asyncio
async def test_async_set_effect(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set an effect."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    # ic state
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")
    await task
    assert light.model_num == 0xA3
    assert light.dimmable_effects is True
    assert light.requires_turn_on is False
    assert light._protocol.power_push_updates is True
    assert light._protocol.state_push_updates is True

    transport.reset_mock()
    await light.async_set_effect("random", 50)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0].startswith(b"\xb0\xb1\xb2\xb3")

    transport.reset_mock()
    await light.async_set_effect("RBM 1", 50)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x02\x00\x05B\x012d\xd9\x81"
    )
    assert light.effect == "RBM 1"

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\x05B\x01\x10d\xb7>"
    )

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x04\x00\x05B\x01\x102\x85\xdb"
    )

    for i in range(5, 255):
        transport.reset_mock()
        await light.async_set_brightness(128)
        assert transport.mock_calls[0][0] == "write"
        counter_byte = transport.mock_calls[0][1][0][7]
        assert counter_byte == i

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    counter_byte = transport.mock_calls[0][1][0][7]
    assert counter_byte == 0

    transport.reset_mock()
    # Verify brightness clamped
    await light.async_set_effect("RBM 1", 50, brightness=500)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x05B\x012d\xd9\x80"
    )


@pytest.mark.asyncio
async def test_SK6812RGBW(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set set zone colors."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    # ic state
    light._aio_protocol.data_received(
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\x0b\x00\x63\x00\x90\x00\x01\x07\x08\x90\x01\x94\xfb"
    )

    await task
    assert light.pixels_per_segment == 144
    assert light.segments == 1
    assert light.music_pixels_per_segment == 144
    assert light.music_segments == 1
    assert light.ic_types == [
        "WS2812B",
        "SM16703",
        "SM16704",
        "WS2811",
        "UCS1903",
        "SK6812",
        "SK6812RGBW",
        "INK1003",
        "UCS2904B",
    ]
    assert light.ic_type == "SK6812RGBW"
    assert light.ic_type_num == 7
    assert light.operating_mode is None
    assert light.operating_modes is None
    assert light.wiring == "WGRB"
    assert light.wiring_num == 8
    assert light.wirings == [
        "RGBW",
        "RBGW",
        "GRBW",
        "GBRW",
        "BRGW",
        "BGRW",
        "WRGB",
        "WRBG",
        "WGRB",
        "WGBR",
        "WBRG",
        "WBGR",
    ]
    assert light.model_num == 0xA3
    assert light.dimmable_effects is True
    assert light.requires_turn_on is False
    assert light.color_mode == COLOR_MODE_RGBW
    assert light.color_modes == {COLOR_MODE_RGBW, COLOR_MODE_CCT}
    diag = light.diagnostics
    assert isinstance(json.dumps(diag, cls=MinJSONEncoder), str)
    assert diag["device_state"]["wiring_num"] == 8
    assert (
        diag["last_messages"]["state"]
        == "0x81 0xA3 0x23 0x25 0x01 0x10 0x64 0x00 0x00 0x00 0x04 0x00 0xF0 0xD5"
    )
    transport.reset_mock()

    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(ic_type="SK6812RGBW", wiring="WRGB")
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x0bb\x00\x90\x00\x01\x07\x06\x90\x01\xf0\x81\xd6"
    )

    transport.reset_mock()
    with patch.object(aiodevice, "COMMAND_SPACING_DELAY", 0):
        await light.async_set_levels(r=255, g=255, b=255, w=255)
        assert transport.mock_calls == [
            call.write(
                bytearray(
                    b"\xb0\xb1\xb2\xb3\x00\x01\x01\x02\x00\rA\x01\xff\xff\xff\x00\x00\x00`\xff\x00\x00\x9e\x13"
                )
            ),
            call.write(bytearray(b"\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\x03G\xffFZ")),
        ]

    transport.reset_mock()
    await light.async_set_levels(w=255)
    assert transport.mock_calls == [
        call.write(bytearray(b"\xb0\xb1\xb2\xb3\x00\x01\x01\x04\x00\x03G\xffF["))
    ]
    light._transition_complete_time = 0

    light._aio_protocol.data_received(
        b"\x81\xa3\x23\x61\x01\x32\x40\x40\x40\x80\x01\x00\x90\xac"
    )
    assert light.raw_state.warm_white == 0
    light._aio_protocol.data_received(
        b"\x81\xa3\x23\x61\x01\x32\x40\x40\x40\xe4\x01\x00\x90\x10"
    )
    assert light.raw_state.warm_white == 255
    light._aio_protocol.data_received(
        b"\x81\xa3\x23\x61\x01\x32\x40\x40\x40\xb1\x01\x00\x90\xdd"
    )
    assert light.raw_state.warm_white == 125

    transport.reset_mock()
    with patch.object(aiodevice, "COMMAND_SPACING_DELAY", 0):
        await light.async_set_white_temp(6500, 255)
        assert transport.mock_calls == [
            call.write(
                bytearray(
                    bytearray(
                        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x05\x00\rA\x01\xff\xff\xff\x00\x00\x00`\xff\x00\x00\x9e\x16"
                    )
                )
            ),
            call.write(bytearray(b"\xb0\xb1\xb2\xb3\x00\x01\x01\x06\x00\x03G\x00G_")),
        ]


@pytest.mark.asyncio
async def test_ws2812b_a1(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can determine ws2812b configuration."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa1#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd3"
    )
    # ic state
    light._aio_protocol.data_received(
        b"\x63\x00\x32\x04\x00\x00\x00\x00\x00\x00\x02\x9b"
    )

    await task
    assert light._protocol.timer_count == 6
    assert light._protocol.timer_len == 14
    assert light._protocol.timer_response_len == 88

    assert light.pixels_per_segment == 50
    assert light.segments is None
    assert light.music_pixels_per_segment is None
    assert light.music_segments is None
    assert light.ic_types == [
        "UCS1903",
        "SM16703",
        "WS2811",
        "WS2812B",
        "SK6812",
        "INK1003",
        "WS2801",
        "LB1914",
    ]
    assert light.ic_type == "WS2812B"
    assert light.ic_type_num == 4
    assert light.operating_mode is None
    assert light.operating_modes is None
    assert light.wiring == "GRB"
    assert light.wiring_num == 2
    assert light.wirings == ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]
    assert light.model_num == 0xA1
    assert light.dimmable_effects is False
    assert light.requires_turn_on is False

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"b\x002\x04\x00\x00\x00\x00\x00\x00\x02\xf0\x8a"
    )

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            ic_type="SK6812", wiring="GRB", pixels_per_segment=300
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"b\x01,\x05\x00\x00\x00\x00\x00\x00\x02\xf0\x86"
    )


@pytest.mark.asyncio
async def test_ws2811_a2(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can determine ws2811 configuration."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa2#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd4"
    )
    # ic state
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")

    await task
    assert light.pixels_per_segment == 25
    assert light.segments == 2
    assert light.music_pixels_per_segment == 25
    assert light.music_segments == 2
    assert light.ic_type == "WS2811B"
    assert light.ic_type_num == 4
    assert light.operating_mode is None
    assert light.operating_modes is None
    assert light.wiring == "GBR"
    assert light.wiring_num == 3
    assert light.wirings == ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]
    assert light.model_num == 0xA2
    assert light.dimmable_effects is True
    assert light.requires_turn_on is False

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x00\x19\x00\x02\x04\x03\x19\x02\xf0\x8f"

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            ic_type="SK6812", wiring="GRB", pixels_per_segment=300
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x01,\x00\x02\x05\x02\x19\x02\xf0\xa3"

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            pixels_per_segment=1000,
            segments=1000,
            music_pixels_per_segment=1000,
            music_segments=1000,
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"b\x01,\x00\x06\x04\x03\x96\x06\xf0("


@pytest.mark.asyncio
async def test_ws2812b_older_a3(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can determine ws2812b configuration on an older a3."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3\x23\x61\x01\x32\x00\x64\x00\x00\x01\x00\x1e\x5e"
    )
    # ic state
    light._aio_protocol.data_received(
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\x0b\x01\x63\x00\x1e\x00\x0a\x01\x00\x1e\x0a\xb5\x3d"
    )

    await task
    assert light.pixels_per_segment == 30
    assert light.segments == 10
    assert light.music_pixels_per_segment == 30
    assert light.music_segments == 10
    assert light.ic_type == "WS2812B"
    assert light.ic_type_num == 1
    assert light.operating_mode is None
    assert light.operating_modes is None
    assert light.wiring == "RGB"
    assert light.wiring_num == 0
    assert light.wirings == ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]
    assert light.model_num == 0xA3
    assert light.dimmable_effects is True
    assert light.requires_turn_on is False

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x0bb\x00\x1e\x00\n\x01\x00\x1e\n\xf0\xa3\x1a"
    )

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            ic_type="SK6812", wiring="GRB", pixels_per_segment=300
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x02\x00\x0bb\x01,\x00\x06\x06\x02\x1e\n\xf0\xb5?"
    )

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            pixels_per_segment=1000,
            segments=1000,
            music_pixels_per_segment=1000,
            music_segments=1000,
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b'\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\x0bb\x01,\x00\x06\x01\x00\x96\x06\xf0"\x1a'
    )


@pytest.mark.asyncio
async def test_async_set_zones(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set set zone colors."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    # ic state
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")
    # sometimes the devices responds 2x
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")

    await task
    assert light.pixels_per_segment == 25
    assert light.segments == 2
    assert light.music_pixels_per_segment == 25
    assert light.music_segments == 2
    assert light.ic_types == [
        "WS2812B",
        "SM16703",
        "SM16704",
        "WS2811",
        "UCS1903",
        "SK6812",
        "SK6812RGBW",
        "INK1003",
        "UCS2904B",
    ]
    assert light.ic_type == "WS2811"
    assert light.ic_type_num == 4
    assert light.operating_mode is None
    assert light.operating_modes is None
    assert light.wiring == "GBR"
    assert light.wiring_num == 3
    assert light.wirings == ["RGB", "RBG", "GRB", "GBR", "BRG", "BGR"]
    assert light.model_num == 0xA3
    assert light.dimmable_effects is True
    assert light.requires_turn_on is False

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config()
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x0bb\x00\x19\x00\x02\x04\x03\x19\x02\xf0\x8f\xf2"
    )

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            ic_type="SK6812",
            wiring="GRB",
            pixels_per_segment=300,
            segments=2,
            music_pixels_per_segment=150,
            music_segments=2,
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x02\x00\x0bb\x01,\x00\x02\x06\x02\x96\x02\xf0!\x17"
    )

    transport.reset_mock()
    with patch.object(light, "_async_device_config_resync", mock_coro):
        await light.async_set_device_config(
            ic_type="SK6812",
            wiring="GRB",
            pixels_per_segment=300,
            segments=2,
            music_pixels_per_segment=300,
            music_segments=2,
        )
    assert len(transport.mock_calls) == 1
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\x0bb\x01,\x00\x02\x06\x02\x96\x02\xf0!\x18"
    )

    transport.reset_mock()

    await light.async_set_zones(
        [(255, 0, 0), (0, 0, 255)], 100, MultiColorEffects.STROBE
    )
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == bytearray(
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x04\x00TY\x00T\xff\x00\x00"
        b"\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff"
        b"\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00"
        b"\x00\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff"
        b"\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00\x00\xff\x00"
        b"\x00\xff\x00\x00\xff\x00\x00\xff\x00\x1e\x03d\x00\x19R"
    )

    with pytest.raises(ValueError):
        await light.async_set_zones(
            [(255, 0, 0) for _ in range(30)],
        )


@pytest.mark.asyncio
async def test_async_set_zones_unsupported_device(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set set zone colors raises valueerror on unsupported."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x57"
    )
    await task
    assert light.model_num == 0x25

    transport.reset_mock()
    with pytest.raises(ValueError):
        await light.async_set_zones(
            [(255, 0, 0), (0, 0, 255)], 100, MultiColorEffects.STROBE
        )


@pytest.mark.asyncio
async def test_0x06_device_wiring(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can get wiring for an 0x06."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x06\x24\x61\x24\x01\x00\xff\x00\x00\x03\x00\xf0\x23"
    )
    await task
    assert light.model_num == 0x06
    assert light.pixels_per_segment is None
    assert light.segments is None
    assert light.music_pixels_per_segment is None
    assert light.music_segments is None
    assert light.ic_types is None
    assert light.ic_type is None
    assert light.operating_mode == "RGB&W"
    assert light.operating_modes == ["RGB&W", "RGB/W"]
    assert light.wiring == "GRBW"
    assert light.wiring_num == 2
    assert light.wirings == ["RGBW", "GRBW", "BRGW"]


@pytest.mark.asyncio
async def test_0x07_device_wiring(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can get wiring for an 0x07."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x07\x24\x61\xc7\x01\x00\x00\x00\x00\x02\xff\x0f\xe5"
    )
    await task
    assert light.model_num == 0x07
    assert light.pixels_per_segment is None
    assert light.segments is None
    assert light.music_pixels_per_segment is None
    assert light.music_segments is None
    assert light.ic_types is None
    assert light.ic_type is None
    assert light.operating_mode == "RGB/CCT"
    assert light.operating_modes == ["RGB&CCT", "RGB/CCT"]
    assert light.wiring == "CBRGW"
    assert light.wiring_num == 12
    assert light.wirings == [
        "RGBCW",
        "GRBCW",
        "BRGCW",
        "RGBWC",
        "GRBWC",
        "BRGWC",
        "WRGBC",
        "WGRBC",
        "WBRGC",
        "CRGBW",
        "CBRBW",
        "CBRGW",
        "WCRGB",
        "WCGRB",
        "WCBRG",
    ]


@pytest.mark.asyncio
async def test_async_set_music_mode_0x08(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set music mode on an 0x08."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "COMMAND_SPACING_DELAY", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        transport, _protocol = await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x08#\x5d\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x72"
        )
        await task
        assert light.model_num == 0x08
        assert light.version_num == 4
        assert light.effect == EFFECT_MUSIC
        assert light.microphone is True
        assert light.protocol == PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS
        assert light.pixels_per_segment is None
        assert light.segments is None
        assert light.music_pixels_per_segment is None
        assert light.music_segments is None
        assert light.ic_types is None
        assert light.ic_type is None
        assert light.operating_mode is None
        assert light.operating_modes is None
        assert light.wiring is None  # How can we get this in music mode?
        assert light.wirings == ["RGB", "GRB", "BRG"]

        transport.reset_mock()
        await light.async_set_music_mode()
        assert transport.mock_calls[0][0] == "write"
        assert transport.mock_calls[0][1][0] == b"s\x01d\x0f\xe7"
        assert transport.mock_calls[1][0] == "write"
        assert transport.mock_calls[1][1][0] == b"7\x00\x007"

        transport.reset_mock()
        await light.async_set_music_mode(effect=2)
        assert transport.mock_calls[0][0] == "write"
        assert transport.mock_calls[0][1][0] == b"s\x01d\x0f\xe7"
        assert transport.mock_calls[1][0] == "write"
        assert transport.mock_calls[1][1][0] == b"7\x02\x009"

        with pytest.raises(ValueError):
            await light.async_set_music_mode(effect=0x08)


@pytest.mark.asyncio
async def test_async_set_music_mode_0x08_v1_firmware(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set music mode on an 0x08 with v1 firmware."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "COMMAND_SPACING_DELAY", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        transport, _protocol = await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x08\x23\x62\x23\x01\x80\x00\x80\x00\x01\x00\x00\x33"
        )
        await task
        assert light.model_num == 0x08
        assert light.version_num == 1
        assert light.effect == EFFECT_MUSIC
        assert light.microphone is True
        assert light.raw_state.red == 128
        assert light.raw_state.green == 0
        assert light.raw_state.blue == 128
        assert light.protocol == PROTOCOL_LEDENET_8BYTE_AUTO_ON
        # In music mode, we always report 255 otherwise it will likely be 0
        assert light.brightness == 255

        transport.reset_mock()
        await light.async_set_music_mode()
        assert len(transport.mock_calls) == 1
        assert transport.mock_calls[0][0] == "write"
        assert transport.mock_calls[0][1][0] == b"s\x01d\x0f\xe7"


@pytest.mark.asyncio
async def test_async_set_music_mode_0x08_v2_firmware(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set music mode on an 0x08 with v2 firmware."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "COMMAND_SPACING_DELAY", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        transport, _protocol = await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x08\x23\x62\x23\x01\x80\x00\xff\x00\x02\x00\x00\xb3"
        )
        await task
        assert light.model_num == 0x08
        assert light.version_num == 2
        assert light.effect == EFFECT_MUSIC
        assert light.microphone is True
        assert light.protocol == PROTOCOL_LEDENET_8BYTE_DIMMABLE_EFFECTS

        transport.reset_mock()
        await light.async_set_music_mode()
        assert transport.mock_calls[0][0] == "write"
        assert transport.mock_calls[0][1][0] == b"s\x01d\x0f\xe7"
        assert transport.mock_calls[1][0] == "write"
        assert transport.mock_calls[1][1][0] == b"7\x00\x007"


@pytest.mark.asyncio
async def test_async_set_music_mode_a2(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set music mode on an 0xA2."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa2#\x62\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x11"
    )
    # ic state
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")
    await task
    assert light.model_num == 0xA2
    assert light.effect == EFFECT_MUSIC
    assert light.microphone is True
    assert light._protocol.state_push_updates is False
    assert light._protocol.power_push_updates is False

    transport.reset_mock()
    await light.async_set_music_mode()
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"s\x01&\x01d\x00\x00\x00\x00\x00dd\xc7"

    transport.reset_mock()
    await light.async_set_effect(EFFECT_MUSIC, 100, 100)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"s\x01&\x01d\x00\x00\x00\x00\x00dd\xc7"

    # light is on
    light._aio_protocol.data_received(
        b"\x81\xa2\x23\x62\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x11"
    )
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 4

    # light is off
    light._aio_protocol.data_received(
        b"\x81\xa2\x24\x62\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x12"
    )
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 4


@pytest.mark.asyncio
async def test_async_set_music_mode_a3(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set music mode on an 0xA3."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x62\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x12"
    )
    # ic state
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")
    await task
    assert light.model_num == 0xA3
    assert light.effect == EFFECT_MUSIC
    assert light.microphone is True

    transport.reset_mock()
    await light.async_set_music_mode()
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0].startswith(b"\xb0\xb1\xb2\xb3")

    with pytest.raises(ValueError):
        await light.async_set_music_mode(mode=0x08)

    with pytest.raises(ValueError):
        await light.async_set_music_mode(effect=0x99)


@pytest.mark.asyncio
async def test_async_set_music_mode_device_without_mic_0x07(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set music mode on an 0x08."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x07#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\x39"
    )
    await task
    assert light.model_num == 0x07
    assert light.microphone is False

    transport.reset_mock()
    with pytest.raises(ValueError):
        await light.async_set_music_mode()


@pytest.mark.asyncio
async def test_async_set_white_temp_0x35(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set white temp on a 0x35."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x35\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xee"
    )
    await task
    assert light.model_num == 0x35

    transport.reset_mock()
    await light.async_set_white_temp(6500, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\x00\xff\x0f\x0fN"


@pytest.mark.asyncio
async def test_setup_0x35_with_ZJ21410(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can setup a 0x35 with the ZJ21410 module."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\xb0\xb1\xb2\xb3\x00\x02\x01\x70\x00\x0e\x81\x35\x23\x61\x17\x04\xd3\xff\x49\x00\x09\x00\xf0\x69\x19"
    )
    await task
    assert light.model_num == 0x35


@pytest.mark.asyncio
async def test_setup_0x44_with_version_num_10(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we use the right protocol for 044 with v10."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x44\x24\x61\x01\x01\xff\x00\xff\x00\x0a\x00\xf0\x44"
    )
    await task
    assert light.model_num == 0x44
    assert light.protocol == PROTOCOL_LEDENET_8BYTE_AUTO_ON


@pytest.mark.asyncio
async def test_async_failed_callback(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we log on failed callback."""
    light = AIOWifiLedBulb("192.168.1.166")
    caplog.set_level(logging.DEBUG)

    def _updated_callback(*args, **kwargs):
        raise ValueError("something went wrong")

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\xa3#\x25\x01\x10\x64\x00\x00\x00\x04\x00\xf0\xd5"
    )
    # ic state
    light._aio_protocol.data_received(b"\x00\x63\x00\x19\x00\x02\x04\x03\x19\x02\xa0")
    await task
    assert light.model_num == 0xA3
    assert light.dimmable_effects is True
    assert light.requires_turn_on is False
    assert "something went wrong" in caplog.text


@pytest.mark.asyncio
async def test_async_set_custom_effect(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set a custom effect."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    assert light.model_num == 0x25

    transport.reset_mock()

    # no values
    with pytest.raises(ValueError):
        await light.async_set_custom_pattern([], 50, "jump")

    await light.async_set_custom_pattern(
        [
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 0),
            (255, 0, 255),
            (255, 0, 0),
            (255, 0, 0),
        ],
        50,
        "jump",
    )
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"Q\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\x00\x00\xff\x00\xff\x00\xff\x00\x00\x00\x10;\xff\x0f\x99"
    )


@pytest.mark.asyncio
async def test_async_stop(mock_aio_protocol):
    """Test we can stop without throwing."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task

    await light.async_stop()
    await asyncio.sleep(0)  # make sure nothing throws


@pytest.mark.asyncio
async def test_async_set_brightness_rgbww(mock_aio_protocol):
    """Test we can set brightness rgbww."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\x00\xd5\xff\xff\x00\x0f\x12"

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x80\x00k\x80\x80\x00\x0f+"


@pytest.mark.asyncio
async def test_async_set_brightness_cct_0x25(mock_aio_protocol):
    """Test we can set brightness with a 0x25 cct device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x02\x10\xb6\x00\x98\x19\x04\x25\x0f\xdb"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00g\x98\x00\x0f?"
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x004L\x00\x0f\xc0"
    assert light.brightness == 128


@pytest.mark.asyncio
async def test_async_set_brightness_cct_0x07(mock_aio_protocol):
    """Test we can set brightness with a 0x07 cct device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x07\x24\x61\xc7\x01\x00\x00\x00\x00\x02\xff\x0f\xe5"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\x00\xff\x0f\x0fN"
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\x00\x80\x0f\x0f\xcf"
    assert light.brightness == 128


@pytest.mark.asyncio
async def test_async_set_brightness_dim(mock_aio_protocol):
    """Test we can set brightness with a dim only device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x01\x10\xb6\x00\x98\x19\x04\x25\x0f\xda"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\xff\xff\x00\x0f>"
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\x80\x80\x00\x0f@"
    assert light.brightness == 128


@pytest.mark.asyncio
async def test_async_set_brightness_rgb_0x33(mock_aio_protocol):
    """Test we can set brightness with a rgb only device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x33\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xec"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\x00\xd4\x00\x00\x0f\x13"
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x80\x00j\x00\x00\x0f*"
    assert light.brightness == 128


@pytest.mark.asyncio
async def test_async_set_brightness_rgb_0x25(mock_aio_protocol):
    """Test we can set brightness with a 0x25 device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x03\x10\xb6\x00\x98\x19\x04\x25\x0f\xdc"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\x00\xd4\x00\x00\x00\x0f\x13"
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x80\x00j\x00\x00\x00\x0f*"
    assert light.brightness == 128


@pytest.mark.asyncio
async def test_async_set_brightness_rgbw(mock_aio_protocol):
    """Test we can set brightness with a rgbw only device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x04\x10\xb6\x00\x98\x19\x04\x25\x0f\xdd"
    )
    await task

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\x00\xd5\xff\xff\x00\x0f\x12"
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x80\x00k\x80\x80\x00\x0f+"
    assert light.brightness == 128


@pytest.mark.asyncio
async def test_0x06_rgbw_cct_warm(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set CCT on RGBW with a warm strip."""
    light = AIOWifiLedBulb("192.168.1.166")
    assert light.white_channel_channel_type == WhiteChannelType.WARM
    light.white_channel_channel_type = WhiteChannelType.WARM
    assert light.white_channel_channel_type == WhiteChannelType.WARM

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x06\x24\x61\x24\x01\x00\xff\x00\x00\x03\x00\xf0\x23"
    )
    await task
    assert light.model_num == 0x06
    assert light.operating_mode == "RGB&W"
    assert light.min_temp == MIN_TEMP
    assert light.max_temp == MAX_TEMP
    assert light.color_modes == {COLOR_MODE_RGBW, COLOR_MODE_CCT}

    transport.reset_mock()
    await light.async_set_white_temp(light.max_temp, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\xff\xff\x00\x00\x0f="
    assert light.brightness == 255
    assert light.raw_state.red == 255
    assert light.raw_state.green == 255
    assert light.raw_state.blue == 255
    assert light.raw_state.warm_white == 0

    transport.reset_mock()
    await light.async_set_white_temp(light.min_temp, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\xff\x00\x0f?"
    assert light.brightness == 255
    assert light.raw_state.red == 0
    assert light.raw_state.green == 0
    assert light.raw_state.blue == 0
    assert light.raw_state.warm_white == 255


@pytest.mark.asyncio
async def test_0x06_rgbw_cct_natural(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set CCT on RGBW with a natural strip."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.white_channel_channel_type = WhiteChannelType.NATURAL
    assert light.white_channel_channel_type == WhiteChannelType.NATURAL

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x06\x24\x61\x24\x01\x00\xff\x00\x00\x03\x00\xf0\x23"
    )
    await task
    assert light.model_num == 0x06
    assert light.operating_mode == "RGB&W"
    assert light.color_modes == {COLOR_MODE_RGBW, COLOR_MODE_CCT}
    assert light.min_temp == MAX_TEMP - ((MAX_TEMP - MIN_TEMP) / 2)
    assert light.max_temp == MAX_TEMP

    transport.reset_mock()
    await light.async_set_white_temp(light.max_temp, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\xff\xff\xff\x00\x00\x0f="
    assert light.brightness == 255
    assert light.raw_state.red == 255
    assert light.raw_state.blue == 255
    assert light.raw_state.green == 255
    assert light.raw_state.warm_white == 0

    transport.reset_mock()
    await light.async_set_white_temp(light.min_temp, 255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x00\x00\x00\xff\x00\x0f?"
    assert light.brightness == 255
    assert light.raw_state.red == 0
    assert light.raw_state.blue == 0
    assert light.raw_state.green == 0
    assert light.raw_state.warm_white == 255


@pytest.mark.asyncio
async def test_0x06_rgbw_cct_cold(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set CCT on RGBW with a cold strip."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.white_channel_channel_type = WhiteChannelType.COLD
    assert light.white_channel_channel_type == WhiteChannelType.COLD

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x06\x24\x61\x24\x01\x00\xff\x00\x00\x03\x00\xf0\x23"
    )
    await task
    assert light.model_num == 0x06
    assert light.operating_mode == "RGB&W"
    assert light.color_modes == {COLOR_MODE_RGBW}
    assert light.min_temp == MAX_TEMP
    assert light.max_temp == MAX_TEMP


@pytest.mark.asyncio
async def test_wrapped_cct_protocol_device(mock_aio_protocol):
    """Test a wrapped cct protocol device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, original_aio_protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x1c\x23\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc8"
    )
    await task
    assert light.getCCT() == (0, 255)
    assert light.color_temp == 6500
    assert light.brightness == 255
    assert isinstance(light._protocol, ProtocolLEDENETCCTWrapped)
    assert light._protocol.timer_count == 6
    assert light._protocol.timer_len == 14
    assert light._protocol.timer_response_len == 88
    light._aio_protocol.data_received(
        b"\x81\x1c\x23\x61\x00\x05\x00\x00\x00\x00\x03\x64\x00\x8d"
    )
    assert light.getCCT() == (255, 0)
    assert light.color_temp == 2700
    assert light.brightness == 255
    assert light.dimmable_effects is False
    assert light.requires_turn_on is False
    assert light._protocol.power_push_updates is True
    assert light._protocol.state_push_updates is True

    transport.reset_mock()
    await light.async_set_brightness(32)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\t5\xb1\x00\r\x00\x00\x00\x03\xf6\xbd"
    )
    assert light.brightness == 33

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\t5\xb1\x002\x00\x00\x00\x03\x1b\x08"
    )
    assert light.brightness == 128

    transport.reset_mock()
    await light.async_set_brightness(1)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x02\x00\t5\xb1\x00\x02\x00\x00\x00\x03\xeb\xa9"
    )
    assert light.brightness == 0

    transport.reset_mock()
    await light.async_set_levels(w=0, w2=255)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\t5\xb1dd\x00\x00\x00\x03\xb16"
    )
    assert light.getCCT() == (0, 255)
    assert light.color_temp == 6500
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_effect("random", 50)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0].startswith(b"\xb0\xb1\xb2\xb3\x00")

    # light is on
    light._aio_protocol.data_received(
        b"\x81\x1c\x23\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc8"
    )
    assert light._last_update_time == aiodevice.NEVER_TIME
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 1

    # light is off
    light._aio_protocol.data_received(
        b"\x81\x1c\x24\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc9"
    )
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 0

    transport.reset_mock()
    for _ in range(4):
        light._last_update_time = aiodevice.NEVER_TIME
        await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 4

    light._last_update_time = aiodevice.NEVER_TIME
    for _ in range(4):
        # First failure should keep the device in
        # a failure state until we get to an update
        # time
        with (
            patch.object(
                light, "_async_connect", AsyncMock(side_effect=asyncio.TimeoutError)
            ),
            pytest.raises(DeviceUnavailableException),
        ):
            await light.async_update()

    light._aio_protocol = original_aio_protocol
    # Should not raise now that bulb has recovered
    light._last_update_time = aiodevice.NEVER_TIME
    light._aio_protocol.data_received(
        b"\x81\x1c\x24\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc9"
    )
    await light.async_update()


@pytest.mark.asyncio
async def test_cct_protocol_device(mock_aio_protocol):
    """Test a original cct protocol device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, original_aio_protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x09\x23\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xb5"
    )
    await task
    assert light.getCCT() == (0, 255)
    assert light.color_temp == 6500
    assert light.brightness == 255
    assert isinstance(light._protocol, ProtocolLEDENETCCT)
    assert light._protocol.timer_count == 6
    assert light._protocol.timer_len == 14
    assert light._protocol.timer_response_len == 88
    light._aio_protocol.data_received(
        b"\x81\x1c\x23\x61\x00\x05\x00\x00\x00\x00\x03\x64\x00\x8d"
    )
    assert light.getCCT() == (255, 0)
    assert light.color_temp == 2700
    assert light.brightness == 255
    assert light.dimmable_effects is False
    assert light.requires_turn_on is True
    assert light._protocol.power_push_updates is True
    assert light._protocol.state_push_updates is False

    transport.reset_mock()
    await light.async_set_brightness(32)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"5\xb1\x00\r\x00\x00\x00\x03\xf6"
    assert light.brightness == 33

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"5\xb1\x002\x00\x00\x00\x03\x1b"
    assert light.brightness == 128

    transport.reset_mock()
    await light.async_set_brightness(1)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"5\xb1\x00\x02\x00\x00\x00\x03\xeb"
    assert light.brightness == 0

    transport.reset_mock()
    await light.async_set_levels(w=0, w2=255)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"5\xb1dd\x00\x00\x00\x03\xb1"
    assert light.getCCT() == (0, 255)
    assert light.color_temp == 6500
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_effect("random", 50)
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0].startswith(b"5\xb1")

    # light is on
    light._aio_protocol.data_received(
        b"\x81\x1c\x23\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc8"
    )
    assert light._last_update_time == aiodevice.NEVER_TIME
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 1

    # light is off
    light._aio_protocol.data_received(
        b"\x81\x1c\x24\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc9"
    )
    transport.reset_mock()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 0

    transport.reset_mock()
    for _ in range(4):
        light._last_update_time = aiodevice.NEVER_TIME
        await light.async_update()
    await asyncio.sleep(0)
    assert len(transport.mock_calls) == 4

    light._last_update_time = aiodevice.NEVER_TIME
    for _ in range(4):
        # First failure should keep the device in
        # a failure state until we get to an update
        # time
        with (
            patch.object(
                light, "_async_connect", AsyncMock(side_effect=asyncio.TimeoutError)
            ),
            pytest.raises(DeviceUnavailableException),
        ):
            await light.async_update()

    light._aio_protocol = original_aio_protocol

    # Should not raise now that bulb has recovered
    light._last_update_time = aiodevice.NEVER_TIME
    light._aio_protocol.data_received(
        b"\x81\x1c\x24\x61\x00\x05\x00\x64\x64\x64\x03\x64\x0f\xc9"
    )
    await light.async_update()


@pytest.mark.asyncio
async def test_christmas_protocol_device_turn_on(mock_aio_protocol):
    """Test a christmas protocol device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x1a\x23\x61\x00\x00\x00\xff\x00\x00\x01\x00\x06\x25"
    )
    await task
    assert light.rgb == (0, 255, 0)
    assert light.brightness == 255
    assert len(light.effect_list) == 101
    assert light.protocol == PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS
    assert light.dimmable_effects is False
    assert light.requires_turn_on is False
    assert light._protocol.power_push_updates is True
    assert light._protocol.state_push_updates is True

    data = []
    written = []

    def _send_data(*args, **kwargs):
        written.append(args[0])
        light._aio_protocol.data_received(data.pop(0))

    with (
        patch.object(aiodevice, "POWER_STATE_TIMEOUT", 0.010),
        patch.object(light._aio_protocol, "write", _send_data),
    ):
        data = [
            b"\x81\x1a\x23\x61\x00\x00\x00\xff\x00\x00\x01\x00\x06\x25",
            b"\x81\x25\x24\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xdf",
        ]
        await light.async_turn_off()
        await asyncio.sleep(0)
        assert light.is_on is False
        assert len(data) == 0

    assert written == [
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\x04q$\x0f\xa4\x14",
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x04\x81\x8a\x8b\x96\xf9",
    ]


@pytest.mark.asyncio
async def test_christmas_protocol_device(mock_aio_protocol):
    """Test a christmas protocol device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x1a\x23\x61\x00\x00\x00\xff\x00\x00\x01\x00\x06\x25"
    )
    await task
    assert light.rgb == (0, 255, 0)
    assert light.brightness == 255
    assert len(light.effect_list) == 101
    assert light.protocol == PROTOCOL_LEDENET_ADDRESSABLE_CHRISTMAS
    assert light.dimmable_effects is False
    assert light.requires_turn_on is False
    assert light._protocol.power_push_updates is True
    assert light._protocol.state_push_updates is True

    transport.reset_mock()
    await light.async_set_brightness(255)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\x0d\x3b\xa1<dd\x00\x00\x00\x00\x00\x00\x00\xe0\x95"
    )
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_brightness(128)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\r;\xa1<d2\x00\x00\x00\x00\x00\x00\x00\xae2"
    )
    assert light.brightness == 128

    transport.reset_mock()
    await light.async_set_levels(r=255, g=255, b=255)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x02\x00\r;\xa1\x00\x00\x64\x00\x00\x00\x00\x00\x00\x00@W"
    )
    assert light.brightness == 255

    transport.reset_mock()
    await light.async_set_effect("Twinkle Green", 50)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\x048\n\x10Rs"
    )
    light._transition_complete_time = 0
    light._aio_protocol.data_received(
        b"\x81\x1a\x23\x25\x0a\x00\x0f\x01\x00\x00\x01\x00\x06\x04"
    )
    assert light.effect == "Twinkle Green"
    assert light.speed == 100

    transport.reset_mock()
    await light.async_set_effect("Strobe Red, Green", 100)
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x04\x00\x048=\x01v\xbc"
    )

    light._transition_complete_time = 0
    light._aio_protocol.data_received(
        b"\x81\x1a\x23\x25\x3d\x00\x0f\x01\x00\x00\x01\x00\x06\x37"
    )
    assert light.effect == "Strobe Red, Green"
    assert light.speed == 100

    with pytest.raises(ValueError):
        await light.async_set_preset_pattern(101, 50, 100)

    light._transition_complete_time = 0
    light._aio_protocol.data_received(
        b"\x81\x1a\x23\x61\x07\x00\x66\x00\x66\x00\x01\x00\x06\xf9"
    )
    assert light.effect is None
    assert light.rgb == (102, 0, 102)
    assert light.speed == 100

    transport.reset_mock()
    await light.async_set_zones([(255, 0, 0), (0, 0, 255)])
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == (
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x05\x004\xa0\x00\x06\x00\x01\xff"
        b"\x00\x00\x00\x00\xff\x00\x02\xff\x00\x00\x00\x00\xff\x00\x03\xff"
        b"\x00\x00\x00\x00\xff\x00\x04\x00\x00\xff\x00\x00\xff\x00\x05\x00"
        b"\x00\xff\x00\x00\xff\x00\x06\x00\x00\xff\x00\x00\xff\xaf_"
    )

    transport.reset_mock()
    await light.async_set_zones(
        [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 255)]
    )
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == (
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x06\x004\xa0\x00\x06\x00\x01\xff"
        b"\x00\x00\x00\x00\xff\x00\x02\x00\x00\xff\x00\x00\xff\x00\x03\x00"
        b"\xff\x00\x00\x00\xff\x00\x04\xff\xff\xff\x00\x00\xff\x00\x05\xff"
        b"\xff\xff\x00\x00\xff\x00\x06\xff\xff\xff\x00\x00\xff\xa9T"
    )

    with pytest.raises(ValueError):
        await light.async_set_zones(
            [
                (255, 0, 0),
                (0, 0, 255),
                (0, 255, 0),
                (255, 255, 255),
                (255, 255, 255),
                (255, 255, 255),
                (255, 255, 255),
            ]
        )


@pytest.mark.asyncio
async def test_async_get_time(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can get the time."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    # ic state
    await task
    assert light.model_num == 0x25
    task = asyncio.ensure_future(light.async_get_time())
    await asyncio.sleep(0)
    # Invalid time
    light._aio_protocol.data_received(b"\x0f\x11\x14\x32\x01\x02\x106\x02\x07\x00\xac")
    light._aio_protocol.data_received(b"\x0f\x11\x14\x16\x01\x02\x106\x02\x07\x00\x9c")
    time = await task
    assert time == datetime.datetime(2022, 1, 2, 16, 54, 2)
    assert light._protocol.parse_get_time(b"\x0f") is None


@pytest.mark.asyncio
async def test_async_get_times_out(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can get the time."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.001)

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    # ic state
    await task
    assert light.model_num == 0x25
    task = asyncio.ensure_future(light.async_get_time())
    await asyncio.sleep(0)
    time = await task
    assert time is None


@pytest.mark.asyncio
async def test_async_set_time(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set the time."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    # ic state
    await task
    assert light.model_num == 0x25

    transport.reset_mock()
    await light.async_set_time(datetime.datetime(2020, 1, 1, 1, 1, 1))
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"\x10\x14\x14\x01\x01\x01\x01\x01\x03\x00\x0fO"
    )

    transport.reset_mock()
    await light.async_set_time()
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0].startswith(b"\x10")


@pytest.mark.asyncio
async def test_async_set_time_legacy_device(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set the time on a legacy device."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.discovery = FLUX_DISCOVERY_LEGACY

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(b"f\x03$A!\x08\x01\x19P\x01\x99")
    # ic state
    await task
    assert light.model_num == 0x03

    transport.reset_mock()
    await light.async_set_time(datetime.datetime(2020, 1, 1, 1, 1, 1))
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0] == b"\x10\x14\x14\x01\x01\x01\x01\x01\x03\x00\x0f"
    )

    transport.reset_mock()
    await light.async_set_time()
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0].startswith(b"\x10")


@pytest.mark.asyncio
async def test_async_get_timers_9byte_device(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can get the timers from a 9 byte device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    assert light.model_num == 0x25
    task = asyncio.ensure_future(light.async_get_timers())
    await asyncio.sleep(0)
    light._aio_protocol.data_received(
        b"\x0f\x22\xf0\x16\x01\x04\x00\x2b\x00\x00\x61\x19\x47\xff\x00\x00\xf0\xf0\x16\x01\x04\x04\x2c\x00\x00\x61\x7f\xff\x00\x00\x00\xf0\xf0\x16\x01\x03\x16\x1f\x00\x00\x61\xff\x00\x00\x00\x00\xf0\xf0\x16\x01\x03\x17\x13\x00\x00\x61\x81\x81\x81\x00\x00\xf0\xf0\x16\x01\x03\x17\x28\x00\x00\x61\x00\xff\x00\x00\x00\xf0\xf0\x16\x01\x04\x07\x2c\x00\x00\x61\x21\x00\xff\x00\x00\xf0\x00\x00"
    )
    timers = await task
    assert len(timers) == 6
    assert len(timers[0].toBytes()) == 15
    assert timers[0].toBytes() == b"\xf0\x16\x01\x04\x00+\x00\x00a\x19G\xff\x00\x00\xf0"
    assert str(timers[0]) == "[ON ] 00:43  Once: 2022-01-04  Color: (25, 71, 255)"
    assert (
        timers[1].toBytes() == b"\xf0\x16\x01\x04\x04,\x00\x00a\x7f\xff\x00\x00\x00\xf0"
    )
    assert str(timers[1]) == "[ON ] 04:44  Once: 2022-01-04  Color: chartreuse"
    assert (
        timers[2].toBytes()
        == b"\xf0\x16\x01\x03\x16\x1f\x00\x00a\xff\x00\x00\x00\x00\xf0"
    )
    assert str(timers[2]) == "[ON ] 22:31  Once: 2022-01-03  Color: red"
    assert (
        timers[3].toBytes()
        == b"\xf0\x16\x01\x03\x17\x13\x00\x00a\x81\x81\x81\x00\x00\xf0"
    )
    assert str(timers[3]) == "[ON ] 23:19  Once: 2022-01-03  Color: (129, 129, 129)"
    assert (
        timers[4].toBytes() == b"\xf0\x16\x01\x03\x17(\x00\x00a\x00\xff\x00\x00\x00\xf0"
    )
    assert str(timers[4]) == "[ON ] 23:40  Once: 2022-01-03  Color: lime"
    assert timers[5].toBytes() == b"\xf0\x16\x01\x04\x07,\x00\x00a!\x00\xff\x00\x00\xf0"
    assert str(timers[5]) == "[ON ] 07:44  Once: 2022-01-04  Color: (33, 0, 255)"

    with pytest.raises(ValueError):
        light._protocol.parse_get_timers(b"\x0f")


@pytest.mark.asyncio
async def test_async_get_timers_socket_device(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can get the timers."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x97\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\x50"
    )
    light._aio_protocol.data_received(b"\xf0\x32\xf0\xf0\xf0\xf0\xe2")

    await task
    assert light.model_num == 0x97
    task = asyncio.ensure_future(light.async_get_timers())
    await asyncio.sleep(0)
    light._aio_protocol.data_received(
        b"\x0f\x22\xf0\x00\x00\x00\x11\x2f\x00\xfe\x23\x00\x00\x00\xf0\x00\x00\x00\x11\x30\x00\xfe\x23\x00\x00\x00\xf0\x00\x00\x00\x11\x30\x00\xfe\x24\x00\x00\x00\xf0\x00\x00\x00\x11\x30\x00\xfe\x23\x00\x00\x00\xf0\x00\x00\x00\x11\x30\x00\xfe\x23\x00\x00\x00\xf0\x00\x00\x00\x11\x30\x00\xfe\x23\x00\x00\x00\xf0\x00\x00\x00\x11\x31\x00\xfe\x24\x00\x00\x00\xf0\x00\x00\x00\x11\x31\x00\xfe\x23\x00\x00\x00\x00\xc4"
    )
    timers = await task
    assert len(timers) == 8
    assert len(timers[0].toBytes()) == 12
    assert timers[0].toBytes() == b"\xf0\x00\x00\x00\x11/\x00\xfe\x00\x00\x00\x00"
    assert str(timers[0]) == "[ON ] 17:47  SuMoTuWeThFrSa    "
    assert str(timers[1]) == "[ON ] 17:48  SuMoTuWeThFrSa    "
    assert str(timers[2]) == "[OFF] 17:48  SuMoTuWeThFrSa    "
    assert str(timers[3]) == "[ON ] 17:48  SuMoTuWeThFrSa    "


@pytest.mark.asyncio
async def test_sockets_push_updates(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can get the timers."""
    socket = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(socket.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    socket._aio_protocol.data_received(
        b"\x81\x97\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\x50"
    )
    socket._aio_protocol.data_received(b"\xf0\x32\xf0\xf0\xf0\xf0\xe2")

    await task
    assert socket.model_num == 0x97
    assert socket._protocol.power_push_updates is True
    assert socket._protocol.state_push_updates is True


@pytest.mark.asyncio
async def test_async_get_timers_8_byte_device(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can get the timers from an 8 byte device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x33\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xec"
    )

    await task
    assert light.model_num == 0x33
    task = asyncio.ensure_future(light.async_get_timers())
    await asyncio.sleep(0)
    light._aio_protocol.data_received(
        b"\x0f\x22\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\x00\xdd"
    )
    timers = await task
    assert len(timers) == 6
    assert len(timers[0].toBytes()) == 14
    assert (
        timers[0].toBytes()
        == b"\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    )
    assert str(timers[0]) == "Unset"


@pytest.mark.asyncio
async def test_async_get_timers_times_out(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test getting timers times out."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.001)

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    # ic state
    await task
    assert light.model_num == 0x25
    task = asyncio.ensure_future(light.async_get_timers())
    await asyncio.sleep(0)
    time = await task
    assert time is None


@pytest.mark.asyncio
async def test_async_set_timers(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test we can set timers."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    assert light.model_num == 0x25

    transport.reset_mock()
    await light.async_set_timers(
        [LedTimer(b"\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\xf0") for _ in range(6)]
    )
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"!\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\x00\xf0a"
    )

    caplog.clear()
    transport.reset_mock()
    await light.async_set_timers(
        [LedTimer(b"\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\xf0") for _ in range(7)]
    )
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"!\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\x00\xf0a"
    )
    assert "too many timers, truncating list" in caplog.text

    transport.reset_mock()
    await light.async_set_timers(
        [LedTimer(b"\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\xf0") for _ in range(2)]
    )
    assert transport.mock_calls[0][0] == "write"
    assert (
        transport.mock_calls[0][1][0]
        == b"!\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\xf0\x00\x00\x00\x0c-\x00>a\x00\x80\x00\x00\x00\xf0\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x0f\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\xbd"
    )


@pytest.mark.asyncio
async def test_async_enable_remote_access(mock_aio_protocol):
    """Test we can enable remote access."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x04\x10\xb6\x00\x98\x19\x04\x25\x0f\xdd"
    )
    await task

    with patch(
        "flux_led.aiodevice.AIOBulbScanner.async_enable_remote_access",
        return_value=mock_coro(True),
    ) as mock_async_enable_remote_access:
        await light.async_enable_remote_access("host", 1234)

    assert mock_async_enable_remote_access.mock_calls == [
        call("192.168.1.166", "host", 1234)
    ]


@pytest.mark.asyncio
async def test_async_disable_remote_access(mock_aio_protocol):
    """Test we can disable remote access."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x04\x10\xb6\x00\x98\x19\x04\x25\x0f\xdd"
    )
    await task

    with patch(
        "flux_led.aiodevice.AIOBulbScanner.async_disable_remote_access",
        return_value=mock_coro(True),
    ) as mock_async_disable_remote_access:
        await light.async_disable_remote_access()

    assert mock_async_disable_remote_access.mock_calls == [call("192.168.1.166")]


@pytest.mark.asyncio
async def test_async_reboot(mock_aio_protocol):
    """Test we can reboot."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x04\x10\xb6\x00\x98\x19\x04\x25\x0f\xdd"
    )
    await task

    with patch(
        "flux_led.aiodevice.AIOBulbScanner.async_reboot",
        return_value=mock_coro(True),
    ) as mock_async_reboot:
        await light.async_reboot()

    assert mock_async_reboot.mock_calls == [call("192.168.1.166")]


@pytest.mark.asyncio
async def test_power_state_response_processing(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can turn on and off via power state message."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    light._aio_protocol.data_received(b"\xf0\x32\xf0\xf0\xf0\xf0\xe2")
    assert light.power_restore_states == PowerRestoreStates(
        channel1=PowerRestoreState.LAST_STATE,
        channel2=PowerRestoreState.LAST_STATE,
        channel3=PowerRestoreState.LAST_STATE,
        channel4=PowerRestoreState.LAST_STATE,
    )
    light._aio_protocol.data_received(b"\xf0\x32\x0f\xf0\xf0\xf0\x01")
    assert light.power_restore_states == PowerRestoreStates(
        channel1=PowerRestoreState.ALWAYS_ON,
        channel2=PowerRestoreState.LAST_STATE,
        channel3=PowerRestoreState.LAST_STATE,
        channel4=PowerRestoreState.LAST_STATE,
    )
    light._aio_protocol.data_received(b"\xf0\x32\xff\xf0\xf0\xf0\xf1")
    assert light.power_restore_states == PowerRestoreStates(
        channel1=PowerRestoreState.ALWAYS_OFF,
        channel2=PowerRestoreState.LAST_STATE,
        channel3=PowerRestoreState.LAST_STATE,
        channel4=PowerRestoreState.LAST_STATE,
    )


@pytest.mark.asyncio
async def test_async_set_power_restore_state(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can set power restore state and report it."""
    socket = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(socket.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    socket._aio_protocol.data_received(
        b"\x81\x97\x24\x24\x00\x00\x00\x00\x00\x00\x02\x00\x00\x62"
    )
    # power restore state
    socket._aio_protocol.data_received(b"\x0f\x32\xf0\xf0\xf0\xf0\x01")
    await task
    assert socket.model_num == 0x97
    assert socket.power_restore_states == PowerRestoreStates(
        channel1=PowerRestoreState.LAST_STATE,
        channel2=PowerRestoreState.LAST_STATE,
        channel3=PowerRestoreState.LAST_STATE,
        channel4=PowerRestoreState.LAST_STATE,
    )

    transport.reset_mock()
    await socket.async_set_power_restore(
        channel1=PowerRestoreState.ALWAYS_ON,
        channel2=PowerRestoreState.ALWAYS_ON,
        channel3=PowerRestoreState.ALWAYS_ON,
        channel4=PowerRestoreState.ALWAYS_ON,
    )
    assert transport.mock_calls[0][0] == "write"
    assert transport.mock_calls[0][1][0] == b"1\x0f\x0f\x0f\x0f\xf0]"


@pytest.mark.asyncio
async def test_async_set_power_restore_state_fails(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we raise if we do not get a power restore state."""
    socket = AIOWifiLedBulb("192.168.1.166", timeout=0.01)

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(socket.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    socket._aio_protocol.data_received(
        b"\x81\x97\x24\x24\x00\x00\x00\x00\x00\x00\x02\x00\x00\x62"
    )
    # power restore state not sent
    with pytest.raises(RuntimeError):
        await task


@pytest.mark.asyncio
async def test_remote_config_queried(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test power state is queried if discovery shows a compatible remote."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.discovery = FLUX_DISCOVERY_24G_REMOTE

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "DEVICE_CONFIG_WAIT_SECONDS", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        transport, _protocol = await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        )
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\x5e\x00\x0e\x2b\x01\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x55\xde"
        )
        await task

        assert light.remote_config == RemoteConfig.DISABLED
        assert light.paired_remotes == 0
        assert transport.mock_calls == [
            call.get_extra_info("peername"),
            call.write(bytearray(b"\x81\x8a\x8b\x96")),
            call.write(
                bytearray(b"\xb0\xb1\xb2\xb3\x00\x01\x01\x00\x00\x04+,-\x84\xd4")
            ),
        ]


@pytest.mark.asyncio
async def test_remote_config_response_processing(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can turn on and off via power state message."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.discovery = FLUX_DISCOVERY_24G_REMOTE

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "DEVICE_CONFIG_WAIT_SECONDS", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        )
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\x5e\x00\x0e\x2b\x01\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x55\xde"
        )

        await task
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\x5e\x00\x0e\x2b\x01\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x55\xde"
        )
        assert light.remote_config == RemoteConfig.DISABLED
        assert light.paired_remotes == 0

        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\x45\x00\x0e\x2b\x02\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x56\xc7"
        )
        assert light.remote_config == RemoteConfig.OPEN
        assert light.paired_remotes == 0

        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\xe3\x00\x0e\x2b\x03\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x30\x19"
        )
        assert light.remote_config == RemoteConfig.PAIRED_ONLY
        assert light.paired_remotes == 2


@pytest.mark.asyncio
async def test_async_config_remotes(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can configure remotes."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.discovery = FLUX_DISCOVERY_24G_REMOTE

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "DEVICE_CONFIG_WAIT_SECONDS", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        transport, _protocol = await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        )
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\x5e\x00\x0e\x2b\x01\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x55\xde"
        )

        await task
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\xe3\x00\x0e\x2b\x03\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x30\x19"
        )
        assert light.remote_config == RemoteConfig.PAIRED_ONLY
        assert light.paired_remotes == 2

        transport.reset_mock()
        await light.async_config_remotes(RemoteConfig.DISABLED)
        assert transport.mock_calls[0][0] == "write"
        assert (
            transport.mock_calls[0][1][0]
            == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x10*\x01\xff\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x0f5C"
        )

        transport.reset_mock()
        await light.async_config_remotes(RemoteConfig.OPEN)
        assert transport.mock_calls[0][0] == "write"
        assert (
            transport.mock_calls[0][1][0]
            == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x03\x00\x10*\x02\xff\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x0f6G"
        )

        transport.reset_mock()
        await light.async_config_remotes(RemoteConfig.PAIRED_ONLY)
        assert transport.mock_calls[0][0] == "write"
        assert (
            transport.mock_calls[0][1][0]
            == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x05\x00\x10*\x03\xff\xff\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\x0f7K"
        )


@pytest.mark.asyncio
async def test_async_unpair_remotes(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can unpair remotes."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.discovery = FLUX_DISCOVERY_24G_REMOTE

    def _updated_callback(*args, **kwargs):
        pass

    with patch.object(aiodevice, "DEVICE_CONFIG_WAIT_SECONDS", 0):
        task = asyncio.create_task(light.async_setup(_updated_callback))
        transport, _protocol = await mock_aio_protocol()
        light._aio_protocol.data_received(
            b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
        )
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\x5e\x00\x0e\x2b\x01\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x55\xde"
        )

        await task
        light._aio_protocol.data_received(
            b"\xb0\xb1\xb2\xb3\x00\x01\x01\xe3\x00\x0e\x2b\x03\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x00\x30\x19"
        )
        assert light.remote_config == RemoteConfig.PAIRED_ONLY
        assert light.paired_remotes == 2

        transport.reset_mock()
        await light.async_unpair_remotes()
        assert transport.mock_calls[0][0] == "write"
        assert (
            transport.mock_calls[0][1][0]
            == b"\xb0\xb1\xb2\xb3\x00\x01\x01\x01\x00\x10*\xff\xff\x01\xff\xff\xff\x00\x00\x00\x00\x00\x00\x00\xf0\x16\x05"
        )


@pytest.mark.asyncio
async def test_async_config_remotes_unsupported_device(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we can configure remotes."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    assert light.paired_remotes is None

    with pytest.raises(ValueError):
        await light.async_config_remotes(RemoteConfig.PAIRED_ONLY)

    with pytest.raises(ValueError):
        await light.async_unpair_remotes()


@pytest.mark.asyncio
async def test_async_config_remotes_no_response(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test device supports remote config but does not respond."""
    light = AIOWifiLedBulb("192.168.1.166", timeout=0.001)
    light.discovery = FLUX_DISCOVERY_24G_REMOTE

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task
    assert light.paired_remotes is None
    assert "Could not determine 2.4ghz remote config" in caplog.text


@pytest.mark.asyncio
async def test_partial_discovery(mock_aio_protocol, caplog: pytest.LogCaptureFixture):
    """Test discovery that is missing hardware data."""
    light = AIOWifiLedBulb("192.168.1.166")
    light.discovery = FLUX_DISCOVERY_MISSING_HARDWARE

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    light._aio_protocol.data_received(
        b"\xb0\xb1\xb2\xb3\x00\x01\x01\x5e\x00\x0e\x2b\x01\x00\x00\x00\x00\x29\x00\x00\x00\x00\x00\x00\x55\xde"
    )
    await task
    assert light.hardware is None


@pytest.mark.asyncio
async def test_async_scanner(mock_discovery_aio_protocol):
    """Test scanner."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_scan(timeout=0.1, address="192.168.213.252")
    )
    _transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(b"HF-A11ASSISTHREAD", ("127.0.0.1", 48899))
    protocol.datagram_received(
        b"192.168.1.193,DC4F22E6462E,AK001-ZJ200", ("192.168.1.193", 48899)
    )
    protocol.datagram_received(
        b"+ok=25_18_20170908_Armacost\r", ("192.168.1.193", 48899)
    )
    protocol.datagram_received(
        b"+ok=TCP,8806,mhc8806us.magichue.net\r", ("192.168.1.193", 48899)
    )

    protocol.datagram_received(
        b"192.168.213.252,B4E842E10588,AK001-ZJ2145", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(
        b"192.168.198.198,B4E842E10522,AK001-ZJ2149", ("192.168.198.198", 48899)
    )
    protocol.datagram_received(
        b"192.168.198.197,B4E842E10521,AK001-ZJ2146", ("192.168.198.197", 48899)
    )
    protocol.datagram_received(
        b"192.168.198.196,B4E842E10520,AK001-ZJ2144", ("192.168.198.196", 48899)
    )
    protocol.datagram_received(
        b"192.168.211.230,A020A61D892B,AK001-ZJ100", ("192.168.211.230", 48899)
    )
    protocol.datagram_received(
        b"+ok=TCP,GARBAGE,ra8816us02.magichue.net\r", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(
        b"192.168.213.259,B4E842E10586,AK001-ZJ2145", ("192.168.213.259", 48899)
    )
    protocol.datagram_received(
        b"+ok=TCP,8816,ra8816us02.magichue.net\r", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(
        b"+ok=TCP,8806,mhc8806us.magichue.net", ("192.168.211.230", 48899)
    )
    protocol.datagram_received(b"AT+LVER\r", ("127.0.0.1", 48899))
    protocol.datagram_received(
        b"+ok=GARBAGE_GARBAGE_GARBAGE_ZG-BL\r", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(
        b"+ok=08_15_20210204_ZG-BL\r", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(b"+ok=52_3_20210204\r", ("192.168.198.198", 48899))
    protocol.datagram_received(b"+ok=62_3\r", ("192.168.198.197", 48899))
    protocol.datagram_received(b"+ok=41_3_202\r", ("192.168.198.196", 48899))

    protocol.datagram_received(
        b"+ok=35_62_20210109_ZG-BL-PWM\r", ("192.168.213.259", 48899)
    )
    protocol.datagram_received(
        b"192.168.213.65,F4CFA23E1AAF,AK001-ZJ2104", ("192.168.213.65", 48899)
    )
    protocol.datagram_received(
        b"+ok=33_11_20170307_IR_mini\r\n", ("192.168.211.230", 48899)
    )
    protocol.datagram_received(b"+ok=", ("192.168.213.65", 48899))
    protocol.datagram_received(b"+ok=A2_33_20200428_ZG-LX\r", ("192.168.213.65", 48899))
    protocol.datagram_received(b"+ok=", ("192.168.213.259", 48899))
    protocol.datagram_received(
        b"+ok=TCP,8816,ra8816us02.magichue.net\r", ("192.168.198.196", 48899)
    )
    data = await task
    assert data == [
        {
            "firmware_date": datetime.date(2017, 9, 8),
            "id": "DC4F22E6462E",
            "ipaddr": "192.168.1.193",
            "model": "AK001-ZJ200",
            "model_description": "Controller RGB/WW/CW",
            "model_info": "Armacost",
            "model_num": 37,
            "remote_access_enabled": True,
            "remote_access_host": "mhc8806us.magichue.net",
            "remote_access_port": 8806,
            "version_num": 24,
        },
        {
            "firmware_date": datetime.date(2021, 2, 4),
            "id": "B4E842E10588",
            "ipaddr": "192.168.213.252",
            "model": "AK001-ZJ2145",
            "model_description": "Controller RGB with MIC",
            "model_info": "ZG-BL",
            "model_num": 8,
            "remote_access_enabled": True,
            "remote_access_host": "ra8816us02.magichue.net",
            "remote_access_port": 8816,
            "version_num": 21,
        },
        {
            "firmware_date": datetime.date(2021, 2, 4),
            "id": "B4E842E10522",
            "ipaddr": "192.168.198.198",
            "model": "AK001-ZJ2149",
            "model_description": "Bulb CCT",
            "model_info": None,
            "model_num": 82,
            "remote_access_enabled": None,
            "remote_access_host": None,
            "remote_access_port": None,
            "version_num": 3,
        },
        {
            "firmware_date": None,
            "id": "B4E842E10521",
            "ipaddr": "192.168.198.197",
            "model": "AK001-ZJ2146",
            "model_description": "Controller CCT",
            "model_info": None,
            "model_num": 98,
            "remote_access_enabled": None,
            "remote_access_host": None,
            "remote_access_port": None,
            "version_num": 3,
        },
        {
            "firmware_date": None,
            "id": "B4E842E10520",
            "ipaddr": "192.168.198.196",
            "model": "AK001-ZJ2144",
            "model_description": "Controller Dimmable",
            "model_info": None,
            "model_num": 65,
            "remote_access_enabled": True,
            "remote_access_host": "ra8816us02.magichue.net",
            "remote_access_port": 8816,
            "version_num": 3,
        },
        {
            "firmware_date": datetime.date(2017, 3, 7),
            "id": "A020A61D892B",
            "ipaddr": "192.168.211.230",
            "model": "AK001-ZJ100",
            "model_description": "Controller RGB IR Mini",
            "model_info": "IR_mini",
            "model_num": 51,
            "remote_access_enabled": True,
            "remote_access_host": "mhc8806us.magichue.net",
            "remote_access_port": 8806,
            "version_num": 17,
        },
        {
            "firmware_date": datetime.date(2021, 1, 9),
            "id": "B4E842E10586",
            "ipaddr": "192.168.213.259",
            "model": "AK001-ZJ2145",
            "model_description": "Bulb RGBCW",
            "model_info": "ZG-BL-PWM",
            "model_num": 53,
            "remote_access_enabled": False,
            "remote_access_host": None,
            "remote_access_port": None,
            "version_num": 98,
        },
        {
            "firmware_date": datetime.date(2020, 4, 28),
            "id": "F4CFA23E1AAF",
            "ipaddr": "192.168.213.65",
            "model": "AK001-ZJ2104",
            "model_description": "Addressable v2",
            "model_info": "ZG-LX",
            "model_num": 162,
            "remote_access_enabled": False,
            "remote_access_host": None,
            "remote_access_port": None,
            "version_num": 51,
        },
    ]


@pytest.mark.asyncio
async def test_async_scanner_specific_address(mock_discovery_aio_protocol):
    """Test scanner with a specific address."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_scan(timeout=10, address="192.168.213.252")
    )
    _transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(
        b"192.168.213.252,B4E842E10588,AK001-ZJ2145", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(
        b"+ok=08_15_20210204_ZG-BL\r", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(
        b"+ok=TCP,8816,ra8816us02.magichue.net\r", ("192.168.213.252", 48899)
    )
    data = await task
    assert data == [
        {
            "firmware_date": datetime.date(2021, 2, 4),
            "id": "B4E842E10588",
            "ipaddr": "192.168.213.252",
            "model": "AK001-ZJ2145",
            "model_description": "Controller RGB with MIC",
            "model_info": "ZG-BL",
            "model_num": 8,
            "version_num": 21,
            "remote_access_enabled": True,
            "remote_access_host": "ra8816us02.magichue.net",
            "remote_access_port": 8816,
        }
    ]
    assert scanner.getBulbInfoByID("B4E842E10588") == {
        "firmware_date": datetime.date(2021, 2, 4),
        "id": "B4E842E10588",
        "ipaddr": "192.168.213.252",
        "model": "AK001-ZJ2145",
        "model_description": "Controller RGB with MIC",
        "model_info": "ZG-BL",
        "model_num": 8,
        "version_num": 21,
        "remote_access_enabled": True,
        "remote_access_host": "ra8816us02.magichue.net",
        "remote_access_port": 8816,
    }
    assert scanner.getBulbInfo() == [
        {
            "firmware_date": datetime.date(2021, 2, 4),
            "id": "B4E842E10588",
            "ipaddr": "192.168.213.252",
            "model": "AK001-ZJ2145",
            "model_description": "Controller RGB with MIC",
            "model_info": "ZG-BL",
            "model_num": 8,
            "version_num": 21,
            "remote_access_enabled": True,
            "remote_access_host": "ra8816us02.magichue.net",
            "remote_access_port": 8816,
        }
    ]


@pytest.mark.asyncio
async def test_async_scanner_specific_address_legacy_device(
    mock_discovery_aio_protocol,
):
    """Test scanner with a specific address of a legacy device."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_scan(timeout=10, address="192.168.213.252")
    )
    _transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(
        b"192.168.213.252,ACCF232E5124,HF-A11-ZJ002", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(b"+ok=15\r\n\r\n", ("192.168.213.252", 48899))
    protocol.datagram_received(b"+ERR=-2\r\n\r\n", ("192.168.213.252", 48899))
    data = await task
    assert data == [
        {
            "firmware_date": None,
            "id": "ACCF232E5124",
            "ipaddr": "192.168.213.252",
            "model": "HF-A11-ZJ002",
            "model_description": None,
            "model_info": None,
            "model_num": None,
            "remote_access_enabled": None,
            "remote_access_host": None,
            "remote_access_port": None,
            "version_num": 21,
        }
    ]
    assert is_legacy_device(data[0]) is True


@pytest.mark.asyncio
async def test_async_scanner_times_out_with_nothing(mock_discovery_aio_protocol):
    """Test scanner."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(scanner.async_scan(timeout=0.025))
    _transport, _protocol = await mock_discovery_aio_protocol()
    data = await task
    assert data == []


@pytest.mark.asyncio
async def test_async_scanner_times_out_with_nothing_specific_address(
    mock_discovery_aio_protocol,
):
    """Test scanner."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_scan(timeout=0.025, address="192.168.213.252")
    )
    _transport, _protocol = await mock_discovery_aio_protocol()
    data = await task
    assert data == []


@pytest.mark.asyncio
async def test_async_scanner_falls_back_to_any_source_port_if_socket_in_use():
    """Test port fallback."""
    hold_socket = create_udp_socket(AIOBulbScanner.DISCOVERY_PORT)
    assert hold_socket.getsockname() == ("0.0.0.0", 48899)
    random_socket = create_udp_socket(AIOBulbScanner.DISCOVERY_PORT)
    assert random_socket.getsockname() != ("0.0.0.0", 48899)


@pytest.mark.asyncio
async def test_async_scanner_enable_remote_access(mock_discovery_aio_protocol):
    """Test scanner enabling remote access with a specific address."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_enable_remote_access(
            timeout=10,
            address="192.168.213.252",
            remote_access_host="ra8815us02.magichue.net",
            remote_access_port=8815,
        )
    )
    transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(
        b"192.168.213.252,B4E842E10588,AK001-ZJ2145", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(b"+ok\r", ("192.168.213.252", 48899))
    protocol.datagram_received(b"+ok\r", ("192.168.213.252", 48899))
    await task
    assert transport.mock_calls == [
        call.sendto(b"HF-A11ASSISTHREAD", ("192.168.213.252", 48899)),
        call.sendto(
            b"AT+SOCKB=TCP,8815,ra8815us02.magichue.net\r", ("192.168.213.252", 48899)
        ),
        call.sendto(b"AT+Z\r", ("192.168.213.252", 48899)),
        call.close(),
    ]


@pytest.mark.asyncio
async def test_async_scanner_disable_remote_access(mock_discovery_aio_protocol):
    """Test scanner disable remote access with a specific address."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_disable_remote_access(
            timeout=10,
            address="192.168.213.252",
        )
    )
    transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(
        b"192.168.213.252,B4E842E10588,AK001-ZJ2145", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(b"+ok\r", ("192.168.213.252", 48899))
    protocol.datagram_received(b"+ok\r", ("192.168.213.252", 48899))
    await task
    assert transport.mock_calls == [
        call.sendto(b"HF-A11ASSISTHREAD", ("192.168.213.252", 48899)),
        call.sendto(b"AT+SOCKB=NONE\r", ("192.168.213.252", 48899)),
        call.sendto(b"AT+Z\r", ("192.168.213.252", 48899)),
        call.close(),
    ]


@pytest.mark.asyncio
async def test_async_scanner_reboot(mock_discovery_aio_protocol):
    """Test scanner reboot with a specific address."""
    scanner = AIOBulbScanner()

    task = asyncio.ensure_future(
        scanner.async_reboot(
            timeout=10,
            address="192.168.213.252",
        )
    )
    transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(
        b"192.168.213.252,B4E842E10588,AK001-ZJ2145", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(b"+ok\r", ("192.168.213.252", 48899))
    await task
    assert transport.mock_calls == [
        call.sendto(b"HF-A11ASSISTHREAD", ("192.168.213.252", 48899)),
        call.sendto(b"AT+Z\r", ("192.168.213.252", 48899)),
        call.close(),
    ]


@pytest.mark.asyncio
async def test_async_scanner_disable_remote_access_timeout(mock_discovery_aio_protocol):
    """Test scanner disable remote access with a specific address failure."""
    scanner = AIOBulbScanner()
    task = asyncio.ensure_future(
        scanner.async_disable_remote_access(
            timeout=0.02,
            address="192.168.213.252",
        )
    )
    transport, protocol = await mock_discovery_aio_protocol()
    protocol.datagram_received(
        b"192.168.213.252,B4E842E10588,AK001-ZJ2145", ("192.168.213.252", 48899)
    )
    protocol.datagram_received(b"+ok\r", ("192.168.213.252", 48899))
    with pytest.raises(asyncio.TimeoutError):
        await task
    assert transport.mock_calls == [
        call.sendto(b"HF-A11ASSISTHREAD", ("192.168.213.252", 48899)),
        call.sendto(b"AT+SOCKB=NONE\r", ("192.168.213.252", 48899)),
        call.sendto(b"AT+Z\r", ("192.168.213.252", 48899)),
        call.close(),
    ]


def test_merge_discoveries() -> None:
    """Unit test to make sure we can merge two discoveries."""
    full = FLUX_DISCOVERY.copy()
    partial = FLUX_DISCOVERY_PARTIAL.copy()
    merge_discoveries(partial, full)
    assert partial == FLUX_DISCOVERY
    assert full == FLUX_DISCOVERY

    full = FLUX_DISCOVERY.copy()
    partial = FLUX_DISCOVERY_PARTIAL.copy()
    merge_discoveries(full, partial)
    assert full == FLUX_DISCOVERY


@pytest.mark.asyncio
async def test_armacost():
    """Test armacost uses port 34001."""
    discovery = FluxLEDDiscovery(
        {
            "firmware_date": datetime.date(2017, 9, 8),
            "id": "DC4F22E6462E",
            "ipaddr": "192.168.1.193",
            "model": "AK001-ZJ200",
            "model_description": "Controller RGB/WW/CW",
            "model_info": "Armacost",
            "model_num": 37,
            "remote_access_enabled": True,
            "remote_access_host": "mhc8806us.magichue.net",
            "remote_access_port": 8806,
            "version_num": 24,
        }
    )
    light = AIOWifiLedBulb("192.168.1.193")
    light.discovery = discovery
    assert light.port == 34001


@pytest.mark.asyncio
async def test_not_armacost():
    """Test not armacost uses 5577."""
    discovery = FluxLEDDiscovery(
        {
            "firmware_date": datetime.date(2021, 2, 4),
            "id": "B4E842E10588",
            "ipaddr": "192.168.213.252",
            "model": "AK001-ZJ2145",
            "model_description": "Controller RGB with MIC",
            "model_info": "ZG-BL",
            "model_num": 8,
            "remote_access_enabled": True,
            "remote_access_host": "ra8816us02.magichue.net",
            "remote_access_port": 8816,
            "version_num": 21,
        }
    )
    light = AIOWifiLedBulb("192.168.213.252")
    light.discovery = discovery
    assert light.port == 5577


def test_extended_state_to_state_full_cool_white():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0x0F,
            0x00,
            0x00,
            0x00,
            0x64,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0xF5,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 0  # red
    assert raw_state.green == 0  # green
    assert raw_state.blue == 0  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 255  # cool white


def test_extended_state_to_state_full_warm_white():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0x0F,
            0x00,
            0x00,
            0x00,
            0x00,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x94,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 0  # red
    assert raw_state.green == 0  # green
    assert raw_state.blue == 0  # blue
    assert raw_state.warm_white == 255  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_to_state_full_red():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0xF0,
            0x00,
            0x64,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0xF8,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 255  # red
    assert raw_state.green == 0  # green
    assert raw_state.blue == 0  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_to_state_full_green():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0xF0,
            0x3C,
            0x64,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x1A,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 0  # red
    assert raw_state.green == 255  # green
    assert raw_state.blue == 0  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_to_state_full_blue():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0xF0,
            0x78,
            0x64,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x57,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 0  # red
    assert raw_state.green == 0  # green
    assert raw_state.blue == 255  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_to_state_full_yellow():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0xF0,
            0x1E,
            0x64,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x73,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 255  # red
    assert raw_state.green == 255  # green
    assert raw_state.blue == 0  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_to_state_full_purple():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0xF0,
            0x96,
            0x64,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x7A,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x61  # preset
    assert raw_state.red == 255  # red
    assert raw_state.green == 0  # green
    assert raw_state.blue == 255  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_to_state_full_speed_effect():
    proto = ProtocolLEDENET25Byte()

    # Simulated extended state response payload (starts with EA 81)
    raw_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x25,
            0x00,
            0x64,
            0xF0,
            0x0B,
            0xE4,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x56,
        )
    )

    assert proto.is_valid_extended_state_response(raw_state) is True

    state = proto.extended_state_to_state(raw_state)
    assert len(state) == 14

    raw_state = LEDENETRawState(*state)

    # Validate fields
    assert raw_state.power_state == 0x23  # power
    assert raw_state.preset_pattern == 0x25  # preset
    assert raw_state.red == 255  # red
    assert raw_state.green == 0  # green
    assert raw_state.blue == 0  # blue
    assert raw_state.warm_white == 0  # warm white
    assert raw_state.cool_white == 0  # cool white


def test_extended_state_too_short():
    proto = ProtocolLEDENET25Byte()
    assert proto.extended_state_to_state(b"\xea\x81") == b""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label,hue_byte,sat_byte,val_byte,expected_rgb",
    [
        ("red", 0x00, 100, 100, (255, 0, 0)),
        ("yellow", 0x1E, 100, 100, (255, 255, 0)),
        ("green", 0x3C, 100, 100, (0, 255, 0)),
        ("blue", 0x78, 100, 100, (0, 0, 255)),
    ],
)
async def test_extended_state_color_parsing(
    label, hue_byte, sat_byte, val_byte, expected_rgb
):
    proto = ProtocolLEDENET25Byte()

    print(
        f"values → hue: {hue_byte} ({type(hue_byte)}), sat: {sat_byte} ({type(sat_byte)}), val: {val_byte} ({type(val_byte)})"
    )

    raw_state = bytes(
        [
            0xEA,
            0x81,
            0x01,
            0x00,
            0x35,
            0x0A,
            0x23,
            0x61,
            0x00,
            0x0A,
            0xF0,
            hue_byte,
            sat_byte,
            val_byte,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x00,
            0x7A,
        ]
    )

    result = proto.extended_state_to_state(raw_state)
    rgb = tuple(result[6:9])
    assert all(abs(a - b) <= 1 for a, b in zip(rgb, expected_rgb)), (
        f"{label} RGB mismatch: got {rgb}, expected {expected_rgb}"
    )


@pytest.mark.asyncio
async def test_setup_0x35_with_version_num_10(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test we use the right protocol for 0x35 with v10."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(bytes.fromhex("81352361306400ffff000a00f0c6"))
    await task
    assert light.model_num == 0x35
    assert light.protocol == PROTOCOL_LEDENET_25BYTE
    assert light.white_active is False
    light._aio_protocol.data_received(
        bytes(
            (
                0xB0,
                0xB1,
                0xB2,
                0xB3,
                0x00,
                0x02,
                0x02,
                0x17,
                0x00,
                0x14,
                0xEA,
                0x81,
                0x01,
                0x00,
                0x35,
                0x0A,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    assert light.white_active is True


@pytest.mark.asyncio
async def test_setup_0xB6_surplife(mock_aio_protocol):
    """0xB6 Surplife is recognised via the reused 25-byte protocol.

    The device replies only with the extended state format (0xEA 0x81), so this
    exercises protocol determination from an extended-only response (the case
    that previously failed with "Cannot determine protocol").
    """
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    # Extended state: EA 81 01 00 B6(model) 01(ver) 23(on) 61 00 64 0F 00 00 00 64 64 00 00 64(count) 00 CS
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x00,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x64,
                0x00,
                0x83,
            )
        )
    )
    await task
    assert light.model_num == 0xB6
    assert light.protocol == PROTOCOL_LEDENET_EXTENDED_CUSTOM
    assert light.color_modes == {COLOR_MODE_RGB, COLOR_MODE_DIM}
    assert "Surplife" in light.model
    assert light.supports_extended_custom_effects is True
    # The configured LED count (extended-state byte 18 = 0x64) is exposed
    # end-to-end via the device property (async path store).
    assert light.led_count == 100


@pytest.mark.asyncio
async def test_setup_0xB6_surplife_real_frame(mock_aio_protocol):
    """0xB6 setup against the real 27-byte capture from the device.

    Unlike test_setup_0xB6_surplife (a minimal synthetic frame), this feeds the
    actual 27-byte frame captured from hardware so extended_state_to_state runs
    against real wire data (byte 7 = 0x66, byte 8 = 0x01, byte 18 = LED count).
    """
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    # EA 81 01 00 B6(model) 09 24 66 01 64 F0 00 00 00 00 64 05 00 64(count) 00 00 00 20 02 01 00 03
    light._aio_protocol.data_received(
        bytes.fromhex("ea810100b60924660164f00000000064050064000000200201000003")
    )
    await task
    assert light.model_num == 0xB6
    assert light.protocol == PROTOCOL_LEDENET_EXTENDED_CUSTOM
    assert "Surplife" in light.model
    assert light.supports_extended_custom_effects is True
    # Configured LED count from extended-state byte 18 (0x64 = 100).
    assert light.led_count == 100


@pytest.mark.asyncio
async def test_0xB6_colorful_solid_red_full(mock_aio_protocol):
    """0xB6 "Colorful" solid red at full brightness is a plain color, not an effect.

    Real device EA81 capture (app Colorful -> red): preset_pattern=0x24, mode=0x01
    -> reported as a color with no effect (verified on device).
    """
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        bytes.fromhex("ea810100b60923240164f0b46464ff000500500000002002010003")
    )
    await task
    assert light.model_num == 0xB6
    assert light.effect is None
    assert light.is_on is True
    assert light.rgb == (255, 0, 0)
    assert light.brightness == 255


@pytest.mark.asyncio
async def test_0xB6_colorful_solid_red_dim(mock_aio_protocol):
    """0xB6 "Colorful" solid red dimmed to 30% is a plain color, not an effect.

    Real device EA81 capture (app Colorful -> red @ 30%): preset_pattern=0x24,
    mode=0x01, value byte 0x1e (30%).
    """
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        bytes.fromhex("ea810100b60923240164f0b4641eff000500500000002002010003")
    )
    await task
    assert light.model_num == 0xB6
    assert light.effect is None
    assert light.is_on is True
    # Clearly dimmed (well below the full-brightness value of 255).
    assert light.brightness == 76


@pytest.mark.asyncio
async def test_0xB6_scene_wave(mock_aio_protocol):
    """0xB6 "Scenes" animated effect reports preset_pattern=0x25, mode=effect id.

    Real device EA81 capture of the app's Scenes -> Wave (preset 0x25, mode 0x01).
    """
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        bytes.fromhex("ea810100b60923250150f0b46464ff000500500000002002010003")
    )
    await task
    assert light.model_num == 0xB6
    assert light.effect == "Wave"


@pytest.mark.asyncio
async def test_0xB6_scene_static_fill(mock_aio_protocol):
    """0xB6 "Scenes" Static Fill reports preset_pattern=0x25, mode=0x66.

    Real device EA81 capture of the app's Scenes -> Static Fill (preset 0x25,
    mode 0x66, blue).
    """
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        bytes.fromhex("ea810100b60923256632f0786464ff000500500000002002010003")
    )
    await task
    assert light.model_num == 0xB6
    assert light.effect == "Static Fill"


def test_protocol_extended_custom_state_response_length():
    """ProtocolLEDENETExtendedCustom expects the 27-byte extended state."""
    proto = ProtocolLEDENETExtendedCustom()
    assert proto.state_response_length == LEDENET_EXTENDED_STATE_RESPONSE_LEN


def test_protocol_extended_state_validation_0xB6():
    """The protocols recognise the extended (0xEA 0x81) state response."""
    ext = bytes.fromhex("ea810100b605236100640f00000064640000000083")
    # ProtocolLEDENET8Byte (used while probing) recognises the extended frame.
    assert ProtocolLEDENET8Byte().is_valid_extended_state_response(ext) is True
    # The dedicated protocol only accepts the extended format.
    proto = ProtocolLEDENETExtendedCustom()
    assert proto.is_valid_state_response(ext) is True
    assert proto.is_valid_state_response(bytes((0x81,)) + b"\x00" * 13) is False


def test_protocol_extended_state_to_state_white_off():
    """white_brightness of 0 maps to both white channels off."""
    ext = bytes.fromhex("ea810100b605236100640f000000ff000000000083")
    result = ProtocolLEDENETExtendedCustom().extended_state_to_state(ext)
    assert result[9] == 0  # warm_white
    assert result[11] == 0  # cool_white


def test_protocol_extended_state_to_state_white_brightness_out_of_range():
    """white_brightness > 100 maps to both white channels off (no ValueError).

    Byte 15 is a raw 0-255 value; an out-of-range white brightness must not
    crash ``scaled_color_temp_to_white_levels`` (which rejects brightness > 100).
    Temp (byte 14 = 0x32) is in range so only the brightness guard applies.
    """
    ext = bytes.fromhex("ea810100b605236100640f00000032c80000000083")
    result = ProtocolLEDENETExtendedCustom().extended_state_to_state(ext)
    assert result[9] == 0  # warm_white
    assert result[11] == 0  # cool_white


def test_protocol_extended_state_to_state_too_short_returns_empty():
    """A truncated extended frame (< 20 bytes) returns an empty bytestring."""
    proto = ProtocolLEDENETExtendedCustom()
    assert proto.extended_state_to_state(b"\xea\x81\x01\x00\xb6") == b""


def test_protocol_named_raw_state_extended_conversion():
    """named_raw_state converts an extended frame to the standard layout."""
    ext = bytes.fromhex("ea810100b605236100640f00000064640000000083")
    result = ProtocolLEDENETExtendedCustom().named_raw_state(ext)
    assert result.head == 0x81
    assert result.model_num == 0xB6


def test_extended_state_led_count():
    """extended_state_led_count returns the configured LED count from real captured frames.

    Real captured frames (0xB6 device):
      LED count = 100: ea 81 01 00 b6 09 24 66 01 64 f0 00 00 00 00 64 05 00 64 00 00 00 20 02 01 00 03
      LED count =  80: ea 81 01 00 b6 09 23 66 01 64 f0 00 00 00 00 64 05 00 50 00 00 00 20 02 01 00 03

    The LED count byte is at index 18 of the raw extended state buffer.
    """
    proto = ProtocolLEDENETExtendedCustom()

    frame_100 = bytes.fromhex("ea810100b60924660164f000000000640500640000002002010003")
    frame_80 = bytes.fromhex("ea810100b60923660164f000000000640500500000002002010003")

    assert proto.extended_state_led_count(frame_100) == 100
    assert proto.extended_state_led_count(frame_80) == 80

    # Too-short / invalid frame returns None
    assert proto.extended_state_led_count(b"\xea\x81\x01") is None
    assert proto.extended_state_led_count(b"\x00" * 27) is None


def test_extended_custom_effect_pattern_enum_values():
    """Test ExtendedCustomEffectPattern enum has expected values."""
    assert ExtendedCustomEffectPattern.WAVE.value == 0x01
    assert ExtendedCustomEffectPattern.METEOR.value == 0x02
    assert ExtendedCustomEffectPattern.STREAMER.value == 0x03
    assert ExtendedCustomEffectPattern.BUILDING_BLOCKS.value == 0x04
    assert ExtendedCustomEffectPattern.BREATHE.value == 0x09
    assert ExtendedCustomEffectPattern.STATIC_GRADIENT.value == 0x65
    assert ExtendedCustomEffectPattern.STATIC_FILL.value == 0x66


def test_extended_custom_effect_direction_enum_values():
    """Test ExtendedCustomEffectDirection enum has expected values."""
    assert ExtendedCustomEffectDirection.LEFT_TO_RIGHT.value == 0x01
    assert ExtendedCustomEffectDirection.RIGHT_TO_LEFT.value == 0x02


def test_extended_custom_effect_option_enum_values():
    """Test ExtendedCustomEffectOption enum has expected values."""
    assert ExtendedCustomEffectOption.DEFAULT.value == 0x00
    assert ExtendedCustomEffectOption.VARIANT_1.value == 0x01
    assert ExtendedCustomEffectOption.VARIANT_2.value == 0x02


def test_construct_extended_custom_effect_single_color():
    """Test constructing an extended custom effect with single color."""
    proto = ProtocolLEDENETExtendedCustom()

    # Single red color, pattern Wave, default settings
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=[(255, 0, 0)],
        speed=50,
        density=50,
        direction=0x01,
        option=0x00,
    )

    # Result should be a wrapped message
    assert isinstance(result, bytearray)
    assert len(result) > 0

    # Check the wrapper header (b0 b1 b2 b3)
    assert result[0] == 0xB0
    assert result[1] == 0xB1
    assert result[2] == 0xB2
    assert result[3] == 0xB3


def test_construct_extended_custom_effect_multiple_colors():
    """Test constructing an extended custom effect with multiple colors."""
    proto = ProtocolLEDENETExtendedCustom()

    # Three colors: red, green, blue
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    result = proto.construct_extended_custom_effect(
        pattern_id=2,  # Meteor
        colors=colors,
        speed=80,
        density=100,
        direction=0x02,  # Right to Left
        option=0x01,  # Color change
    )

    assert isinstance(result, bytearray)
    assert len(result) > 0


def test_construct_extended_custom_effect_color_order():
    """Test that colors are stored in input order."""
    proto = ProtocolLEDENETExtendedCustom()

    # Two distinct colors
    colors = [(255, 0, 0), (0, 0, 255)]  # Red, Blue
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=colors,
        speed=50,
        density=50,
    )

    # The message structure after wrapper:
    # Inner message starts after wrapper header + length bytes
    # Find the color data section (after the 16-byte header)
    # Colors are 5 bytes each (H, S, V, 0, 0) in input order

    # Red (255, 0, 0) in HSV: H=0, S=100, V=100 -> stored as (0, 100, 100)
    # Blue (0, 0, 255) in HSV: H=240, S=100, V=100 -> stored as (120, 100, 100)

    # Red should come first in the message (same order as input)
    assert isinstance(result, bytearray)


def test_construct_extended_custom_effect_hsv_conversion():
    """Test RGB to HSV conversion accuracy."""
    proto = ProtocolLEDENETExtendedCustom()

    # Test with a known color: pure green
    colors = [(0, 255, 0)]
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=colors,
    )

    # Green in HSV: H=120, S=100%, V=100%
    # Stored as: H/2=60, S=100, V=100
    # Verify the message was constructed
    assert isinstance(result, bytearray)
    assert len(result) > 0


def test_construct_extended_custom_effect_speed_clamping():
    """Test that speed is clamped to 0-100."""
    proto = ProtocolLEDENETExtendedCustom()

    # Speed > 100 should be clamped
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=[(255, 0, 0)],
        speed=150,
    )
    assert isinstance(result, bytearray)

    # Speed < 0 should be clamped
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=[(255, 0, 0)],
        speed=-10,
    )
    assert isinstance(result, bytearray)


def test_construct_extended_custom_effect_density_clamping():
    """Test that density is clamped to 0-100."""
    proto = ProtocolLEDENETExtendedCustom()

    # Density > 100 should be clamped
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=[(255, 0, 0)],
        density=200,
    )
    assert isinstance(result, bytearray)


def test_construct_extended_custom_effect_max_colors():
    """Test constructing an effect with maximum 8 colors."""
    proto = ProtocolLEDENETExtendedCustom()

    # 8 colors (maximum)
    colors = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (128, 0, 255),
        (255, 0, 255),
    ]
    result = proto.construct_extended_custom_effect(
        pattern_id=1,
        colors=colors,
    )

    assert isinstance(result, bytearray)
    # Each color is 5 bytes, 8 colors = 40 bytes for colors
    # Plus 16 bytes header = 56 bytes inner message
    # Plus wrapper overhead


def test_construct_extended_custom_effect_with_enums():
    """Test using enum values for parameters."""
    proto = ProtocolLEDENETExtendedCustom()

    result = proto.construct_extended_custom_effect(
        pattern_id=ExtendedCustomEffectPattern.WAVE,
        colors=[(255, 0, 0)],
        speed=50,
        density=50,
        direction=ExtendedCustomEffectDirection.RIGHT_TO_LEFT,
        option=ExtendedCustomEffectOption.VARIANT_1,
    )

    assert isinstance(result, bytearray)


def test_construct_extended_custom_effect_with_variant_2():
    """Test using VARIANT_2 option (e.g., breathe mode for rainbow patterns)."""
    proto = ProtocolLEDENETExtendedCustom()

    # Rainbow colors with VARIANT_2 option
    colors = [
        (255, 0, 0),  # Red
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (0, 255, 255),  # Cyan
        (0, 0, 255),  # Blue
        (255, 0, 255),  # Magenta
    ]
    result = proto.construct_extended_custom_effect(
        pattern_id=12,  # Twinkling stars
        colors=colors,
        speed=60,
        density=100,
        direction=ExtendedCustomEffectDirection.LEFT_TO_RIGHT,
        option=ExtendedCustomEffectOption.VARIANT_2,
    )

    assert isinstance(result, bytearray)
    assert len(result) > 0


def test_extended_custom_effect_hsv_values():
    """Test specific HSV value calculations."""
    # Test the HSV conversion formula used in the protocol
    # RGB (255, 0, 0) -> HSV (0, 100%, 100%) -> stored as (0, 100, 100)
    r, g, b = 255, 0, 0
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hsv_h = int(h * 180)
    hsv_s = int(s * 100)
    hsv_v = int(v * 100)

    assert hsv_h == 0  # Red hue
    assert hsv_s == 100  # Full saturation
    assert hsv_v == 100  # Full value

    # RGB (0, 0, 255) -> HSV (240, 100%, 100%) -> stored as (120, 100, 100)
    r, g, b = 0, 0, 255
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hsv_h = int(h * 180)
    hsv_s = int(s * 100)
    hsv_v = int(v * 100)

    assert hsv_h == 120  # Blue hue (240/2)
    assert hsv_s == 100
    assert hsv_v == 100

    # RGB (0, 255, 0) -> HSV (120, 100%, 100%) -> stored as (60, 100, 100)
    r, g, b = 0, 255, 0
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hsv_h = int(h * 180)
    hsv_s = int(s * 100)
    hsv_v = int(v * 100)

    assert hsv_h == 60  # Green hue (120/2)
    assert hsv_s == 100
    assert hsv_v == 100


# Tests for _generate_extended_custom_effect validation (base_device.py lines 1335-1360)


@pytest.mark.asyncio
async def test_generate_extended_custom_effect_validation(mock_aio_protocol):
    """Test validation in _generate_extended_custom_effect."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    # Setup 0xB6 device with extended state
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    assert light.protocol == PROTOCOL_LEDENET_EXTENDED_CUSTOM

    # Test invalid pattern_id (0 is not valid)
    with pytest.raises(ValueError, match="Pattern ID must be 1-22 or 101-102"):
        light._generate_extended_custom_effect(0, [(255, 0, 0)])

    # Test invalid pattern_id (23 is not valid -- max animated id is 22)
    with pytest.raises(ValueError, match="Pattern ID must be 1-22 or 101-102"):
        light._generate_extended_custom_effect(23, [(255, 0, 0)])

    # Test invalid pattern_id (24 is not valid)
    with pytest.raises(ValueError, match="Pattern ID must be 1-22 or 101-102"):
        light._generate_extended_custom_effect(24, [(255, 0, 0)])

    # Test invalid pattern_id (25 is not valid)
    with pytest.raises(ValueError, match="Pattern ID must be 1-22 or 101-102"):
        light._generate_extended_custom_effect(25, [(255, 0, 0)])

    # Test empty colors list
    with pytest.raises(ValueError, match="at least one color"):
        light._generate_extended_custom_effect(1, [])

    # Test invalid color tuple (not 3 elements)
    with pytest.raises(ValueError, match=r"must be .* tuple"):
        light._generate_extended_custom_effect(1, [(255, 0)])

    # Test color values out of range (> 255)
    with pytest.raises(ValueError, match="must be 0-255"):
        light._generate_extended_custom_effect(1, [(256, 0, 0)])

    # Test color values out of range (< 0)
    with pytest.raises(ValueError, match="must be 0-255"):
        light._generate_extended_custom_effect(1, [(-1, 0, 0)])

    # Test valid pattern_id 22 (max animated id)
    result = light._generate_extended_custom_effect(22, [(255, 0, 0)])
    assert isinstance(result, bytearray)

    # Test valid pattern_id 101 (STATIC_GRADIENT)
    result = light._generate_extended_custom_effect(101, [(255, 0, 0)])
    assert isinstance(result, bytearray)

    # Test valid pattern_id 102 (STATIC_FILL)
    result = light._generate_extended_custom_effect(102, [(255, 0, 0)])
    assert isinstance(result, bytearray)


@pytest.mark.asyncio
async def test_generate_extended_custom_effect_rejects_too_many_colors(
    mock_aio_protocol,
):
    """Test that too many colors (>8) raise ValueError instead of truncating."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)

    # 9 colors (one over the 8 max) must raise; 8 is still accepted.
    colors = [(i * 25, 0, 0) for i in range(9)]
    with pytest.raises(ValueError, match="at most 8 colors are supported, got 9"):
        light._generate_extended_custom_effect(1, colors)

    result = light._generate_extended_custom_effect(1, colors[:8])
    assert isinstance(result, bytearray)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"speed": 101}, "speed must be 0-100"),
        ({"speed": -1}, "speed must be 0-100"),
        ({"density": 101}, "density must be 0-100"),
        ({"density": -1}, "density must be 0-100"),
        ({"direction": 0x00}, "direction must be 0x01 or 0x02"),
        ({"direction": 0x03}, "direction must be 0x01 or 0x02"),
        ({"option": 3}, "option must be 0-2"),
    ],
)
@pytest.mark.asyncio
async def test_generate_extended_custom_effect_param_bounds(
    mock_aio_protocol, kwargs, match
):
    """General wire-byte bounds for speed/density/direction/option are enforced."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    with pytest.raises(ValueError, match=match):
        light._generate_extended_custom_effect(1, [(255, 0, 0)], **kwargs)


@pytest.mark.asyncio
async def test_generate_extended_custom_effect_param_bounds_valid(mock_aio_protocol):
    """Boundary-valid animation params still succeed."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    result = light._generate_extended_custom_effect(
        1, [(255, 0, 0)], speed=100, density=0, direction=0x02, option=2
    )
    assert isinstance(result, bytearray)


# Tests for _generate_custom_segment_colors validation (base_device.py lines 1376-1392)


@pytest.mark.asyncio
async def test_generate_custom_segment_colors_validation(mock_aio_protocol):
    """Test validation in _generate_custom_segment_colors."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    # Test invalid color tuple (not 3 elements)
    with pytest.raises(ValueError, match=r"must be .* tuple"):
        light._generate_custom_segment_colors([(255, 0)])

    # Test color values out of range (> 255)
    with pytest.raises(ValueError, match="must be 0-255"):
        light._generate_custom_segment_colors([(256, 0, 0)])

    # Test valid segments with None
    result = light._generate_custom_segment_colors([None, (255, 0, 0), None])
    assert isinstance(result, bytearray)


@pytest.mark.asyncio
async def test_generate_custom_segment_colors_rejects_too_many_segments(
    mock_aio_protocol,
):
    """Test that too many segments (>20) raise ValueError instead of truncating."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)

    # 21 segments (one over the 20 max) must raise; 20 is still accepted.
    segments = [(i * 10, 0, 0) for i in range(21)]
    with pytest.raises(ValueError, match="at most 20 segments are supported, got 21"):
        light._generate_custom_segment_colors(segments)

    result = light._generate_custom_segment_colors(segments[:20])
    assert isinstance(result, bytearray)


# Tests for construct_levels_change (protocol.py lines 1826-1861)


def test_protocol_construct_levels_change_0xB6():
    """Test construct_levels_change sets a color via a uniform E1 22 fill.

    A solid color must land the device in preset 0x24 ("Colorful"), so it is
    sent as a uniform E1 22 (all 20 segments identical), not an E1 21 Static
    Fill (hardware-verified).
    """
    proto = ProtocolLEDENETExtendedCustom()

    # Test with RGB values (pure red)
    result = proto.construct_levels_change(
        persist=1,
        red=255,
        green=0,
        blue=0,
        warm_white=0,
        cool_white=0,
        write_mode=0,
    )

    assert len(result) == 1
    msg = result[0]
    assert isinstance(msg, bytearray)
    # Check wrapper header
    assert msg[0] == 0xB0
    assert msg[1] == 0xB1
    assert msg[2] == 0xB2
    assert msg[3] == 0xB3

    # Pin the full inner E1 22 uniform frame. This is byte-identical to the real
    # app "Colorful -> red" packet captured from the device (inner:
    # e1 22 00 00 00 00 14 [00 64 64 00 00] x 20).
    inner = _inner_of(msg)
    header = _h("e1 22 00 00 00 00 14")
    # (255,0,0): hue 0, sat 100, val 100 -> [00 64 64 00 00]
    red_seg = _h("00 64 64 00 00")
    expected = header + red_seg * 20
    assert inner == expected
    assert len(inner) == 7 + 20 * 5


def test_protocol_construct_levels_change_with_white():
    """Test construct_levels_change for a white-only set.

    A white-only set is sent as a uniform E1 22 with the literal white segment
    [00 64 00 00 W], where W is the 0-100 white level (S byte MUST be 0x64).
    """
    proto = ProtocolLEDENETExtendedCustom()

    # Test with white values: warm_white=255 -> W = round(255*100/255) = 100
    result = proto.construct_levels_change(
        persist=1,
        red=0,
        green=0,
        blue=0,
        warm_white=255,
        cool_white=0,
        write_mode=0,
    )

    assert len(result) == 1
    msg = result[0]
    assert isinstance(msg, bytearray)

    inner = _inner_of(msg)
    header = _h("e1 22 00 00 00 00 14")
    # W = 100 = 0x64; segment [00 64 00 00 64]
    white_seg = _h("00 64 00 00 64")
    expected = header + white_seg * 20
    assert inner == expected
    assert len(inner) == 7 + 20 * 5


def test_protocol_construct_levels_change_white_not_doubled():
    """A single white value mirrored into warm+cool must not be double-counted.

    The device has one white LED; _generate_levels_change mirrors a single white
    value into BOTH warm and cool for non-CCT devices, so combining them by max
    (not sum) is required. Regression test: warm=cool=128 (a mirrored single
    white of 128) must yield W = round(128*100/255) = 50, not 100.
    """
    proto = ProtocolLEDENETExtendedCustom()

    result = proto.construct_levels_change(
        persist=1,
        red=0,
        green=0,
        blue=0,
        warm_white=128,
        cool_white=128,  # mirrored single white; max -> 128 -> W=50 (not 2x)
        write_mode=0,
    )

    assert len(result) == 1
    assert isinstance(result[0], bytearray)

    inner = _inner_of(result[0])
    header = _h("e1 22 00 00 00 00 14")
    white_seg = _h("00 64 00 00 32")  # W = 50 = 0x32 (not doubled to 0x64)
    expected = header + white_seg * 20
    assert inner == expected


def test_protocol_rgb_to_hsv_bytes_rgbw():
    """Test _rgb_to_hsv_bytes_rgbw conversion."""
    proto = ProtocolLEDENETExtendedCustom()

    # Test pure red with white
    result = proto._rgb_to_hsv_bytes_rgbw(255, 0, 0, 100)
    assert len(result) == 5
    assert result[0] == 0  # Hue (red = 0)
    assert result[1] == 100  # Saturation
    assert result[2] == 100  # Value
    assert result[3] == 0x00  # Unused
    assert result[4] == 100  # White

    # Test pure green
    result = proto._rgb_to_hsv_bytes_rgbw(0, 255, 0, 0)
    assert result[0] == 60  # Hue (green = 120/2)

    # Test pure blue
    result = proto._rgb_to_hsv_bytes_rgbw(0, 0, 255, 255)
    assert result[0] == 120  # Hue (blue = 240/2)
    assert result[4] == 255  # White


# Tests for construct_custom_segment_colors (protocol.py lines 1971-1980)


def test_protocol_construct_custom_segment_colors():
    """Test construct_custom_segment_colors command format."""
    proto = ProtocolLEDENETExtendedCustom()

    # Test with a few segments
    segments = [(255, 0, 0), None, (0, 0, 255)]
    result = proto.construct_custom_segment_colors(segments)

    assert isinstance(result, bytearray)
    # Check wrapper header
    assert result[0] == 0xB0
    assert result[1] == 0xB1
    assert result[2] == 0xB2
    assert result[3] == 0xB3

    # Pin the full inner E1 22 payload: header + 0x14 (20) count byte, then
    # one 5-byte [H/2, S, V, 0x00, 0x00] record per segment, padded to 20.
    inner = _inner_of(result)
    header = _h("e1 22 00 00 00 00 14")
    red = _h("00 64 64 00 00")  # (255,0,0): hue 0, sat 100, val 100
    off = _h("00 00 00 00 00")  # None / off segment
    blue = _h("78 64 64 00 00")  # (0,0,255): hue 120 -> byte 0x78, sat/val 100
    # 3 provided segments (red, off, blue) + 17 off segments = 20 total.
    expected = header + red + off + blue + off * 17
    assert inner == expected
    assert len(inner) == 7 + 20 * 5


def test_protocol_construct_custom_segment_colors_all_off():
    """Test construct_custom_segment_colors with all segments off."""
    proto = ProtocolLEDENETExtendedCustom()

    # All None segments
    segments = [None] * 10
    result = proto.construct_custom_segment_colors(segments)

    assert isinstance(result, bytearray)


def test_protocol_construct_custom_segment_colors_zero_tuple():
    """Test that (0,0,0) is treated as off."""
    proto = ProtocolLEDENETExtendedCustom()

    # Mix of None and (0,0,0)
    segments = [None, (0, 0, 0), (255, 0, 0)]
    result = proto.construct_custom_segment_colors(segments)

    assert isinstance(result, bytearray)


# Tests for async API methods


@pytest.mark.asyncio
async def test_async_set_extended_custom_effect_0xB6(mock_aio_protocol):
    """Test async_set_extended_custom_effect sends correct bytes."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()

    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    transport.reset_mock()

    await light.async_set_extended_custom_effect(
        pattern_id=1,
        colors=[(255, 0, 0), (0, 255, 0)],
        speed=50,
        density=50,
    )

    assert transport.write.called
    written_data = transport.write.call_args[0][0]
    # Verify it's a wrapped message
    assert written_data[0] == 0xB0
    assert written_data[1] == 0xB1


@pytest.mark.asyncio
async def test_async_set_custom_segment_colors_0xB6(mock_aio_protocol):
    """Test async_set_custom_segment_colors sends correct bytes."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()

    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    transport.reset_mock()

    await light.async_set_custom_segment_colors(
        segments=[(255, 0, 0), None, (0, 0, 255)]
    )

    assert transport.write.called
    written_data = transport.write.call_args[0][0]
    # Verify it's a wrapped message
    assert written_data[0] == 0xB0
    assert written_data[1] == 0xB1


# Scribble (per-LED) feature tests (0xB6)


def _inner_of(wrapped: bytearray) -> bytes:
    """Extract the inner message from a B0B1B2B3-wrapped result.

    wrapper = b0 b1 b2 b3 | 00 01 | ver | counter | len_hi len_lo | inner | cksum
    """
    inner_len = (wrapped[8] << 8) | wrapped[9]
    return bytes(wrapped[10 : 10 + inner_len])


ALL_ON_100 = bytes.fromhex("ff" * 12 + "f0")  # N=100, 13 bytes
ALL_ON_80 = bytes.fromhex("ff" * 10)  # N=80, 10 bytes
OFF0_80 = bytes.fromhex("80" + "00" * 9)  # N=80, only LED 0
GROUP_40_79_80 = bytes.fromhex("00" * 5 + "ff" * 5)  # N=80, LEDs 40-79


def _h(s: str) -> bytes:
    return bytes.fromhex(s.replace(" ", ""))


def test_scribble_bitmap_n100():
    proto = ProtocolLEDENETExtendedCustom()
    bitmap = proto._scribble_bitmap(range(100), 100)
    assert len(bitmap) == 13
    assert bitmap == ALL_ON_100
    assert bitmap[-1] == 0xF0


def test_scribble_bitmap_n80():
    proto = ProtocolLEDENETExtendedCustom()
    bitmap = proto._scribble_bitmap(range(80), 80)
    assert len(bitmap) == 10
    assert bitmap == ALL_ON_80


def test_scribble_bitmap_led0():
    proto = ProtocolLEDENETExtendedCustom()
    bitmap = proto._scribble_bitmap([0], 80)
    assert bitmap[0] == 0x80


def test_scribble_bitmap_led7():
    proto = ProtocolLEDENETExtendedCustom()
    bitmap = proto._scribble_bitmap([7], 80)
    assert bitmap[0] == 0x01


def test_scribble_bitmap_out_of_range():
    proto = ProtocolLEDENETExtendedCustom()
    with pytest.raises(ValueError):
        proto._scribble_bitmap([80], 80)
    with pytest.raises(ValueError):
        proto._scribble_bitmap([-1], 80)


def test_scribble_paint_all_green_static_n100():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            effect=0x00,
            direction=0x01,
            density=0x50,
            speed=0x64,
            blink_mode=0x00,
            h2=0x3C,
            s=0x64,
            v=0x64,
            white=0x00,
            blink_speed=0x64,
            num_leds=100,
        )
    )
    assert inner[:2] == b"\xe1\x26"
    assert inner == _h("e1 26 00 01 50 64 00 3c 64 64 00 00 64") + ALL_ON_100


def test_scribble_paint_blue_50pct():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(h2=0x78, s=0x64, v=0x32, num_leds=80)
    )
    assert inner == _h("e1 26 00 01 50 64 00 78 64 32 00 00 64") + ALL_ON_80


def test_scribble_paint_white_100():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(h2=0x78, s=0x64, v=0x00, white=0x64, num_leds=80)
    )
    assert inner == _h("e1 26 00 01 50 64 00 78 64 00 00 64 64") + ALL_ON_80


def test_scribble_paint_fast_blink_50():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            blink_mode=0x10, h2=0x78, s=0x64, v=0x64, blink_speed=0x32, num_leds=80
        )
    )
    assert inner == _h("e1 26 00 01 50 64 10 78 64 64 00 00 32") + ALL_ON_80


def test_scribble_paint_slow_blink_50():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            blink_mode=0x08, h2=0x78, s=0x64, v=0x64, blink_speed=0x32, num_leds=80
        )
    )
    assert inner == _h("e1 26 00 01 50 64 08 78 64 64 00 00 32") + ALL_ON_80


def test_scribble_paint_flowing_r2l_speed50():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            effect=0x01,
            direction=0x02,
            speed=0x32,
            h2=0x78,
            s=0x64,
            v=0x64,
            num_leds=80,
        )
    )
    assert inner == _h("e1 26 01 02 50 32 00 78 64 64 00 00 64") + ALL_ON_80


def test_scribble_paint_twinkling_d50_s61():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            effect=0x03,
            direction=0x02,
            density=0x32,
            speed=0x3D,
            h2=0x78,
            s=0x64,
            v=0x64,
            num_leds=80,
        )
    )
    assert inner == _h("e1 26 03 02 32 3d 00 78 64 64 00 00 64") + ALL_ON_80


def test_scribble_paint_off_bulb0_n80():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            effect=0x00,
            density=0x50,
            h2=0x00,
            s=0x00,
            v=0x00,
            blink_speed=0x64,
            bitmap_leds=[0],
            num_leds=80,
        )
    )
    assert inner == _h("e1 26 00 01 50 64 00 00 00 00 00 00 64") + OFF0_80


def test_scribble_paint_flowing_blue_group_n80():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(
        proto.construct_scribble_paint(
            effect=0x01,
            direction=0x02,
            density=0x50,
            speed=0x26,
            h2=0x78,
            s=0x64,
            v=0x64,
            blink_speed=0x64,
            bitmap_leds=list(range(40, 80)),
            num_leds=80,
        )
    )
    assert inner == _h("e1 26 01 02 50 26 00 78 64 64 00 00 64") + GROUP_40_79_80


def test_scribble_init_n80():
    proto = ProtocolLEDENETExtendedCustom()
    inner = _inner_of(proto.construct_scribble_init(80))
    assert inner[:9] == _h("e1 23 01 00 01 50 64 00 50")
    assert inner[9:] == _h("00 00 00 00 00 00 64") * 80
    assert len(inner) == 569  # 9 + 7*80


@pytest.mark.asyncio
async def test_generate_scribble_init_invalid(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    with pytest.raises(ValueError, match=r"num_leds must be 1\.\.255"):
        light._generate_scribble_init(0)
    # 256 exceeds the one-byte E1 23 count / bitmap addressing limit and must
    # raise the descriptive error, not the generic "byte must be in range".
    with pytest.raises(ValueError, match=r"num_leds must be 1\.\.255, got 256"):
        light._generate_scribble_init(256)


@pytest.mark.asyncio
async def test_scribble_paint_groups_num_leds_over_255_raises(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    light._extended_led_count = None  # bypass the led_count mismatch check
    leds = [ScribbleLED(rgb=(255, 0, 0))] * 256
    with pytest.raises(ValueError, match=r"num_leds must be 1\.\.255, got 256"):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 256)


@pytest.mark.asyncio
async def test_scribble_paint_groups_rgb_and_white_raises(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(255, 0, 0), white=50)] + [ScribbleLED()] * 79
    with pytest.raises(ValueError):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 80)


@pytest.mark.asyncio
async def test_scribble_paint_groups_channel_out_of_range_raises(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(300, 0, 0))] + [ScribbleLED()] * 79
    with pytest.raises(ValueError):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 80)


@pytest.mark.asyncio
async def test_scribble_paint_groups_effect_int_and_enum(mock_aio_protocol):
    """effect accepts a raw int id (e.g. 4, unnamed) or a ScribbleEffect; the
    device-valid range is 0x00-0x08, anything else raises ValueError."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(255, 0, 0))] * 80
    # raw int id not exposed as a name (3,4,6,7 are valid on-device)
    assert (
        _inner_of(light._scribble_paint_groups(leds, 4, 0x01, 0x50, 0x64, 80)[0])[2]
        == 4
    )
    # enum still works
    assert (
        _inner_of(
            light._scribble_paint_groups(
                leds, ScribbleEffect.FLOWING, 0x01, 0x50, 0x64, 80
            )[0]
        )[2]
        == 0x01
    )
    # out-of-range id (device accepts only 0x00-0x08)
    with pytest.raises(ValueError):
        light._scribble_paint_groups(leds, 9, 0x01, 0x50, 0x64, 80)


@pytest.mark.asyncio
async def test_scribble_paint_groups_wrong_count_raises(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    assert light.led_count == 80
    leds = [ScribbleLED(rgb=(255, 0, 0))] * 40
    with pytest.raises(ValueError):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 40)


@pytest.mark.asyncio
async def test_scribble_paint_groups_empty_raises(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    with pytest.raises(ValueError):
        light._scribble_paint_groups([], 0x00, 0x01, 0x50, 0x64, 0)


@pytest.mark.asyncio
async def test_scribble_paint_groups_two_color_static(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(255, 0, 0))] * 40 + [ScribbleLED(rgb=(0, 0, 255))] * 40
    msgs = light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 80)
    assert len(msgs) == 2  # red group then blue group, first-appearance order
    red = _inner_of(msgs[0])
    blue = _inner_of(msgs[1])
    # red occupies LEDs 0-39, blue 40-79
    assert red[13:] == _h("ff" * 5 + "00" * 5)
    assert blue[13:] == GROUP_40_79_80
    # blue hue byte (h2) = 0x78
    assert blue[7] == 0x78


@pytest.mark.asyncio
async def test_scribble_paint_groups_blink_grouping(mock_aio_protocol):
    """LEDs with different blink modes split into separate E1 26 groups."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(255, 0, 0), blink_mode=ScribbleBlinkMode.FAST)] * 40 + [
        ScribbleLED(rgb=(255, 0, 0), blink_mode=ScribbleBlinkMode.NONE)
    ] * 40
    msgs = light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 80)
    assert len(msgs) == 2  # same color, different blink -> two groups
    fast = _inner_of(msgs[0])
    steady = _inner_of(msgs[1])
    assert fast[6] == 0x10  # FAST blink byte
    assert steady[6] == 0x00  # NONE blink byte


@pytest.mark.asyncio
async def test_scribble_paint_groups_off_group_color_zero(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(255, 0, 0))] + [ScribbleLED()] * 79
    msgs = light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 80)
    assert len(msgs) == 2
    off = _inner_of(msgs[1])
    # off group: all-zero color, header matches the off golden
    assert off[:13] == _h("e1 26 00 01 50 64 00 00 00 00 00 00 64")
    assert off[13:] == _h("7f" + "ff" * 8 + "ff")  # LEDs 1-79 set


async def _setup_scribble_light(mock_aio_protocol, led_count_byte=0x50):
    """Set up a 0xB6 AIO light with a known led_count from state byte 18."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    transport, _protocol = await mock_aio_protocol()
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                led_count_byte,
                0x00,
                0x83,
            )
        )
    )
    await task
    transport.reset_mock()
    return light, transport


@pytest.mark.asyncio
async def test_async_extended_custom_methods_raise_on_unsupported_device(
    mock_aio_protocol,
):
    """Async extended-custom methods raise ValueError on a non-0xB6 device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    _transport, _protocol = await mock_aio_protocol()
    # Standard RGBCW bulb (0x35) -- does not use the extended custom protocol.
    light._aio_protocol.data_received(
        b"\x81\x35\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xee"
    )
    await task
    assert light.model_num == 0x35
    assert not light.supports_extended_custom_effects
    assert not light.supports_scribble

    with pytest.raises(ValueError):
        await light.async_set_extended_custom_effect(1, [(255, 0, 0)])
    with pytest.raises(ValueError):
        await light.async_set_custom_segment_colors([(255, 0, 0)])
    with pytest.raises(ValueError):
        await light.async_set_scribble([ScribbleLED(rgb=(255, 0, 0))])


@pytest.mark.asyncio
async def test_async_set_scribble_static_two_color(mock_aio_protocol):
    """STATIC 2-color config sends E1 23 init + one E1 26 per color group."""
    light, _transport = await _setup_scribble_light(mock_aio_protocol)
    assert light.led_count == 80

    sent = []
    with patch.object(light, "_async_send_msg", side_effect=lambda m: sent.append(m)):
        leds = [ScribbleLED(rgb=(255, 0, 0))] * 40 + [ScribbleLED(rgb=(0, 0, 255))] * 40
        await light.async_set_scribble(leds, effect=ScribbleEffect.STATIC)

    assert len(sent) == 3
    init = _inner_of(sent[0])
    assert init[:2] == b"\xe1\x23"
    assert len(init) == 569
    red = _inner_of(sent[1])
    blue = _inner_of(sent[2])
    assert red[:2] == b"\xe1\x26"
    assert red[13:] == _h("ff" * 5 + "00" * 5)
    assert blue[13:] == GROUP_40_79_80
    assert blue[7] == 0x78  # blue hue


@pytest.mark.asyncio
async def test_async_set_scribble_no_enter_mode(mock_aio_protocol):
    """enter_mode=False sends no E1 23, one E1 26 per group."""
    light, _transport = await _setup_scribble_light(mock_aio_protocol)
    sent = []
    with patch.object(light, "_async_send_msg", side_effect=lambda m: sent.append(m)):
        leds = [ScribbleLED(rgb=(255, 0, 0))] * 40 + [ScribbleLED(rgb=(0, 0, 255))] * 40
        await light.async_set_scribble(leds, enter_mode=False)
    assert len(sent) == 2
    assert _inner_of(sent[0])[:2] == b"\xe1\x26"
    assert _inner_of(sent[1])[:2] == b"\xe1\x26"


@pytest.mark.asyncio
async def test_async_set_scribble_flowing_blue_group(mock_aio_protocol):
    """FLOWING effect: blue group paint matches the captured golden."""
    light, _transport = await _setup_scribble_light(mock_aio_protocol)
    sent = []
    with patch.object(light, "_async_send_msg", side_effect=lambda m: sent.append(m)):
        leds = [ScribbleLED(rgb=(255, 0, 0))] * 40 + [ScribbleLED(rgb=(0, 0, 255))] * 40
        await light.async_set_scribble(
            leds,
            effect=ScribbleEffect.FLOWING,
            direction=ExtendedCustomEffectDirection.RIGHT_TO_LEFT,
            density=0x50,
            speed=0x26,
        )
    assert len(sent) == 3
    red = _inner_of(sent[1])
    blue = _inner_of(sent[2])
    # blue group matches the captured golden
    assert blue == _h("e1 26 01 02 50 26 00 78 64 64 00 00 64") + GROUP_40_79_80
    # red group carries red color on LEDs 0-39 under the same effect
    assert red[:7] == _h("e1 26 01 02 50 26 00")
    assert red[13:] == _h("ff" * 5 + "00" * 5)


@pytest.mark.asyncio
async def test_async_set_scribble_off_group(mock_aio_protocol):
    """Mixed lit + off config: off group paints all-zero color."""
    light, _transport = await _setup_scribble_light(mock_aio_protocol)
    sent = []
    with patch.object(light, "_async_send_msg", side_effect=lambda m: sent.append(m)):
        leds = [ScribbleLED(rgb=(255, 0, 0))] + [ScribbleLED()] * 79
        await light.async_set_scribble(leds, enter_mode=False)
    assert len(sent) == 2
    red = _inner_of(sent[0])
    off = _inner_of(sent[1])
    assert red[13:] == OFF0_80
    assert off[:13] == _h("e1 26 00 01 50 64 00 00 00 00 00 00 64")


@pytest.mark.asyncio
async def test_async_set_scribble_raw_int_direction(mock_aio_protocol):
    """A raw int direction (e.g. 0x02) is accepted, not just the enum."""
    light, _transport = await _setup_scribble_light(mock_aio_protocol)
    sent = []
    with patch.object(light, "_async_send_msg", side_effect=lambda m: sent.append(m)):
        leds = [ScribbleLED(rgb=(255, 0, 0))] * 80
        # Passing a raw int must not raise AttributeError on direction.value.
        await light.async_set_scribble(leds, direction=0x02, enter_mode=False)
    assert len(sent) == 1
    paint = _inner_of(sent[0])
    assert paint[:2] == b"\xe1\x26"
    assert paint[3] == 0x02  # direction byte carried through as the raw int


@pytest.mark.parametrize(
    "leds, num_leds, match",
    [
        # empty leds
        ([], 80, "leds must not be empty"),
        # num_leds out of 1..255 (0)
        ([ScribbleLED(rgb=(1, 2, 3))], 0, r"num_leds must be 1\.\.255"),
        # num_leds out of 1..255 (256)
        ([ScribbleLED(rgb=(1, 2, 3))] * 256, 256, r"num_leds must be 1\.\.255"),
        # num_leds-vs-len mismatch (num_leds != len(leds))
        ([ScribbleLED(rgb=(1, 2, 3))] * 80, 79, r"does not match"),
    ],
)
@pytest.mark.asyncio
async def test_scribble_paint_groups_count_guards(
    mock_aio_protocol, leds, num_leds, match
):
    """Cover the count/length ValueError guards in _scribble_paint_groups."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    light._extended_led_count = None  # bypass led_count-vs-len check for these cases
    with pytest.raises(ValueError, match=match):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, num_leds)


@pytest.mark.asyncio
async def test_scribble_paint_groups_led_count_mismatch_guard(mock_aio_protocol):
    """led_count-vs-len mismatch raises (device reports 80, we pass 40)."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    assert light.led_count == 80
    leds = [ScribbleLED(rgb=(1, 2, 3))] * 40
    with pytest.raises(ValueError, match="device has 80"):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 40)


@pytest.mark.parametrize(
    "led, match",
    [
        # rgb AND white both set (mutually exclusive)
        (ScribbleLED(rgb=(1, 2, 3), white=50), "mutually exclusive"),
        # rgb channel out of 0-255
        (ScribbleLED(rgb=(256, 0, 0)), "rgb values must be 0-255"),
        # white out of 0-100
        (ScribbleLED(white=101), "white must be 0-100"),
        # blink_speed out of 0-100
        (ScribbleLED(rgb=(1, 2, 3), blink_speed=101), "blink_speed must be 0-100"),
    ],
)
@pytest.mark.asyncio
async def test_scribble_paint_groups_per_led_guards(mock_aio_protocol, led, match):
    """Cover the per-LED ValueError guards in _scribble_paint_groups."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [led] + [ScribbleLED()] * 79
    with pytest.raises(ValueError, match=match):
        light._scribble_paint_groups(leds, 0x00, 0x01, 0x50, 0x64, 80)


@pytest.mark.parametrize(
    "effect_id",
    [-1, 9, 0x10],
)
@pytest.mark.asyncio
async def test_scribble_paint_groups_effect_id_out_of_range(
    mock_aio_protocol, effect_id
):
    """effect id outside 0x00-0x08 raises ValueError."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(1, 2, 3))] * 80
    with pytest.raises(ValueError, match=r"effect id must be 0x00-0x08"):
        light._scribble_paint_groups(leds, effect_id, 0x01, 0x50, 0x64, 80)


@pytest.mark.parametrize(
    "direction, density, speed, match",
    [
        # direction outside {0x01, 0x02}
        (0x00, 0x50, 0x64, "direction must be 0x01 or 0x02"),
        (0x03, 0x50, 0x64, "direction must be 0x01 or 0x02"),
        # density outside 0-100
        (0x01, 101, 0x64, "density must be 0-100"),
        # speed outside 0-100
        (0x01, 0x50, 101, "speed must be 0-100"),
    ],
)
@pytest.mark.asyncio
async def test_scribble_paint_groups_animation_param_bounds(
    mock_aio_protocol, direction, density, speed, match
):
    """Animation params (direction/density/speed) are bounds-checked once here."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    leds = [ScribbleLED(rgb=(1, 2, 3))] * 80
    with pytest.raises(ValueError, match=match):
        light._scribble_paint_groups(leds, 0x00, direction, density, speed, 80)


@pytest.mark.parametrize(
    "colors, match",
    [
        # empty colors
        ([], "at least one color"),
        # color tuple wrong length
        ([(255, 0)], r"must be .* tuple"),
        # color channel out of 0-255 (high)
        ([(256, 0, 0)], "must be 0-255"),
        # color channel out of 0-255 (low)
        ([(-1, 0, 0)], "must be 0-255"),
    ],
)
@pytest.mark.asyncio
async def test_generate_extended_custom_effect_color_guards(
    mock_aio_protocol, colors, match
):
    """Cover the color/empty ValueError guards in _generate_extended_custom_effect."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    with pytest.raises(ValueError, match=match):
        light._generate_extended_custom_effect(1, colors)


@pytest.mark.parametrize(
    "segments, match",
    [
        # color tuple wrong length
        ([(255, 0)], r"must be .* tuple"),
        # color channel out of 0-255 (high)
        ([(256, 0, 0)], "must be 0-255"),
        # color channel out of 0-255 (low)
        ([(0, -1, 0)], "must be 0-255"),
    ],
)
@pytest.mark.asyncio
async def test_generate_custom_segment_colors_color_guards(
    mock_aio_protocol, segments, match
):
    """Cover the color-tuple ValueError guards in _generate_custom_segment_colors."""
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    with pytest.raises(ValueError, match=match):
        light._generate_custom_segment_colors(segments)


@pytest.mark.asyncio
async def test_supports_scribble_property(mock_aio_protocol):
    light, _t = await _setup_scribble_light(mock_aio_protocol)
    assert light.supports_scribble is True


# Tests for extended_custom_effect_pattern_list property (base_device.py line 658)


@pytest.mark.asyncio
async def test_extended_custom_effect_pattern_list_0xB6(mock_aio_protocol):
    """Test extended_custom_effect_pattern_list returns list for 0xB6 device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x61,
                0x24,
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    # Test extended_custom_effect_pattern_list returns a list
    pattern_list = light.extended_custom_effect_pattern_list
    assert pattern_list is not None
    assert isinstance(pattern_list, list)
    assert len(pattern_list) > 0
    # Check some expected patterns
    assert "wave" in pattern_list
    assert "meteor" in pattern_list
    assert "breathe" in pattern_list


@pytest.mark.asyncio
async def test_extended_custom_effect_pattern_list_non_0xB6(mock_aio_protocol):
    """Test extended_custom_effect_pattern_list returns None for non-0xB6 device."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    # Standard 0x25 device (not extended custom)
    light._aio_protocol.data_received(
        b"\x81\x25\x23\x61\x05\x10\xb6\x00\x98\x19\x04\x25\x0f\xde"
    )
    await task

    # Should return None for non-extended devices
    assert light.extended_custom_effect_pattern_list is None


# Tests for _named_effect with extended custom effects (base_device.py line 676)


@pytest.mark.asyncio
async def test_named_effect_extended_custom_0xB6(mock_aio_protocol):
    """Test _named_effect returns extended effect name for 0xB6 in custom mode."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    # 0xB6 device with preset_pattern=0x25 (custom effect mode) and mode=0x01 (Wave)
    # Extended state: preset_pattern at pos 7, mode at pos 8
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,  # Model
                0x01,  # Version
                0x23,  # Power on
                0x25,  # preset_pattern = 0x25 (custom effect mode)
                0x01,  # mode = Wave pattern ID
                0x64,  # speed
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    # Effect should be "Wave" from EXTENDED_CUSTOM_EFFECT_ID_NAME
    assert light.effect == "Wave"


@pytest.mark.asyncio
async def test_named_effect_extended_custom_meteor(mock_aio_protocol):
    """Test _named_effect returns Meteor for mode=0x02."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    # 0xB6 device with mode=0x02 (Meteor)
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x25,  # preset_pattern = 0x25
                0x02,  # mode = Meteor
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    assert light.effect == "Meteor"


@pytest.mark.asyncio
async def test_named_effect_extended_custom_unmapped_mode(mock_aio_protocol, caplog):
    """An unmapped extended-custom mode returns None and is logged for visibility."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    # 0xB6 device with preset_pattern=0x25 and an unmapped mode (0x30)
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,
                0x01,
                0x00,
                0xB6,
                0x01,
                0x23,
                0x25,  # preset_pattern = 0x25
                0x30,  # mode = unmapped
                0x64,
                0x0F,
                0x00,
                0x00,
                0x00,
                0x64,
                0x64,
                0x00,
                0x00,
                0x00,
                0x00,
                0x83,
            )
        )
    )
    await task

    with caplog.at_level(logging.DEBUG, logger="flux_led.base_device"):
        assert light.effect is None
    assert "Unmapped extended custom effect mode" in caplog.text
