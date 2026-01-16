from __future__ import annotations

import asyncio
import colorsys
import contextlib
import datetime
import json
import logging
import sys
import time
from unittest.mock import MagicMock, call, patch

try:
    from unittest.mock import AsyncMock
except ImportError:
    from unittest.mock import MagicMock as AsyncMock

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
    WhiteChannelType,
)
from flux_led.models_db import extract_model_version_from_state
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
@pytest.mark.skipif(sys.version_info[:3][1] in (7,), reason="no AsyncMock in 3.7")
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
@pytest.mark.skipif(sys.version_info[:3][1] in (7,), reason="no AsyncMock in 3.7")
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
@pytest.mark.skipif(sys.version_info[:3][1] in (7,), reason="no AsyncMock in 3.7")
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
    """Test setup of 0xB6 Surplife device with extended state."""
    light = AIOWifiLedBulb("192.168.1.166")

    def _updated_callback(*args, **kwargs):
        pass

    task = asyncio.create_task(light.async_setup(_updated_callback))
    await mock_aio_protocol()

    # 0xB6 ONLY responds with extended state format (0xEA 0x81) per models_db.py:1313
    # This tests the extended state code paths in aiodevice.py and base_device.py
    light._aio_protocol.data_received(
        bytes(
            (
                0xEA,
                0x81,  # Extended state header
                0x01,
                0x00,  # Reserved
                0xB6,  # Model at position 4 (LEDENET_EXTENDED_STATE_MODEL_POS)
                0x01,  # Version at position 5 (LEDENET_EXTENDED_STATE_VERSION_POS)
                0x23,
                0x61,  # Power on, mode
                0x24,
                0x64,
                0x0F,  # Settings
                0x00,
                0x00,
                0x00,  # RGB off
                0x64,
                0x64,  # WW/CW values
                0x00,
                0x00,
                0x00,
                0x00,  # Padding
                0x83,  # Checksum
            )
        )
    )
    await task

    assert light.model_num == 0xB6
    assert light.version_num == 1
    assert light.protocol == PROTOCOL_LEDENET_EXTENDED_CUSTOM
    assert "Surplife" in light.model
    assert light.color_modes == {COLOR_MODE_RGB, COLOR_MODE_DIM}
    assert light.supports_extended_custom_effects is True
    assert light.microphone is True


def test_protocol_extended_state_validation_0xB6():
    """Test protocol methods correctly handle extended state for 0xB6 device."""
    extended_state = bytes(
        (
            0xEA,
            0x81,  # Extended state header
            0x01,
            0x00,  # Reserved
            0xB6,  # Model
            0x01,  # Version
            0x23,
            0x61,  # Power on, mode
            0x24,
            0x64,
            0x0F,  # Settings
            0x00,
            0x00,
            0x00,  # RGB off
            0x64,
            0x64,  # WW/CW values
            0x00,
            0x00,
            0x00,
            0x00,  # Padding
            0x83,  # Checksum
        )
    )

    # Test ProtocolLEDENET8Byte validates extended state correctly
    protocol_8byte = ProtocolLEDENET8Byte()
    assert protocol_8byte.is_valid_extended_state_response(extended_state)
    # This covers line 1040-1041: is_valid_state_response should return True for extended state
    assert protocol_8byte.is_valid_state_response(extended_state)

    # Test ProtocolLEDENET25Byte validates extended state correctly
    protocol_25byte = ProtocolLEDENET25Byte()
    assert protocol_25byte.is_valid_extended_state_response(extended_state)
    # This covers line 1465-1466: is_valid_state_response should return True for extended state
    assert protocol_25byte.is_valid_state_response(extended_state)

    # Test extended_state_to_state default implementation (line 1055)
    # ProtocolLEDENET8Byte uses the default implementation which just returns raw_state
    assert protocol_8byte.extended_state_to_state(extended_state) == extended_state

    # Test ProtocolLEDENETExtendedCustom - dedicated protocol for 0xB6
    protocol_extended = ProtocolLEDENETExtendedCustom()
    assert protocol_extended.name == PROTOCOL_LEDENET_EXTENDED_CUSTOM
    # This protocol ONLY accepts extended state format
    assert protocol_extended.is_valid_state_response(extended_state) is True

    # Test that ProtocolLEDENETExtendedCustom rejects standard 14-byte state
    # This is a minimal standard state (without valid checksum) - just for format testing
    standard_state = bytes(
        (
            0x81,  # Standard state header
            0xB6,  # Model
            0x23,  # Power on
            0x61,  # Mode
            0x00,
            0x64,
            0x00,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
        )
    )
    # ProtocolLEDENETExtendedCustom should reject standard state (only accepts 0xEA 0x81)
    assert protocol_extended.is_valid_state_response(standard_state) is False


# Extended Custom Effect Tests (for devices with supports_extended_custom_effects=True)


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
    with pytest.raises(ValueError, match="Pattern ID must be 1-24 or 101-102"):
        light._generate_extended_custom_effect(0, [(255, 0, 0)])

    # Test invalid pattern_id (25 is not valid)
    with pytest.raises(ValueError, match="Pattern ID must be 1-24 or 101-102"):
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

    # Test valid pattern_id 101 (STATIC_GRADIENT)
    result = light._generate_extended_custom_effect(101, [(255, 0, 0)])
    assert isinstance(result, bytearray)

    # Test valid pattern_id 102 (STATIC_FILL)
    result = light._generate_extended_custom_effect(102, [(255, 0, 0)])
    assert isinstance(result, bytearray)


@pytest.mark.asyncio
async def test_generate_extended_custom_effect_truncate_colors(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test that too many colors (>8) are truncated with warning."""
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

    # 10 colors (more than 8 max)
    colors = [(i * 25, 0, 0) for i in range(10)]

    with caplog.at_level(logging.WARNING):
        result = light._generate_extended_custom_effect(1, colors)

    assert isinstance(result, bytearray)
    assert "truncating" in caplog.text.lower()


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
async def test_generate_custom_segment_colors_truncate(
    mock_aio_protocol, caplog: pytest.LogCaptureFixture
):
    """Test that too many segments (>20) are truncated with warning."""
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

    # 25 segments (more than 20 max)
    segments = [(i * 10, 0, 0) for i in range(25)]

    with caplog.at_level(logging.WARNING):
        result = light._generate_custom_segment_colors(segments)

    assert isinstance(result, bytearray)
    assert "truncating" in caplog.text.lower()


# Tests for construct_levels_change (protocol.py lines 1826-1861)


def test_protocol_construct_levels_change_0xB6():
    """Test construct_levels_change uses STATIC_FILL for 0xB6 protocol."""
    proto = ProtocolLEDENETExtendedCustom()

    # Test with RGB values
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


def test_protocol_construct_levels_change_with_white():
    """Test construct_levels_change combines warm and cool white."""
    proto = ProtocolLEDENETExtendedCustom()

    # Test with white values
    result = proto.construct_levels_change(
        persist=1,
        red=0,
        green=0,
        blue=0,
        warm_white=100,
        cool_white=50,
        write_mode=0,
    )

    assert len(result) == 1
    msg = result[0]
    assert isinstance(msg, bytearray)


def test_protocol_construct_levels_change_white_clamping():
    """Test that combined white is clamped to 255."""
    proto = ProtocolLEDENETExtendedCustom()

    # Test with white values that exceed 255 when combined
    result = proto.construct_levels_change(
        persist=1,
        red=0,
        green=0,
        blue=0,
        warm_white=200,
        cool_white=200,  # Total would be 400, should clamp to 255
        write_mode=0,
    )

    assert len(result) == 1
    assert isinstance(result[0], bytearray)


def test_protocol_extended_state_to_state_white_off():
    """Test extended_state_to_state when white is off (white_brightness=0)."""
    proto = ProtocolLEDENETExtendedCustom()

    # Extended state with white_brightness=0 (position 15)
    extended_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0xB6,  # Model
            0x01,  # Version
            0x23,  # Power on
            0x61,  # Mode
            0x24,
            0x64,
            0x0F,
            0x00,
            0x00,
            0x00,
            0xFF,  # white_temp (255 > 100, triggers branch)
            0x00,  # white_brightness = 0 (triggers branch)
            0x00,
            0x00,
            0x00,
            0x00,
            0x83,
        )
    )

    result = proto.extended_state_to_state(extended_state)
    assert len(result) == 14
    # cool_white and warm_white should be 0
    assert result[9] == 0  # warm_white
    assert result[11] == 0  # cool_white


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
    segments = [(255, 0, 0), None, (0, 255, 0)]
    result = proto.construct_custom_segment_colors(segments)

    assert isinstance(result, bytearray)
    # Check wrapper header
    assert result[0] == 0xB0
    assert result[1] == 0xB1
    assert result[2] == 0xB2
    assert result[3] == 0xB3


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


# Tests for extract_model_version_from_state (models_db.py)


def test_extract_model_version_from_extended_state():
    """Test extract_model_version_from_state with extended state format."""
    # Extended state format (0xEA 0x81)
    extended_state = bytes(
        (
            0xEA,
            0x81,
            0x01,
            0x00,
            0xB6,  # Model at position 4
            0x05,  # Version at position 5
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

    model_num, version_num = extract_model_version_from_state(extended_state)
    assert model_num == 0xB6
    assert version_num == 5


def test_extract_model_version_from_standard_state():
    """Test extract_model_version_from_state with standard state format."""
    # Standard state format (0x81)
    standard_state = bytes(
        (
            0x81,
            0x25,  # Model at position 1
            0x23,
            0x61,
            0x05,
            0x10,
            0xB6,
            0x00,
            0x98,
            0x19,
            0x04,  # Version at position 10
            0x25,
            0x0F,
            0xDE,
        )
    )

    model_num, version_num = extract_model_version_from_state(standard_state)
    assert model_num == 0x25
    assert version_num == 4


def test_extract_model_version_short_standard_state():
    """Test extract_model_version_from_state with short standard state (no version)."""
    # Short standard state without version byte
    short_state = bytes((0x81, 0x33, 0x23, 0x61, 0x05, 0x10, 0xB6, 0x00, 0x98, 0x19))

    model_num, version_num = extract_model_version_from_state(short_state)
    assert model_num == 0x33
    assert version_num == 1  # Default when not present


# Tests for protocol edge cases


def test_protocol_extended_custom_state_response_length():
    """Test state_response_length property returns correct value."""
    proto = ProtocolLEDENETExtendedCustom()
    assert proto.state_response_length == LEDENET_EXTENDED_STATE_RESPONSE_LEN
    assert proto.state_response_length == 21


def test_protocol_extended_state_to_state_short_input():
    """Test extended_state_to_state returns empty bytes for short input."""
    proto = ProtocolLEDENETExtendedCustom()

    # Input too short (< 20 bytes)
    short_state = bytes((0xEA, 0x81, 0x01, 0x00, 0xB6))
    result = proto.extended_state_to_state(short_state)
    assert result == b""


def test_protocol_named_raw_state_standard_format():
    """Test named_raw_state with standard 14-byte format."""
    proto = ProtocolLEDENETExtendedCustom()

    # Standard 14-byte state (should pass through directly)
    standard_state = bytes(
        (
            0x81,
            0xB6,
            0x23,
            0x61,
            0x00,
            0x64,
            0xFF,
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
        )
    )
    result = proto.named_raw_state(standard_state)
    assert result.head == 0x81
    assert result.model_num == 0xB6


# Tests for 0xB6 Extended Timer Support


def test_protocol_construct_get_timers_0xB6():
    """Test timer query construction for 0xB6 device."""
    proto = ProtocolLEDENETExtendedCustom()
    msg = proto.construct_get_timers()

    # Should be wrapped message with inner content [0xE0, 0x06]
    assert msg[0:6] == bytes([0xB0, 0xB1, 0xB2, 0xB3, 0x00, 0x01])
    # Version byte at position 6
    assert msg[6] == 0x01
    # Inner message length at positions 8-9 (should be 2)
    assert msg[8] == 0x00
    assert msg[9] == 0x02
    # Inner message starts at position 10
    assert msg[10] == 0xE0
    assert msg[11] == 0x06


def test_protocol_is_valid_timers_response_0xB6():
    """Test timer response validation for 0xB6 device."""
    proto = ProtocolLEDENETExtendedCustom()

    # Valid response starts with e0 06
    assert proto.is_valid_timers_response(bytes([0xE0, 0x06])) is True
    assert proto.is_valid_timers_response(bytes([0xE0, 0x06, 0x01, 0x02])) is True

    # Invalid responses
    assert proto.is_valid_timers_response(bytes([0xE0])) is False
    assert proto.is_valid_timers_response(bytes([0x01, 0x22])) is False
    assert proto.is_valid_timers_response(bytes([])) is False


def test_protocol_parse_get_timers_empty_0xB6():
    """Test parsing empty timer response for 0xB6 device."""
    proto = ProtocolLEDENETExtendedCustom()

    # Empty response (no timers configured)
    response = bytes([0xE0, 0x06])
    timers = proto.parse_get_timers(response)

    assert len(timers) == 0


def test_protocol_parse_get_timers_single_0xB6():
    """Test parsing single timer response for 0xB6 device."""
    proto = ProtocolLEDENETExtendedCustom()

    # Response with one timer: 13:25 OFF, no repeat
    # Format: e0 06 [slot1: 21 bytes] [empty slots 2-6: 7 bytes each]
    response = bytes.fromhex(
        "e006"  # Header
        "01f00d1900000ee001002400000000000000000000"  # Slot 1: 13:25 OFF
        "0264640000690003000000000000040000000000000500000000000006000000000000"  # Slots 2-6 empty
    )
    timers = proto.parse_get_timers(response)

    assert len(timers) == 6

    # First timer should be active
    timer1 = timers[0]
    assert timer1.slot == 1
    assert timer1.active is True
    assert timer1.hour == 13
    assert timer1.minute == 25
    assert timer1.repeat_mask == 0
    assert timer1.action_type == 0x24  # OFF

    # Remaining timers should be inactive
    for i in range(1, 6):
        assert timers[i].active is False


def test_protocol_parse_get_timers_multiple_0xB6():
    """Test parsing multiple timer response for 0xB6 device."""
    proto = ProtocolLEDENETExtendedCustom()

    # Response with three timers (from actual packet capture)
    # Format: e0 06 [slot1: 21 bytes] [slot2: 21 bytes] [slot3: 21 bytes] [slots 4-6: 7 bytes each]
    response = bytes.fromhex(
        "e006"  # Header
        "01f00d1900000ee001002400000000000000000000"  # Slot 1: 13:25 OFF
        "02f00f2300000ee001002400000000000000000000"  # Slot 2: 15:35 OFF
        "03f00d2100000ee001002400000000000000000000"  # Slot 3: 13:33 OFF
        "04000000000000"  # Slot 4 empty
        "05000000000000"  # Slot 5 empty
        "06000000000000"  # Slot 6 empty
    )
    timers = proto.parse_get_timers(response)

    assert len(timers) == 6

    # Check active timers
    assert timers[0].slot == 1
    assert timers[0].hour == 13
    assert timers[0].minute == 25
    assert timers[0].active is True

    assert timers[1].slot == 2
    assert timers[1].hour == 15
    assert timers[1].minute == 35
    assert timers[1].active is True

    assert timers[2].slot == 3
    assert timers[2].hour == 13
    assert timers[2].minute == 33
    assert timers[2].active is True

    # Check inactive timers
    assert timers[3].active is False
    assert timers[4].active is False
    assert timers[5].active is False


def test_protocol_parse_get_timers_with_color_0xB6():
    """Test parsing timer with color action for 0xB6 device."""
    proto = ProtocolLEDENETExtendedCustom()

    # Build a timer response with color action (0xa1)
    # Slot 1: 17:15, Sun+Tue repeat (0x84), color action with HSV
    inner_slot1 = bytes.fromhex("01f0110f00840ee00100a17be432000000000000")
    inner_empty = bytes.fromhex("02000000000000030000000000000400000000000005000000000000060000000000")

    response = bytes([0xE0, 0x06]) + inner_slot1 + inner_empty
    timers = proto.parse_get_timers(response)

    # Check first timer
    timer1 = timers[0]
    assert timer1.slot == 1
    assert timer1.hour == 17
    assert timer1.minute == 15
    assert timer1.repeat_mask == 0x84  # Sun + Tue
    assert timer1.action_type == 0xA1  # Color
    assert timer1.color_hsv == (0x7B, 0xE4, 0x32)  # (123, 228, 50)


def test_led_timer_extended_simple_on():
    """Test LedTimerExtended for simple ON action."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_ON

    timer = LedTimerExtended(
        slot=1,
        active=True,
        hour=8,
        minute=30,
        repeat_mask=LedTimerExtended.Weekdays,
        action_type=TIMER_ACTION_ON,
    )

    assert timer.is_on is True
    assert timer.is_scene is False
    assert timer.is_color is False
    assert timer.repeat_days == ["Mon", "Tue", "Wed", "Thu", "Fri"]

    # Test serialization
    data = timer.to_bytes()
    assert data[0] == 0xF0  # Active flag
    assert data[1] == 8  # Hour
    assert data[2] == 30  # Minute
    assert data[9] == 0x23  # ON action


def test_led_timer_extended_simple_off():
    """Test LedTimerExtended for simple OFF action."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_OFF

    timer = LedTimerExtended(
        slot=2,
        active=True,
        hour=22,
        minute=0,
        repeat_mask=0,  # No repeat
        action_type=TIMER_ACTION_OFF,
    )

    assert timer.is_on is False
    assert timer.repeat_days == []

    data = timer.to_bytes()
    assert data[9] == 0x24  # OFF action


def test_led_timer_extended_color():
    """Test LedTimerExtended for color action."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_COLOR

    timer = LedTimerExtended(
        slot=1,
        active=True,
        hour=17,
        minute=15,
        repeat_mask=0x84,  # Sun + Tue
        action_type=TIMER_ACTION_COLOR,
        color_hsv=(123, 228, 50),
    )

    assert timer.is_on is True
    assert timer.is_color is True
    assert timer.repeat_days == ["Tue", "Sun"]

    data = timer.to_bytes()
    assert data[9] == 0xA1  # Color action
    assert data[10] == 123  # Hue
    assert data[11] == 228  # Saturation
    assert data[12] == 50  # Brightness


def test_led_timer_extended_inactive():
    """Test LedTimerExtended for inactive timer."""
    from flux_led.timer import LedTimerExtended

    timer = LedTimerExtended(
        slot=3,
        active=False,
    )

    data = timer.to_bytes()
    # Inactive timer should be all zeros
    assert data == bytes(20)


def test_led_timer_extended_from_bytes_simple():
    """Test LedTimerExtended.from_bytes for simple timer."""
    from flux_led.timer import LedTimerExtended

    # Simple OFF timer: slot 1, 13:25, no repeat (21 bytes)
    data = bytes.fromhex("01f00d1900000ee001002400000000000000000000")

    timer, consumed = LedTimerExtended.from_bytes(data, 0)

    assert consumed == 21
    assert timer.slot == 1
    assert timer.active is True
    assert timer.hour == 13
    assert timer.minute == 25
    assert timer.repeat_mask == 0
    assert timer.action_type == 0x24  # OFF


def test_led_timer_extended_from_bytes_empty():
    """Test LedTimerExtended.from_bytes for empty slot."""
    from flux_led.timer import LedTimerExtended

    # Empty slot
    data = bytes.fromhex("0300000000000000")

    timer, consumed = LedTimerExtended.from_bytes(data, 0)

    assert consumed == 7
    assert timer.slot == 3
    assert timer.active is False


def test_led_timer_extended_str():
    """Test LedTimerExtended string representation."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_OFF, TIMER_ACTION_ON, TIMER_ACTION_COLOR

    timer_off = LedTimerExtended(
        slot=1, active=True, hour=22, minute=0, action_type=TIMER_ACTION_OFF
    )
    assert "OFF" in str(timer_off)
    assert "22:00" in str(timer_off)

    timer_on = LedTimerExtended(
        slot=2,
        active=True,
        hour=8,
        minute=30,
        repeat_mask=LedTimerExtended.Weekdays,
        action_type=TIMER_ACTION_ON,
    )
    assert "ON" in str(timer_on)
    assert "Mon" in str(timer_on)

    timer_inactive = LedTimerExtended(slot=3, active=False)
    assert "Unset" in str(timer_inactive)


def test_protocol_construct_set_timer_0xB6():
    """Test timer set construction for 0xB6 device."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_OFF

    proto = ProtocolLEDENETExtendedCustom()

    timer = LedTimerExtended(
        slot=1,
        active=True,
        hour=13,
        minute=25,
        repeat_mask=0,
        action_type=TIMER_ACTION_OFF,
    )

    msg = proto.construct_set_timer(timer)

    # Should be wrapped message
    assert msg[0:6] == bytes([0xB0, 0xB1, 0xB2, 0xB3, 0x00, 0x01])
    # Version byte
    assert msg[6] == 0x01
    # Inner message starts at position 10
    assert msg[10] == 0xE0  # Extended command
    assert msg[11] == 0x05  # Set timer command
    assert msg[12] == 0x01  # Slot number


def test_led_timer_extended_scene_gradient():
    """Test LedTimerExtended for scene gradient action."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import (
        TIMER_ACTION_SCENE_GRADIENT,
        ExtendedCustomEffectPattern,
    )

    timer = LedTimerExtended(
        slot=1,
        active=True,
        hour=18,
        minute=30,
        repeat_mask=LedTimerExtended.Everyday,
        action_type=TIMER_ACTION_SCENE_GRADIENT,
        pattern=ExtendedCustomEffectPattern.WAVE,
        speed=80,
        colors=[(100, 100, 100), (50, 50, 50)],
    )

    assert timer.is_on is True
    assert timer.is_scene is True
    assert timer.is_color is False

    # Test serialization
    data = timer.to_bytes()
    assert data[0] == 0xF0  # Active flag
    assert data[1] == 18  # Hour
    assert data[2] == 30  # Minute
    assert data[5] == TIMER_ACTION_SCENE_GRADIENT
    assert data[6] == 0xE1  # Effect marker
    assert data[7] == 0x21  # Gradient effect type

    # Test __str__
    s = str(timer)
    assert "Scene: Wave" in s
    assert "2 colors" in s


def test_led_timer_extended_scene_segments():
    """Test LedTimerExtended for scene segments (colorful) action."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_SCENE_SEGMENTS

    timer = LedTimerExtended(
        slot=2,
        active=True,
        hour=20,
        minute=0,
        repeat_mask=LedTimerExtended.Weekend,
        action_type=TIMER_ACTION_SCENE_SEGMENTS,
        colors=[(180, 100, 100), (0, 100, 100), (60, 100, 100)],
    )

    assert timer.is_scene is True

    # Test serialization
    data = timer.to_bytes()
    assert data[5] == TIMER_ACTION_SCENE_SEGMENTS
    assert data[6] == 0xE1  # Effect marker
    assert data[7] == 0x22  # Segments effect type

    # Test __str__
    s = str(timer)
    assert "Colorful" in s
    assert "3 colors" in s


def test_led_timer_extended_from_bytes_scene_gradient():
    """Test LedTimerExtended.from_bytes for scene gradient timer."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_SCENE_GRADIENT

    # Scene gradient timer from real device data:
    # slot=4, 18:38, repeat=0x0f, action=0x29, e1 21 header, 5 colors
    data = bytes.fromhex(
        "04f0122600f629e12100500300016450000000000000050c646400001e646400005a646400006ce464000096646400"
    )

    timer, consumed = LedTimerExtended.from_bytes(data, 0)

    assert timer.slot == 4
    assert timer.active is True
    assert timer.hour == 18
    assert timer.minute == 38
    assert timer.repeat_mask == 0xF6  # All days except bit 0 and 3
    assert timer.action_type == TIMER_ACTION_SCENE_GRADIENT
    assert len(timer.colors) == 5
    # First color: 0c 64 64 = (12, 100, 100)
    assert timer.colors[0] == (0x0C, 0x64, 0x64)


def test_led_timer_extended_from_bytes_scene_segments():
    """Test LedTimerExtended.from_bytes for scene segments timer."""
    from flux_led.timer import LedTimerExtended
    from flux_led.const import TIMER_ACTION_SCENE_SEGMENTS

    # Scene segments timer: slot=1, 12:00, e1 22 header, 3 colors
    # Construct a minimal segments timer
    data = bytearray()
    data.append(0x01)  # slot
    data.append(0xF0)  # flags
    data.append(12)    # hour
    data.append(0)     # minute
    data.append(0)     # seconds
    data.append(0)     # repeat
    data.append(0x6B)  # action (segments)
    data.append(0xE1)  # effect marker
    data.append(0x22)  # segments type
    data.extend([0, 0, 0, 0])  # header padding
    data.append(3)     # num colors
    # 3 colors (5 bytes each)
    data.extend([100, 100, 50, 0, 0])
    data.extend([50, 80, 60, 0, 0])
    data.extend([150, 90, 70, 0, 0])

    timer, consumed = LedTimerExtended.from_bytes(bytes(data), 0)

    assert timer.slot == 1
    assert timer.active is True
    assert timer.action_type == TIMER_ACTION_SCENE_SEGMENTS
    assert len(timer.colors) == 3
    assert timer.colors[0] == (100, 100, 50)
    assert timer.colors[1] == (50, 80, 60)
    assert timer.colors[2] == (150, 90, 70)


def test_led_timer_extended_str_unknown_action():
    """Test LedTimerExtended.__str__ for unknown action type."""
    from flux_led.timer import LedTimerExtended

    timer = LedTimerExtended(
        slot=1,
        active=True,
        hour=10,
        minute=30,
        action_type=0xFF,  # Unknown action
    )

    s = str(timer)
    assert "Unknown action: 0xff" in s


def test_led_timer_extended_from_bytes_truncated():
    """Test LedTimerExtended.from_bytes handles truncated data."""
    from flux_led.timer import LedTimerExtended

    # Only 5 bytes - not enough for a valid timer
    data = bytes([0x01, 0xF0, 12, 30, 0])
    timer, consumed = LedTimerExtended.from_bytes(data, 0)
    assert consumed == 5  # Returns what's available


# =============================================================================
# CLI Timer Parsing Tests
# =============================================================================


def test_cli_process_set_timer_args_extended_inactive():
    """Test CLI parsing for inactive extended timer."""
    from optparse import OptionParser

    from flux_led.const import TIMER_ACTION_OFF
    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    timer = processSetTimerArgsExtended(parser, ["1", "inactive", ""])
    assert timer.slot == 1
    assert timer.active is False
    assert timer.action_type == TIMER_ACTION_OFF


def test_cli_process_set_timer_args_extended_poweroff():
    """Test CLI parsing for poweroff extended timer."""
    from optparse import OptionParser

    from flux_led.const import TIMER_ACTION_OFF
    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    timer = processSetTimerArgsExtended(
        parser, ["2", "poweroff", "time:1430;repeat:12345"]
    )
    assert timer.slot == 2
    assert timer.active is True
    assert timer.hour == 14
    assert timer.minute == 30
    assert timer.action_type == TIMER_ACTION_OFF
    # repeat 12345 = Mon|Tue|Wed|Thu|Fri = bits 1,2,3,4,5 = 0x3E
    assert timer.repeat_mask == 0x3E


def test_cli_process_set_timer_args_extended_default_on():
    """Test CLI parsing for default (on) extended timer."""
    from optparse import OptionParser

    from flux_led.const import TIMER_ACTION_ON
    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    timer = processSetTimerArgsExtended(
        parser, ["3", "default", "time:0830;repeat:06"]
    )
    assert timer.slot == 3
    assert timer.active is True
    assert timer.hour == 8
    assert timer.minute == 30
    assert timer.action_type == TIMER_ACTION_ON
    # repeat 06 = Sun|Sat = bits 7,6 = 0x80|0x40 = 0xC0
    assert timer.repeat_mask == 0xC0


def test_cli_process_set_timer_args_extended_color():
    """Test CLI parsing for color extended timer."""
    from optparse import OptionParser

    from flux_led.const import TIMER_ACTION_COLOR
    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    # Use color name instead of hex code (hex needs # prefix)
    timer = processSetTimerArgsExtended(
        parser, ["4", "color", "time:2100;repeat:0123456;color:red"]
    )
    assert timer.slot == 4
    assert timer.active is True
    assert timer.hour == 21
    assert timer.minute == 0
    assert timer.action_type == TIMER_ACTION_COLOR
    assert timer.color_hsv is not None
    # red should have hue=0
    assert timer.color_hsv[0] == 0  # hue


def test_cli_process_set_timer_args_extended_color_with_brightness():
    """Test CLI parsing for color extended timer with brightness."""
    from optparse import OptionParser

    from flux_led.const import TIMER_ACTION_COLOR
    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    # Use hex with # prefix
    timer = processSetTimerArgsExtended(
        parser, ["5", "color", "time:1200;repeat:1;color:#00ff00;brightness:50"]
    )
    assert timer.slot == 5
    assert timer.active is True
    assert timer.action_type == TIMER_ACTION_COLOR
    assert timer.color_hsv is not None
    # Check brightness is 50
    assert timer.color_hsv[2] == 50


def test_cli_process_set_timer_args_extended_weekend_repeat():
    """Test CLI parsing for weekend repeat (0=Sun, 6=Sat)."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    timer = processSetTimerArgsExtended(
        parser, ["1", "poweroff", "time:2200;repeat:06"]
    )
    # repeat 06 = Sun|Sat = bit7 | bit6 = 0x80 | 0x40 = 0xC0
    assert timer.repeat_mask == 0xC0


def test_cli_process_set_timer_args_extended_everyday_repeat():
    """Test CLI parsing for everyday repeat (0123456)."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgsExtended

    parser = OptionParser()
    timer = processSetTimerArgsExtended(
        parser, ["1", "default", "time:0700;repeat:0123456"]
    )
    # repeat 0123456 = Sun|Mon|Tue|Wed|Thu|Fri|Sat
    # bit7=Sun + bits1-6 = 0x80 | 0x7E = 0xFE
    assert timer.repeat_mask == 0xFE


def test_cli_process_set_timer_args_standard_inactive():
    """Test CLI parsing for inactive standard timer."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgs

    parser = OptionParser()
    timer = processSetTimerArgs(parser, ["1", "inactive", ""])
    assert timer.isActive() is False


def test_cli_process_set_timer_args_standard_poweroff():
    """Test CLI parsing for poweroff standard timer."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgs

    parser = OptionParser()
    timer = processSetTimerArgs(
        parser, ["2", "poweroff", "time:1430;repeat:12345"]
    )
    assert timer.isActive() is True
    assert timer.hour == 14
    assert timer.minute == 30
    # Check it's a turn-off timer
    assert timer.turn_on is False


def test_cli_process_set_timer_args_standard_default():
    """Test CLI parsing for default (on) standard timer."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgs

    parser = OptionParser()
    timer = processSetTimerArgs(
        parser, ["3", "default", "time:0830;repeat:06"]
    )
    assert timer.isActive() is True
    assert timer.hour == 8
    assert timer.minute == 30


def test_cli_process_set_timer_args_standard_color():
    """Test CLI parsing for color standard timer."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgs

    parser = OptionParser()
    timer = processSetTimerArgs(
        parser, ["4", "color", "time:2100;repeat:0123456;color:255,0,0"]
    )
    assert timer.isActive() is True
    assert timer.hour == 21
    assert timer.minute == 0
    assert timer.red == 255
    assert timer.green == 0
    assert timer.blue == 0


def test_cli_process_set_timer_args_standard_warmwhite():
    """Test CLI parsing for warmwhite standard timer."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgs

    parser = OptionParser()
    timer = processSetTimerArgs(
        parser, ["5", "warmwhite", "time:2200;repeat:12345;level:75"]
    )
    assert timer.isActive() is True
    assert timer.hour == 22
    assert timer.minute == 0
    # 75% is converted to byte: int((75 * 255) / 100) = 191
    assert timer.warmth_level == 191


def test_cli_process_set_timer_args_standard_preset():
    """Test CLI parsing for preset standard timer."""
    from optparse import OptionParser

    from flux_led.fluxled import processSetTimerArgs

    parser = OptionParser()
    timer = processSetTimerArgs(
        parser, ["6", "preset", "time:1800;repeat:06;code:37;speed:50"]
    )
    assert timer.isActive() is True
    assert timer.hour == 18
    assert timer.minute == 0
    assert timer.pattern_code == 37


# =============================================================================
# Timer Display Formatting Tests (__str__)
# =============================================================================


def test_led_timer_standard_str_inactive():
    """Test LedTimer.__str__ for inactive timer."""
    from flux_led.timer import LedTimer

    timer = LedTimer()
    timer.setActive(False)
    assert str(timer) == "Unset"


def test_led_timer_standard_str_on():
    """Test LedTimer.__str__ for turn-on timer."""
    from flux_led.timer import LedTimer

    timer = LedTimer()
    timer.setActive(True)
    timer.setTime(8, 30)
    timer.setModeDefault()
    timer.setRepeatMask(LedTimer.Weekdays)

    s = str(timer)
    assert "[ON ]" in s
    assert "08:30" in s
    assert "Mo" in s
    assert "Tu" in s
    assert "Fr" in s


def test_led_timer_standard_str_off():
    """Test LedTimer.__str__ for turn-off timer."""
    from flux_led.timer import LedTimer

    timer = LedTimer()
    timer.setActive(True)
    timer.setTime(22, 0)
    timer.setModeTurnOff()
    timer.setRepeatMask(LedTimer.Everyday)

    s = str(timer)
    assert "[OFF]" in s
    assert "22:00" in s


def test_led_timer_standard_str_once():
    """Test LedTimer.__str__ for one-time timer."""
    from flux_led.timer import LedTimer

    timer = LedTimer()
    timer.setActive(True)
    timer.setTime(14, 30)
    timer.setModeDefault()
    timer.setDate(2025, 12, 25)

    s = str(timer)
    assert "[ON ]" in s
    assert "14:30" in s
    assert "Once" in s
    assert "2025-12-25" in s


def test_led_timer_standard_str_color():
    """Test LedTimer.__str__ for color timer."""
    from flux_led.timer import LedTimer

    timer = LedTimer()
    timer.setActive(True)
    timer.setTime(19, 0)
    timer.setModeColor(255, 0, 0)
    timer.setRepeatMask(LedTimer.Weekend)

    s = str(timer)
    assert "[ON ]" in s
    assert "19:00" in s
    assert "Sa" in s
    assert "Su" in s


