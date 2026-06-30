"""Tests for the synchronous BulbScanner (flux_led/scanner.py)."""

from unittest.mock import MagicMock, patch

from flux_led.scanner import (
    MESSAGE_SEND_INTERLEAVE_DELAY,
    BulbScanner,
    FluxLEDDiscovery,
)


def _discovery(ipaddr: str, id: str) -> FluxLEDDiscovery:
    return FluxLEDDiscovery(
        ipaddr=ipaddr,
        id=id,
        model="AK001-ZJ2145",
        model_num=8,
        version_num=21,
        firmware_date=None,
        model_info="ZG-BL",
        model_description="Controller RGB with MIC",
        remote_access_enabled=None,
        remote_access_host=None,
        remote_access_port=None,
    )


def test_get_bulb_info_by_id_returns_match() -> None:
    scanner = BulbScanner()
    scanner._discoveries = {
        "10.0.0.1": _discovery("10.0.0.1", "AAAAAAAAAAAA"),
        "10.0.0.2": _discovery("10.0.0.2", "BBBBBBBBBBBB"),
    }
    assert scanner.getBulbInfoByID("BBBBBBBBBBBB")["ipaddr"] == "10.0.0.2"


def test_get_bulb_info_by_id_missing_returns_none() -> None:
    """A miss must return None, not the last bulb iterated."""
    scanner = BulbScanner()
    scanner._discoveries = {
        "10.0.0.1": _discovery("10.0.0.1", "AAAAAAAAAAAA"),
        "10.0.0.2": _discovery("10.0.0.2", "BBBBBBBBBBBB"),
    }
    assert scanner.getBulbInfoByID("CCCCCCCCCCCC") is None


def test_get_bulb_info_by_id_empty_returns_none() -> None:
    """No discoveries must return None, not raise UnboundLocalError."""
    scanner = BulbScanner()
    assert scanner.getBulbInfoByID("AAAAAAAAAAAA") is None


@patch("flux_led.scanner.time.sleep")
def test_send_messages_sleeps_between_only(mock_sleep: MagicMock) -> None:
    """N messages -> N-1 inter-message delays, none after the last."""
    scanner = BulbScanner()
    scanner._send_message = MagicMock()
    messages = [b"a", b"b", b"c"]

    scanner._send_messages(messages, MagicMock(), ("10.0.0.255", 48899))

    assert scanner._send_message.call_count == 3
    assert mock_sleep.call_count == 2
    mock_sleep.assert_called_with(MESSAGE_SEND_INTERLEAVE_DELAY)


@patch("flux_led.scanner.time.sleep")
def test_send_messages_single_no_sleep(mock_sleep: MagicMock) -> None:
    """A single message must not trigger any delay."""
    scanner = BulbScanner()
    scanner._send_message = MagicMock()

    scanner._send_messages([b"only"], MagicMock(), ("10.0.0.255", 48899))

    assert scanner._send_message.call_count == 1
    mock_sleep.assert_not_called()
