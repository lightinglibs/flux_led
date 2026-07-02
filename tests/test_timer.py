"""Unit tests for flux_led.timer."""

from __future__ import annotations

from flux_led.timer import BuiltInTimer, LedTimer


def _new_timer(length: int = 14) -> LedTimer:
    timer = LedTimer(length=length)
    timer.setActive(True)
    timer.setTime(6, 30)
    timer.setDate(2024, 1, 15)
    return timer


def test_setModeSunrise_sets_mode_and_pattern() -> None:
    timer = _new_timer()
    timer.setModeSunrise(startBrightness=10, endBrightness=90, duration=30)

    assert timer.mode == "Sunrise"
    assert timer.pattern_code == BuiltInTimer.sunrise
    assert timer.turn_on is True
    assert timer.duration == 30


def test_setModeSunset_sets_mode_to_sunset() -> None:
    timer = _new_timer()
    timer.setModeSunset(startBrightness=90, endBrightness=10, duration=45)

    assert timer.mode == "Sunset"
    assert timer.pattern_code == BuiltInTimer.sunset
    assert timer.turn_on is True
    assert timer.duration == 45


def test_sunrise_timer_roundtrip_preserves_mode() -> None:
    """A sunrise timer should still be classified as sunrise after fromBytes."""
    timer = _new_timer()
    timer.setModeSunrise(startBrightness=20, endBrightness=80, duration=60)

    raw = timer.toBytes()
    restored = LedTimer(raw)

    assert restored.mode == "Sunrise"
    assert restored.pattern_code == BuiltInTimer.sunrise
    assert restored.duration == 60


def test_sunset_timer_roundtrip_preserves_mode() -> None:
    timer = _new_timer()
    timer.setModeSunset(startBrightness=80, endBrightness=10, duration=15)

    raw = timer.toBytes()
    restored = LedTimer(raw)

    assert restored.mode == "Sunset"
    assert restored.pattern_code == BuiltInTimer.sunset


def test_warm_white_roundtrip_still_ww() -> None:
    """Warm white timer must still parse back as ww (regression guard)."""
    timer = _new_timer()
    timer.setModeWarmWhite(level=50)

    restored = LedTimer(timer.toBytes())

    assert restored.mode == "ww"
    assert restored.pattern_code == 0x61


def test_color_with_zero_warmth_stays_color() -> None:
    timer = _new_timer()
    timer.setModeColor(255, 128, 0)

    restored = LedTimer(timer.toBytes())

    assert restored.mode == "color"
    assert (restored.red, restored.green, restored.blue) == (255, 128, 0)


def test_socket_timer_turn_on_byte_roundtrip() -> None:
    """Socket timers (length 12) must serialize byte[8] = 0x23 when on."""
    raw_on = b"\xf0\x00\x00\x00\x11\x2f\x00\xfe\x23\x00\x00\x00"
    timer = LedTimer(raw_on)
    assert timer.turn_on is True
    assert timer.toBytes() == raw_on


def test_socket_timer_turn_off_byte_roundtrip() -> None:
    """Socket timers (length 12) must serialize byte[8] = 0x24 when off."""
    raw_off = b"\xf0\x00\x00\x00\x11\x2f\x00\xfe\x24\x00\x00\x00"
    timer = LedTimer(raw_off)
    assert timer.turn_on is False
    # Inactive entries short-circuit; active off entries must keep 0x24.
    assert timer.active is True
    assert timer.toBytes() == raw_off


def test_inactive_timer_roundtrip_returns_zero_filled() -> None:
    timer = LedTimer(length=14)  # constructed inactive by default
    raw = timer.toBytes()
    assert raw[0] == 0x0F
    assert raw == bytearray(14).replace(b"\x00", b"\x00", 1).__class__(
        [0x0F] + [0x00] * 13
    )


def test_setModeSunset_brightness_does_not_clobber_mode_on_parse() -> None:
    """Even when warmth_level byte is nonzero, sunset stays classified."""
    timer = _new_timer()
    timer.setModeSunset(startBrightness=100, endBrightness=100, duration=10)
    assert timer.warmth_level != 0  # written by setModeSunset
    restored = LedTimer(timer.toBytes())
    assert restored.mode == "Sunset"


def test_BuiltInTimer_valtostr() -> None:
    assert BuiltInTimer.valtostr(BuiltInTimer.sunrise) == "Sunrise"
    assert BuiltInTimer.valtostr(BuiltInTimer.sunset) == "Sunset"
