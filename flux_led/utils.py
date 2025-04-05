from __future__ import annotations

import ast
import colorsys
import contextlib
import datetime
from collections.abc import Iterable
from typing import NamedTuple, cast

import webcolors  # type: ignore[import-untyped]

from .const import MAX_TEMP, MIN_TEMP

MAX_MIN_TEMP_DIFF = MAX_TEMP - MIN_TEMP


class WhiteLevels(NamedTuple):
    """White level for a color."""

    warm_white: int
    cool_white: int


class TemperatureBrightness(NamedTuple):
    """Temperature and brightness for a color."""

    temperature: int
    brightness: int


class utils:
    @staticmethod
    def color_object_to_tuple(
        color: tuple[int, ...] | str,
    ) -> tuple[int, ...] | None:
        # see if it's already a color tuple
        if isinstance(color, tuple) and len(color) in [3, 4, 5]:
            return color

        # can't convert non-string
        if not isinstance(color, str):
            return None
        color = color.strip()

        # try to convert from an english name
        with contextlib.suppress(Exception):
            return cast("tuple[int, int, int]", webcolors.name_to_rgb(color))

        # try to convert an web hex code
        with contextlib.suppress(Exception):
            return cast(
                "tuple[int, int, int]",
                webcolors.hex_to_rgb(webcolors.normalize_hex(color)),
            )

        # try to convert a string RGB tuple
        with contextlib.suppress(Exception):
            val = ast.literal_eval(color)
            if type(val) is not tuple or len(val) not in [3, 4, 5]:
                raise Exception
            return val

        return None

    @staticmethod
    def color_tuple_to_string(rgb: tuple[int, int, int]) -> str:
        # try to convert to an english name
        with contextlib.suppress(Exception):
            return cast("str", webcolors.rgb_to_name(rgb))
        return str(rgb)

    @staticmethod
    def get_color_names_list() -> list[str]:
        return sorted(
            {
                *webcolors.names(webcolors.CSS2),
                *webcolors.names(webcolors.CSS21),
                *webcolors.names(webcolors.CSS3),
                *webcolors.names(webcolors.HTML4),
            }
        )

    @staticmethod
    def date_has_passed(dt: datetime.datetime) -> bool:
        return (dt - datetime.datetime.now()).total_seconds() < 0

    @staticmethod
    def raw_state_to_dec(rx: Iterable[int]) -> str:
        raw_state_str = ""
        for _r in rx:
            raw_state_str += str(_r) + ","
        return raw_state_str

    max_delay = 0x1F

    @staticmethod
    def delayToSpeed(delay: int) -> int:
        # speed is 0-100, delay is 1-31
        # 1st translate delay to 0-30
        delay = delay - 1
        delay = max(0, min(utils.max_delay - 1, delay))
        inv_speed = int((delay * 100) / (utils.max_delay - 1))
        speed = 100 - inv_speed
        return speed

    @staticmethod
    def speedToDelay(speed: int) -> int:
        # speed is 0-100, delay is 1-31
        speed = max(0, min(100, speed))
        inv_speed = 100 - speed
        delay = int((inv_speed * (utils.max_delay - 1)) / 100)
        # translate from 0-30 to 1-31
        delay = delay + 1
        return delay

    @staticmethod
    def byteToPercent(byte: int) -> int:
        return int((max(0, min(255, byte)) * 100) / 255)

    @staticmethod
    def percentToByte(percent: int) -> int:
        return int((max(0, min(100, percent)) * 255) / 100)

    @staticmethod
    def A3WarmWhiteToByte(val: int) -> int:
        return round(((min(228, max(128, val)) - 128) * 255) / 100)


def rgbwc_to_rgbcw(
    rgbwc_data: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    r, g, b, w, c = rgbwc_data
    return r, g, b, c, w


def rgbcw_to_rgbwc(
    rgbcw_data: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    r, g, b, c, w = rgbcw_data
    return r, g, b, w, c


def _adjust_brightness(
    current_brightness: int,
    new_brightness: int,
    color_brightness: int,
    cw_brightness: int,
    ww_brightness: int,
) -> tuple[int, int, int]:
    if new_brightness < current_brightness:
        change_brightness_pct = (
            current_brightness - new_brightness
        ) / current_brightness
        ww_brightness = round(ww_brightness * (1 - change_brightness_pct))
        color_brightness = round(color_brightness * (1 - change_brightness_pct))
        cw_brightness = round(cw_brightness * (1 - change_brightness_pct))
    else:
        change_brightness_pct = (new_brightness - current_brightness) / (
            255 - current_brightness
        )
        ww_brightness = round(
            (255 - ww_brightness) * change_brightness_pct + ww_brightness
        )
        color_brightness = round(
            (255 - color_brightness) * change_brightness_pct + color_brightness
        )
        cw_brightness = round(
            (255 - cw_brightness) * change_brightness_pct + cw_brightness
        )

    return color_brightness, cw_brightness, ww_brightness


def rgbw_brightness(
    rgbw_data: tuple[int, int, int, int],
    brightness: int | None = None,
) -> tuple[int, int, int, int]:
    """Convert rgbw to brightness."""
    original_r, original_g, original_b = rgbw_data[0:3]
    h, s, v = colorsys.rgb_to_hsv(original_r / 255, original_g / 255, original_b / 255)
    color_brightness = round(v * 255)
    ww_brightness = rgbw_data[3]
    current_brightness = round((color_brightness + ww_brightness) / 2)

    if not brightness or brightness == current_brightness:
        return rgbw_data

    if brightness < current_brightness:
        change_brightness_pct = (current_brightness - brightness) / current_brightness
        ww_brightness = round(ww_brightness * (1 - change_brightness_pct))
        color_brightness = round(color_brightness * (1 - change_brightness_pct))

    else:
        change_brightness_pct = (brightness - current_brightness) / (
            255 - current_brightness
        )
        ww_brightness = round(
            (255 - ww_brightness) * change_brightness_pct + ww_brightness
        )
        color_brightness = round(
            (255 - color_brightness) * change_brightness_pct + color_brightness
        )

    r, g, b = colorsys.hsv_to_rgb(h, s, color_brightness / 255)
    return (round(r * 255), round(g * 255), round(b * 255), ww_brightness)


def rgbww_brightness(
    rgbww_data: tuple[int, int, int, int, int],
    brightness: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Convert rgbww to brightness."""
    original_r, original_g, original_b = rgbww_data[0:3]
    h, s, v = colorsys.rgb_to_hsv(original_r / 255, original_g / 255, original_b / 255)
    color_brightness = round(v * 255)
    ww_brightness = rgbww_data[3]
    cw_brightness = rgbww_data[4]
    current_brightness = round((color_brightness + ww_brightness + cw_brightness) / 3)

    if not brightness or brightness == current_brightness:
        return rgbww_data

    color_brightness, cw_brightness, ww_brightness = _adjust_brightness(
        current_brightness, brightness, color_brightness, cw_brightness, ww_brightness
    )
    r, g, b = colorsys.hsv_to_rgb(h, s, color_brightness / 255)
    return (
        round(r * 255),
        round(g * 255),
        round(b * 255),
        ww_brightness,
        cw_brightness,
    )


def rgbcw_brightness(
    rgbcw_data: tuple[int, int, int, int, int],
    brightness: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Convert rgbww to brightness."""
    original_r, original_g, original_b = rgbcw_data[0:3]
    h, s, v = colorsys.rgb_to_hsv(original_r / 255, original_g / 255, original_b / 255)
    color_brightness = round(v * 255)
    cw_brightness = rgbcw_data[3]
    ww_brightness = rgbcw_data[4]
    current_brightness = round((color_brightness + ww_brightness + cw_brightness) / 3)

    if not brightness or brightness == current_brightness:
        return rgbcw_data

    color_brightness, cw_brightness, ww_brightness = _adjust_brightness(
        current_brightness, brightness, color_brightness, cw_brightness, ww_brightness
    )
    r, g, b = colorsys.hsv_to_rgb(h, s, color_brightness / 255)
    return (
        round(r * 255),
        round(g * 255),
        round(b * 255),
        cw_brightness,
        ww_brightness,
    )


def color_temp_to_white_levels(
    temperature: int,
    brightness: float,
    min_temp: int = MIN_TEMP,
    max_temp: int = MAX_TEMP,
) -> WhiteLevels:
    # Assume output temperature of between 2700 and 6500 Kelvin, and scale
    # the warm and cold LEDs linearly to provide that
    temperature = min(max(min_temp, temperature), max_temp)
    if not (0 <= brightness <= 255):
        raise ValueError(
            f"Brightness of {brightness} is not valid and must be between 0 and 255"
        )
    brightness = round(brightness / 255, 2)
    warm = ((max_temp - temperature) / (max_temp - min_temp)) * brightness
    cold = brightness - warm
    return WhiteLevels(round(255 * warm), round(255 * cold))


def scaled_color_temp_to_white_levels(
    temperature: int, brightness: float
) -> WhiteLevels:
    # Assume output temperature of between 0 and 100, and scale
    # the warm and cold LEDs linearly to provide that
    if not (0 <= temperature <= 100):
        raise ValueError(
            f"Temperature of {temperature} is not valid and must be between {0} and {100}"
        )
    if not (0 <= brightness <= 100):
        raise ValueError(
            f"Brightness of {brightness} is not valid and must be between 0 and 100"
        )
    brightness = round(brightness / 100, 2)
    warm = ((100 - temperature) / 100) * brightness
    cold = brightness - warm
    return WhiteLevels(round(255 * warm), round(255 * cold))


def white_levels_to_color_temp(
    warm_white: int, cool_white: int, min_temp: int = MIN_TEMP, max_temp: int = MAX_TEMP
) -> TemperatureBrightness:
    if not (0 <= warm_white <= 255):
        raise ValueError(
            f"Warm White of {warm_white} is not valid and must be between 0 and 255"
        )
    if not (0 <= cool_white <= 255):
        raise ValueError(
            f"Cool White of {cool_white} is not valid and must be between 0 and 255"
        )
    warm = warm_white / 255
    cold = cool_white / 255
    brightness = warm + cold
    if brightness == 0:
        temperature: float = min_temp
    else:
        temperature = ((cold / brightness) * (max_temp - min_temp)) + min_temp
    return TemperatureBrightness(round(temperature), min(255, round(brightness * 255)))


def white_levels_to_scaled_color_temp(
    warm_white: int, cool_white: int
) -> TemperatureBrightness:
    if not (0 <= warm_white <= 255):
        raise ValueError(
            f"Warm White of {warm_white} is not valid and must be between 0 and 255"
        )
    if not (0 <= cool_white <= 255):
        raise ValueError(
            f"Cool White of {cool_white} is not valid and must be between 0 and 255"
        )
    warm = warm_white / 255
    cold = cool_white / 255
    brightness = warm + cold
    if brightness == 0:
        temperature: float = 0
    else:
        temperature = (cold / brightness) * 100
    return TemperatureBrightness(round(temperature), min(100, round(brightness * 100)))
