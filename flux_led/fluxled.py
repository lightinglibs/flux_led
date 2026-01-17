#!/usr/bin/env python
"""
This is a utility for controlling stand-alone Flux WiFi LED light bulbs.

The protocol was reverse-engineered by studying packet captures between a
bulb and the controlling "Magic Home" mobile app.  The code here dealing

with the network protocol is littered with magic numbers, and ain't so pretty.
But it does seem to work!

So far most of the functionality of the apps is available here via the CLI
and/or programmatically.

The classes in this project could very easily be used as an API, and incorporated into a GUI app written
in PyQt, Kivy, or some other framework.

##### Available:
* Discovering bulbs on LAN
* Turning on/off bulb
* Get state information
* Setting "warm white" mode
* Setting single color mode
* Setting preset pattern mode
* Setting custom pattern mode
* Reading timers
* Setting timers

##### Some missing pieces:
* Initial administration to set up WiFi SSID and passphrase/key.
* Remote access administration
* Music-relating pulsing. This feature isn't so impressive on the Magic Home app,
and looks like it might be a bit of work.

##### Cool feature:
* Specify colors with names or web hex values.  Requires that python "webcolors"
package is installed.  (Easily done via pip, easy_install, or apt-get, etc.)
See the following for valid color names: http://www.w3schools.com/html/html_colornames.asp

"""

from __future__ import annotations

import asyncio
import datetime
import logging
import sys
from optparse import OptionGroup, OptionParser, Values
from typing import Any, cast

from .aio import AIOWifiLedBulb
from .aioscanner import AIOBulbScanner
from .const import (
    ATTR_ID,
    ATTR_IPADDR,
    TIMER_ACTION_COLOR,
    TIMER_ACTION_OFF,
    TIMER_ACTION_ON,
    ExtendedCustomEffectDirection,
    ExtendedCustomEffectPattern,
)
from .pattern import PresetPattern
from .protocol import ProtocolLEDENETExtendedCustom
from .scanner import FluxLEDDiscovery
from .timer import LedTimer, LedTimerExtended
from .utils import utils

_LOGGER = logging.getLogger(__name__)


# =======================================================================
def showUsageExamples() -> None:
    example_text = """
Examples:

Scan network:
    %prog% -s

Scan network and show info about all:
    %prog% -sSti

Turn on:
    %prog% 192.168.1.100 --on
    %prog% 192.168.1.100 -192.168.1.101 -1

Turn on all bulbs on LAN:
    %prog% -sS --on

Turn off:
    %prog% 192.168.1.100 --off
    %prog% 192.168.1.100 --0
    %prog% -sS --off

Set warm white, 75%
    %prog% 192.168.1.100 -w 75

Set cold white, 55%
    %prog% 192.168.1.100 -d 55

Set CCT, 3500 85%
    %prog% 192.168.1.100 -k 3500 85

Set fixed color red :
    %prog% 192.168.1.100 -c Red
    %prog% 192.168.1.100 -c 255,0,0
    %prog% 192.168.1.100 -c "#FF0000"

Set RGBW 25 100 200 50:
    %prog% 192.168.1.100 -c 25,100,200,50

Set RGBWW 25 100 200 50 30:
    %prog% 192.168.1.100 -c 25,100,200,50,30

Set preset pattern #35 with 40% speed:
    %prog% 192.168.1.100 -p 35 40

Set custom pattern 25% speed, red/green/blue, gradual change:
    %prog% 192.168.1.100 -C gradual 25 "red green (0,0,255)"

Set extended custom effect (0xB6 devices) - wave pattern, 80% speed:
    %prog% 192.168.1.100 -C wave 80 "red green blue"

Set extended effect with density, direction, and color change options:
    %prog% 192.168.1.100 -C meteor 50 "purple orange" --density 75 --direction r2l --colorchange

Set custom segment colors (0xB6 devices) - set each segment individually:
    %prog% 192.168.1.100 -C segments 0 "red green blue off off red green blue"

Sync all bulb's clocks with this computer's:
    %prog% -sS --setclock

Set timer #1 to turn on red at 5:30pm on weekdays:
    %prog% 192.168.1.100 -T 1 color "time:1730;repeat:12345;color:red"

Deactivate timer #4:
    %prog% 192.168.1.100 -T 4 inactive ""

Use --timerhelp for more details on setting timers
    """

    print(example_text.replace("%prog%", sys.argv[0]))


def showTimerHelp() -> None:
    timerhelp_text = """
    There are 6 timers available for each bulb.

Mode Details:
    inactive:   timer is inactive and unused
    poweroff:   turns off the light
    default:    turns on the light in default mode
    color:      turns on the light with specified color
    preset:     turns on the light with specified preset and speed
    warmwhite:  turns on the light with warm white at specified brightness

Settings available for each mode:
    Timer Mode | Settings
    --------------------------------------------
    inactive:   [none]
    poweroff:   time, (repeat | date)
    default:    time, (repeat | date)
    color:      time, (repeat | date), color
    preset:     time, (repeat | date), code, speed
    warmwhite:  time, (repeat | date), level
    sunrise:    time, (repeat | date), startBrightness, endBrightness, duration
    sunset:     time, (repeat | date), startBrightness, endBrightness, duration

Setting Details:

    time: 4 digit string with zeros, no colons
        e.g:
        "1000"  - for 10:00am
        "2312"  - for 11:23pm
        "0315"  - for 3:15am

    repeat: Days of the week that the timer should repeat
            (Mutually exclusive with date)
            0=Sun, 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri, 6=Sat
        e.g:
        "0123456"  - everyday
        "06"       - weekends
        "12345"    - weekdays
        "2"        - only Tuesday

    date: Date that the one-time timer should fire
            (Mutually exclusive with repeat)
        e.g:
        "2015-09-13"
        "2016-12-03"

    color: Color name, hex code, or rgb triple

    level: Level of the warm while light (0-100)

    code:  Code of the preset pattern (use -l to list them)

    speed: Speed of the preset pattern transitions (0-100)

    startBrightness: starting brightness of warmlight (0-100)

    endBrightness: ending brightness of warmlight (0-100)

    duration: transition time in minutes

Example setting strings:
    "time:2130;repeat:0123456"
    "time:2130;date:2015-08-11"
    "time:1245;repeat:12345;color:123,345,23"
    "time:1245;repeat:12345;color:green"
    "time:1245;repeat:06;code:50;speed:30"
    "time:0345;date:2015-08-11;level:100"
    """

    print(timerhelp_text)


def processSetTimerArgs(parser: OptionParser, args: Any) -> LedTimer:
    mode = args[1]
    num = args[0]
    settings = args[2]

    if not num.isdigit() or int(num) > 6 or int(num) < 1:
        parser.error("Timer number must be between 1 and 6")

    # create a dict from the settings string
    settings_list = settings.split(";")
    settings_dict = {}
    for s in settings_list:
        pair = s.split(":")
        key = pair[0].strip().lower()
        val = ""
        if len(pair) > 1:
            val = pair[1].strip().lower()
        settings_dict[key] = val

    keys = list(settings_dict.keys())
    timer = LedTimer()

    if mode == "inactive":
        # no setting needed
        timer.setActive(False)

    elif mode in [
        "poweroff",
        "default",
        "color",
        "preset",
        "warmwhite",
        "sunrise",
        "sunset",
    ]:
        timer.setActive(True)

        if "time" not in keys:
            parser.error(f"This mode needs a time: {mode}")
        if "repeat" in keys and "date" in keys:
            parser.error(f"This mode only a repeat or a date, not both: {mode}")

        # validate time format
        if len(settings_dict["time"]) != 4 or not settings_dict["time"].isdigit():
            parser.error("time must be a 4 digits")
        hour = int(settings_dict["time"][0:2:])
        minute = int(settings_dict["time"][2:4:])
        if hour > 23:
            parser.error("timer hour can't be greater than 23")
        if minute > 59:
            parser.error("timer minute can't be greater than 59")

        timer.setTime(hour, minute)

        # validate date format
        if "repeat" not in keys and "date" not in keys:
            # Generate date for next occurance of time
            print("No time or repeat given. Defaulting to next occurance of time")
            now = datetime.datetime.now()
            dt = now.replace(hour=hour, minute=minute)
            if utils.date_has_passed(dt):
                dt = dt + datetime.timedelta(days=1)
            # settings_dict["date"] = date
            timer.setDate(dt.year, dt.month, dt.day)
        elif "date" in keys:
            try:
                dt = datetime.datetime.strptime(settings_dict["date"], "%Y-%m-%d")
                timer.setDate(dt.year, dt.month, dt.day)
            except ValueError:
                parser.error("date is not properly formatted: YYYY-MM-DD")

        # validate repeat format
        if "repeat" in keys:
            if len(settings_dict["repeat"]) == 0:
                parser.error("Must specify days to repeat")
            days = set()
            for c in list(settings_dict["repeat"]):
                if c not in ["0", "1", "2", "3", "4", "5", "6"]:
                    parser.error("repeat can only contain digits 0-6")
                days.add(int(c))

            repeat = 0
            if 0 in days:
                repeat |= LedTimer.Su
            if 1 in days:
                repeat |= LedTimer.Mo
            if 2 in days:
                repeat |= LedTimer.Tu
            if 3 in days:
                repeat |= LedTimer.We
            if 4 in days:
                repeat |= LedTimer.Th
            if 5 in days:
                repeat |= LedTimer.Fr
            if 6 in days:
                repeat |= LedTimer.Sa
            timer.setRepeatMask(repeat)

        if mode == "default":
            timer.setModeDefault()

        if mode == "poweroff":
            timer.setModeTurnOff()

        if mode == "color":
            if "color" not in keys:
                parser.error("color mode needs a color setting")
            # validate color val
            c = utils.color_object_to_tuple(settings_dict["color"])  # type: ignore
            if c is None:
                parser.error("Invalid color value: {}".format(settings_dict["color"]))
            assert c is not None
            timer.setModeColor(c[0], c[1], c[2])  # type: ignore

        if mode == "preset":
            if "code" not in keys:
                parser.error(f"preset mode needs a code: {mode}")
            if "speed" not in keys:
                parser.error(f"preset mode needs a speed: {mode}")
            code = settings_dict["code"]
            speed = settings_dict["speed"]
            if not speed.isdigit() or int(speed) > 100:
                parser.error("preset speed must be a percentage (0-100)")
            if not code.isdigit() or not PresetPattern.valid(int(code)):
                parser.error("preset code must be in valid range")
            timer.setModePresetPattern(int(code), int(speed))

        if mode == "warmwhite":
            if "level" not in keys:
                parser.error(f"warmwhite mode needs a level: {mode}")
            level = settings_dict["level"]
            if not level.isdigit() or int(level) > 100:
                parser.error("warmwhite level must be a percentage (0-100)")
            timer.setModeWarmWhite(int(level))

        if mode == "sunrise" or mode == "sunset":
            if "startbrightness" not in keys:
                parser.error(f"{mode} mode needs a startBrightness (0% -> 100%)")
            startBrightness = int(settings_dict["startbrightness"])

            if "endbrightness" not in keys:
                parser.error(f"{mode} mode needs an endBrightness (0% -> 100%)")
            endBrightness = int(settings_dict["endbrightness"])

            if "duration" not in keys:
                parser.error(f"{mode} mode needs a duration (minutes)")
            duration = int(settings_dict["duration"])

            if mode == "sunrise":
                timer.setModeSunrise(startBrightness, endBrightness, duration)

            elif mode == "sunset":
                timer.setModeSunset(startBrightness, endBrightness, duration)

    else:
        parser.error(f"Not a valid timer mode: {mode}")

    return timer


def processSetTimerArgsExtended(
    parser: OptionParser, args: list[str]
) -> LedTimerExtended:
    """Process timer args for 0xB6 extended timer format.

    Supported modes: inactive, default (on), poweroff, color
    Settings format: mode;time:HHMM;repeat:0123456;color:RRGGBB
    """
    num = args[0]
    mode = args[1].lower() if len(args) > 1 else "inactive"
    settings = args[2] if len(args) > 2 else ""

    if not num.isdigit() or int(num) > 6 or int(num) < 1:
        parser.error("Timer number must be between 1 and 6")

    slot = int(num)

    # parse settings
    settings_dict: dict[str, str] = {}
    if settings:
        for s in settings.split(";"):
            pair = s.split(":")
            key = pair[0].strip().lower()
            val = pair[1].strip().lower() if len(pair) > 1 else ""
            settings_dict[key] = val

    if mode == "inactive":
        return LedTimerExtended(
            slot=slot,
            active=False,
            hour=0,
            minute=0,
            repeat_mask=0,
            action_type=TIMER_ACTION_OFF,
        )

    if mode not in ["poweroff", "default", "color"]:
        parser.error(
            f"Not a valid timer mode for this device: {mode}. "
            "Supported: inactive, default, poweroff, color"
        )

    # validate time
    if "time" not in settings_dict:
        parser.error(f"This mode needs a time: {mode}")
    time_str = settings_dict["time"]
    if len(time_str) != 4 or not time_str.isdigit():
        parser.error("time must be 4 digits (HHMM)")
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    if hour > 23:
        parser.error("timer hour can't be greater than 23")
    if minute > 59:
        parser.error("timer minute can't be greater than 59")

    # parse repeat mask
    # Format: repeat:0123456 where 0=Sun, 1=Mon, ..., 6=Sat
    repeat_mask = 0
    if "repeat" in settings_dict:
        repeat_str = settings_dict["repeat"]
        for c in repeat_str:
            if c not in "0123456":
                parser.error("repeat can only contain digits 0-6")
            day = int(c)
            # Map: 0=Sun->bit7, 1=Mon->bit1, 2=Tue->bit2, ..., 6=Sat->bit6
            if day == 0:
                repeat_mask |= 0x80  # Sunday = bit 7
            else:
                repeat_mask |= 1 << day  # Mon=bit1, Tue=bit2, etc.

    # determine action type and build timer
    if mode == "poweroff":
        return LedTimerExtended(
            slot=slot,
            active=True,
            hour=hour,
            minute=minute,
            repeat_mask=repeat_mask,
            action_type=TIMER_ACTION_OFF,
        )

    if mode == "default":
        return LedTimerExtended(
            slot=slot,
            active=True,
            hour=hour,
            minute=minute,
            repeat_mask=repeat_mask,
            action_type=TIMER_ACTION_ON,
        )

    if mode == "color":
        if "color" not in settings_dict:
            parser.error("color mode needs a color setting")
        color_val = settings_dict["color"]
        # If it looks like hex without # prefix (6 hex chars), add the prefix
        if len(color_val) == 6 and all(c in "0123456789abcdef" for c in color_val):
            color_val = "#" + color_val
        rgb = utils.color_object_to_tuple(color_val)
        if rgb is None:
            parser.error(f"Invalid color value: {color_val}")
        assert rgb is not None
        # Convert RGB to HSV (0-255 scale)
        r, g, b = rgb[0], rgb[1], rgb[2]
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        if max_c == 0:
            hsv_s = 0
            hsv_h = 0
        else:
            hsv_s = int((max_c - min_c) * 255 / max_c)
            delta = max_c - min_c
            if delta == 0:
                hsv_h = 0
            elif max_c == r:
                hsv_h = int(((g - b) / delta) * 255 / 6) % 256
            elif max_c == g:
                hsv_h = int((2.0 + (b - r) / delta) * 255 / 6) % 256
            else:
                hsv_h = int((4.0 + (r - g) / delta) * 255 / 6) % 256

        # brightness from settings or default to 100%
        brightness = 100
        if "brightness" in settings_dict:
            brightness = int(settings_dict["brightness"])
            brightness = min(brightness, 100)

        return LedTimerExtended(
            slot=slot,
            active=True,
            hour=hour,
            minute=minute,
            repeat_mask=repeat_mask,
            action_type=TIMER_ACTION_COLOR,
            color_hsv=(hsv_h, hsv_s, brightness),
        )

    # Should never reach here due to earlier validation
    parser.error(f"Not a valid timer mode: {mode}")
    raise SystemExit(1)  # For type checker


def processCustomArgs(
    parser: OptionParser,
    args: Any,
    density: int = 50,
    direction: str = "l2r",
    colorchange: bool = False,
) -> dict[str, Any] | None:
    """Process custom pattern arguments.

    Supports both standard patterns (jump, gradual, strobe) and extended
    patterns (wave, meteor, etc.) for devices that support them.

    Returns a dict with:
        - mode: "standard", "extended", or "segments"
        - For standard: type, speed, colors
        - For extended: pattern_id, speed, density, colors, direction, option
        - For segments: colors (list of up to 20 colors)
    """
    # Build mapping of extended pattern names to IDs
    extended_pattern_names = {
        p.name.lower().replace("_", " "): p.value for p in ExtendedCustomEffectPattern
    }
    # Also add underscore versions and single word versions
    for p in ExtendedCustomEffectPattern:
        extended_pattern_names[p.name.lower()] = p.value

    pattern_type = args[0].lower()
    speed = int(args[1])

    # Check if this is a standard pattern
    if pattern_type in ["gradual", "jump", "strobe"]:
        # Standard custom pattern
        try:
            color_list_str = args[2].strip()
            str_list = color_list_str.split(" ")
            color_list: list[tuple[int, ...]] = []
            for s in str_list:
                c = utils.color_object_to_tuple(s)
                if c is not None:
                    color_list.append(c)
                else:
                    raise ValueError(f"Invalid color: {s}")
        except Exception:
            parser.error(
                "COLORLIST isn't formatted right. It should be a space separated list "
                "of RGB tuples, color names or web hex values"
            )
            return None

        return {
            "mode": "standard",
            "type": pattern_type,
            "speed": speed,
            "colors": color_list,
        }

    # Check if this is "segments" mode
    if pattern_type == "segments":
        try:
            color_list_str = args[2].strip()
            str_list = color_list_str.split(" ")
            segment_list: list[tuple[int, int, int] | None] = []
            for s in str_list:
                s = s.strip().lower()
                if not s:
                    continue
                if s == "off":
                    segment_list.append(None)
                else:
                    c = utils.color_object_to_tuple(s)
                    if c is not None and len(c) >= 3:
                        segment_list.append((c[0], c[1], c[2]))
                    else:
                        raise ValueError(f"Invalid color: {s}")
            if len(segment_list) == 0:
                parser.error("At least one segment color is required")
                return None
            if len(segment_list) > 20:
                print("Warning: More than 20 segments provided, truncating to 20")
                segment_list = segment_list[:20]
        except Exception as e:
            parser.error(
                f"COLORLIST isn't formatted right: {e}. It should be a space-separated "
                "list of color names, hex values, RGB triples, or 'off'"
            )
            return None

        return {
            "mode": "segments",
            "colors": segment_list,
        }

    # Check if this is an extended pattern name
    if pattern_type in extended_pattern_names:
        pattern_id = extended_pattern_names[pattern_type]

        # Parse color list
        try:
            color_list_str = args[2].strip()
            str_list = color_list_str.split(" ")
            ext_color_list: list[tuple[int, int, int]] = []
            for s in str_list:
                c = utils.color_object_to_tuple(s)
                if c is not None and len(c) >= 3:
                    ext_color_list.append((c[0], c[1], c[2]))
                else:
                    raise ValueError(f"Invalid color: {s}")
            if len(ext_color_list) == 0:
                parser.error("At least one color is required")
                return None
            if len(ext_color_list) > 8:
                print("Warning: More than 8 colors provided, truncating to 8")
                ext_color_list = ext_color_list[:8]
        except Exception as e:
            parser.error(
                f"COLORLIST isn't formatted right: {e}. It should be a space-separated "
                "list of color names, hex values, or RGB triples"
            )
            return None

        # Parse direction
        if direction.lower() in ("l2r", "left", "ltr"):
            dir_value = ExtendedCustomEffectDirection.LEFT_TO_RIGHT.value
        elif direction.lower() in ("r2l", "right", "rtl"):
            dir_value = ExtendedCustomEffectDirection.RIGHT_TO_LEFT.value
        else:
            parser.error(f"Invalid direction: {direction}. Use l2r or r2l")
            return None

        # Option value
        option_value = 0x01 if colorchange else 0x00

        return {
            "mode": "extended",
            "pattern_id": pattern_id,
            "speed": speed,
            "density": density,
            "colors": ext_color_list,
            "direction": dir_value,
            "option": option_value,
        }

    # Unknown pattern type - show valid options
    parser.error(
        f"Unknown pattern type: {pattern_type}. Valid types include: "
        f"jump, gradual, strobe, segments, wave, meteor, breathe, etc. "
        f"Use --listpresets for the full list."
    )
    return None


def parseArgs() -> tuple[Values, Any]:
    parser = OptionParser()

    parser.description = "A utility to control Flux WiFi LED Bulbs. "
    # parser.description += ""
    # parser.description += "."
    power_group = OptionGroup(parser, "Power options (mutually exclusive)")
    mode_group = OptionGroup(parser, "Mode options (mutually exclusive)")
    info_group = OptionGroup(parser, "Program help and information option")
    other_group = OptionGroup(parser, "Other options")

    parser.add_option_group(info_group)
    info_group.add_option(
        "--debug",
        action="store_true",
        dest="debug",
        default=False,
        help="Enable debug logging",
    )
    info_group.add_option(
        "-e",
        "--examples",
        action="store_true",
        dest="showexamples",
        default=False,
        help="Show usage examples",
    )
    info_group.add_option(
        "",
        "--timerhelp",
        action="store_true",
        dest="timerhelp",
        default=False,
        help="Show detailed help for setting timers",
    )
    info_group.add_option(
        "-l",
        "--listpresets",
        action="store_true",
        dest="listpresets",
        default=False,
        help="List preset codes (including extended patterns for 0xB6 devices)",
    )
    info_group.add_option(
        "--listcolors",
        action="store_true",
        dest="listcolors",
        default=False,
        help="List color names",
    )

    parser.add_option(
        "-s",
        "--scan",
        action="store_true",
        dest="scan",
        default=False,
        help="Search for bulbs on local network",
    )
    parser.add_option(
        "-S",
        "--scanresults",
        action="store_true",
        dest="scanresults",
        default=False,
        help="Operate on scan results instead of arg list",
    )
    power_group.add_option(
        "-1",
        "--on",
        action="store_true",
        dest="on",
        default=False,
        help="Turn on specified bulb(s)",
    )
    power_group.add_option(
        "-0",
        "--off",
        action="store_true",
        dest="off",
        default=False,
        help="Turn off specified bulb(s)",
    )
    parser.add_option_group(power_group)

    mode_group.add_option(
        "-c",
        "--color",
        dest="color",
        default=None,
        help="""For setting a single color mode.  Can be either color name, web hex, or comma-separated RGB triple.
        For setting an RGBW can be a comma-seperated RGBW list
        For setting an RGBWW can be a comma-seperated RGBWW list""",
        metavar="COLOR",
    )
    mode_group.add_option(
        "-w",
        "--warmwhite",
        dest="ww",
        default=None,
        help="Set warm white mode (LEVELWW is percent)",
        metavar="LEVELWW",
        type="int",
    )
    mode_group.add_option(
        "-d",
        "--coldwhite",
        dest="cw",
        default=None,
        help="Set cold white mode (LEVELCW is percent)",
        metavar="LEVELCW",
        type="int",
    )
    mode_group.add_option(
        "-k",
        "--CCT",
        dest="cct",
        default=None,
        help="Temperture and brightness (CCT Kelvin, brightness percent)",
        metavar="LEVELCCT",
        type="int",
        nargs=2,
    )
    mode_group.add_option(
        "-p",
        "--preset",
        dest="preset",
        default=None,
        help="Set preset pattern mode (SPEED is percent)",
        metavar="CODE SPEED",
        type="int",
        nargs=2,
    )
    mode_group.add_option(
        "-C",
        "--custom",
        dest="custom",
        metavar="TYPE SPEED COLORLIST",
        default=None,
        nargs=3,
        help="Set custom pattern mode. "
        + "TYPE: jump, gradual, strobe (standard), or extended pattern names "
        + "(wave, meteor, breathe, etc. for 0xB6 devices), or 'segments' for static colors. "
        + "SPEED is percent (0-100). "
        + "COLORLIST is a space-separated list of color names, hex values, or RGB triples. "
        + "Use --density, --direction, --colorchange for extended pattern options.",
    )
    parser.add_option_group(mode_group)

    other_group.add_option(
        "--density",
        dest="density",
        default=50,
        type="int",
        metavar="DENSITY",
        help="Pattern density 0-100 for extended effects (default: 50)",
    )
    other_group.add_option(
        "--direction",
        dest="direction",
        default="l2r",
        metavar="DIR",
        help="Direction for extended effect: l2r (left to right) or r2l (right to left). Default: l2r",
    )
    other_group.add_option(
        "--colorchange",
        action="store_true",
        dest="colorchange",
        default=False,
        help="Enable color change option for extended effect",
    )

    parser.add_option(
        "-i",
        "--info",
        action="store_true",
        dest="info",
        default=False,
        help="Info about bulb(s) state",
    )
    parser.add_option(
        "",
        "--getclock",
        action="store_true",
        dest="getclock",
        default=False,
        help="Get clock",
    )
    parser.add_option(
        "",
        "--setclock",
        action="store_true",
        dest="setclock",
        default=False,
        help="Set clock to same as current time on this computer",
    )
    parser.add_option(
        "-t",
        "--timers",
        action="store_true",
        dest="showtimers",
        default=False,
        help="Show timers",
    )
    parser.add_option(
        "-T",
        "--settimer",
        dest="settimer",
        metavar="NUM MODE SETTINGS",
        default=None,
        nargs=3,
        help="Set timer. "
        + "NUM: number of the timer (1-6). "
        + "MODE: inactive, poweroff, default, color, preset, or warmwhite. "
        + "SETTINGS: a string of settings including time, repeatdays or date, "
        + "and other mode specific settings.   Use --timerhelp for more details.",
    )

    parser.add_option(
        "--protocol",
        dest="protocol",
        default=None,
        metavar="PROTOCOL",
        help="Set the device protocol. Currently only supports LEDENET",
    )

    other_group.add_option(
        "-v",
        "--volatile",
        action="store_true",
        dest="volatile",
        default=False,
        help="Don't persist mode setting with hard power cycle (RGB and WW modes only).",
    )
    parser.add_option_group(other_group)

    parser.usage = "usage: %prog [-sS10cwdkpCiltThe] [addr1 [addr2 [addr3] ...]."
    (options, args) = parser.parse_args()

    if options.debug:
        logging.basicConfig(level=logging.DEBUG)

    if options.showexamples:
        showUsageExamples()
        sys.exit(0)

    if options.timerhelp:
        showTimerHelp()
        sys.exit(0)

    if options.listpresets:
        print("Standard preset patterns (-p option):")
        for c in range(
            PresetPattern.seven_color_cross_fade, PresetPattern.seven_color_jumping + 1
        ):
            print(f"  {c:2} {PresetPattern.valtostr(c)}")
        print("\nStandard custom pattern types (-C option):")
        print("  jump     - Colors change instantly")
        print("  gradual  - Colors fade smoothly")
        print("  strobe   - Colors flash rapidly")
        print("\nExtended custom patterns (-C option, 0xB6 devices only):")
        for p in ExtendedCustomEffectPattern:
            print(f"  {p.name.lower().replace('_', ' '):<20} (ID: {p.value})")
        print("\nSegment colors (-C segments, 0xB6 devices only):")
        print("  segments - Set individual segment colors (up to 20)")
        sys.exit(0)

    if options.listcolors:
        for c in utils.get_color_names_list():  # type: ignore
            print(f"{c}, ")
        print()
        sys.exit(0)

    # Timer processing is deferred to _async_run_commands since we need
    # to know the device protocol first (0xB6 uses extended timer format)
    options.new_timer = None

    mode_count = 0
    if options.color:
        mode_count += 1
    if options.ww:
        mode_count += 1
    if options.cw:
        mode_count += 1
    if options.cct:
        mode_count += 1
    if options.preset:
        mode_count += 1
    if options.custom:
        mode_count += 1
    if mode_count > 1:
        parser.error(
            "options --color, --*white, --preset, --CCT, and --custom are mutually exclusive"
        )

    if options.on and options.off:
        parser.error("options --on and --off are mutually exclusive")

    if options.custom:
        options.custom = processCustomArgs(
            parser,
            options.custom,
            density=options.density,
            direction=options.direction,
            colorchange=options.colorchange,
        )

    if options.color:
        options.color = utils.color_object_to_tuple(options.color)
        if options.color is None:
            parser.error("bad color specification")

    if options.preset:
        if not PresetPattern.valid(options.preset[0]):
            parser.error("Preset code is not in range")

    # asking for timer info, implicitly gets the state
    if options.showtimers:
        options.info = True

    op_count = mode_count
    if options.on:
        op_count += 1
    if options.off:
        op_count += 1
    if options.info:
        op_count += 1
    if options.getclock:
        op_count += 1
    if options.setclock:
        op_count += 1
    if options.listpresets:
        op_count += 1
    if options.settimer:
        op_count += 1

    if (not options.scan or options.scanresults) and (op_count == 0):
        parser.error("An operation must be specified")

    # if we're not scanning, IP addresses must be specified as positional args
    if not options.scan and not options.scanresults and not options.listpresets:
        if len(args) == 0:
            parser.error(
                "You must specify at least one IP address as an argument, or use scan results"
            )

    return (options, args)


# -------------------------------------------


async def _async_run_commands(
    bulb: AIOWifiLedBulb, info: FluxLEDDiscovery, options: Any
) -> None:
    """Run requested commands on a bulb."""
    buffer = ""

    def buf_in(str: str) -> None:
        nonlocal buffer
        buffer += str + "\n"

    if options.getclock:
        buf_in(
            "{} [{}] {}".format(info["id"], info["ipaddr"], await bulb.async_get_time())
        )

    if options.setclock:
        await bulb.async_set_time()

    if options.ww is not None:
        if options.ww > 100:
            raise ValueError("Input can not be higher than 100%")
        buf_in(f"Setting warm white mode, level: {options.ww}%")
        await bulb.async_set_levels(
            w=utils.percentToByte(options.ww), persist=not options.volatile
        )

    if options.cw is not None:
        if options.cw > 100:
            raise ValueError("Input can not be higher than 100%")
        buf_in(f"Setting cold white mode, level: {options.cw}%")
        await bulb.async_set_levels(
            w2=utils.percentToByte(options.cw), persist=not options.volatile
        )

    if options.cct is not None:
        if options.cct[1] > 100:
            raise ValueError("Brightness can not be higher than 100%")
        if options.cct[0] < 2700 or options.cct[0] > 6500:
            buf_in("Color Temp must be between 2700 and 6500")
        else:
            buf_in(
                f"Setting LED temperature {options.cct[0]}K and brightness: {options.cct[1]}%"
            )
            await bulb.async_set_white_temp(
                options.cct[0], options.cct[1] * 2.55, persist=not options.volatile
            )

    if options.color is not None:
        buf_in(
            f"Setting color RGB:{options.color}",
        )
        name = utils.color_tuple_to_string(options.color)
        if name is None:
            buf_in("")
        else:
            buf_in(f"[{name}]")
        if any(i < 0 or i > 255 for i in options.color):
            raise ValueError("Invalid value received must be between 0-255")
        if len(options.color) == 3:
            await bulb.async_set_levels(
                options.color[0],
                options.color[1],
                options.color[2],
                persist=not options.volatile,
            )
        elif len(options.color) == 4:
            await bulb.async_set_levels(
                options.color[0],
                options.color[1],
                options.color[2],
                options.color[3],
                persist=not options.volatile,
            )
        elif len(options.color) == 5:
            await bulb.async_set_levels(
                options.color[0],
                options.color[1],
                options.color[2],
                options.color[3],
                options.color[4],
                persist=not options.volatile,
            )

    elif options.custom is not None:
        custom = options.custom
        mode = custom["mode"]

        if mode == "standard":
            # Standard custom pattern (jump, gradual, strobe)
            await bulb.async_set_custom_pattern(
                custom["colors"], custom["speed"], custom["type"]
            )
            buf_in(
                f"Setting custom pattern: {custom['type']}, "
                f"Speed={custom['speed']}%, {custom['colors']}"
            )

        elif mode == "extended":
            # Extended custom effect (wave, meteor, etc.)
            if not bulb.supports_extended_custom_effects:
                raise ValueError(
                    f"Device {bulb.model} (model_num=0x{bulb.model_num:02X}) "
                    "does not support extended custom effects. "
                    "Use jump, gradual, or strobe for this device."
                )
            pattern_id = custom["pattern_id"]
            speed = custom["speed"]
            density = custom["density"]
            colors = custom["colors"]
            direction = custom["direction"]
            option = custom["option"]

            # Get pattern name for display
            try:
                pattern_name = (
                    ExtendedCustomEffectPattern(pattern_id)
                    .name.lower()
                    .replace("_", " ")
                )
            except ValueError:
                pattern_name = f"pattern {pattern_id}"
            dir_name = (
                "L->R"
                if direction == ExtendedCustomEffectDirection.LEFT_TO_RIGHT.value
                else "R->L"
            )
            buf_in(
                f"Setting extended effect: {pattern_name}, "
                f"Speed={speed}%, Density={density}%, Direction={dir_name}, "
                f"Colors={colors}"
            )
            await bulb.async_set_extended_custom_effect(
                pattern_id, colors, speed, density, direction, option
            )

        elif mode == "segments":
            # Custom segment colors
            if not bulb.supports_extended_custom_effects:
                raise ValueError(
                    f"Device {bulb.model} (model_num=0x{bulb.model_num:02X}) "
                    "does not support custom segment colors. "
                    "This feature is only available on 0xB6 devices."
                )
            buf_in(f"Setting custom segment colors: {len(custom['colors'])} segments")
            await bulb.async_set_custom_segment_colors(custom["colors"])

    elif options.preset is not None:
        buf_in(
            f"Setting preset pattern: {PresetPattern.valtostr(options.preset[0])}, Speed={options.preset[1]}%"
        )
        await bulb.async_set_preset_pattern(options.preset[0], options.preset[1])

    if options.on:
        buf_in(f"Turning on bulb at {bulb.ipaddr}")
        await bulb.async_turn_on()
    elif options.off:
        buf_in(f"Turning off bulb at {bulb.ipaddr}")
        await bulb.async_turn_off()

    if options.info:
        buf_in("{} [{}] {} ({})".format(info["id"], info["ipaddr"], bulb, bulb.model))

    if options.settimer:
        # Check if device uses extended timer format (0xB6)
        is_extended = isinstance(bulb._protocol, ProtocolLEDENETExtendedCustom)
        num = int(options.settimer[0])

        if is_extended:
            # Create a minimal parser for error handling
            temp_parser = OptionParser()
            ext_timer = processSetTimerArgsExtended(temp_parser, options.settimer)
            buf_in(f"New Timer ---- #{num}: {ext_timer}")
            await bulb.async_set_timer(ext_timer)
        else:
            temp_parser = OptionParser()
            std_timer = processSetTimerArgs(temp_parser, options.settimer)
            buf_in(f"New Timer ---- #{num}: {std_timer}")
            if std_timer.isExpired():
                buf_in("[timer is already expired, will be deactivated]")
            timers_result = await bulb.async_get_timers()
            timers_list = cast("list[LedTimer]", timers_result) if timers_result else []
            if len(timers_list) < num:
                # Extend list if needed
                timers_list.extend([LedTimer() for _ in range(num - len(timers_list))])
            timers_list[num - 1] = std_timer
            await bulb.async_set_timers(timers_list)

    if options.showtimers:
        show_timers = await bulb.async_get_timers()
        if show_timers:
            for idx, t in enumerate(show_timers):
                buf_in(f"  Timer #{idx + 1}: {t}")
        buf_in("")

    print(buffer.rstrip("\n"))


async def _async_process_bulb(info: FluxLEDDiscovery, options: Any) -> None:
    """Process a bulb."""
    bulb = AIOWifiLedBulb(info["ipaddr"], discovery=info)
    await bulb.async_setup(lambda *args: None)
    try:
        await _async_run_commands(bulb, info, options)
    finally:
        await bulb.async_stop()


async def async_main() -> None:
    (options, args) = parseArgs()
    scanner = AIOBulbScanner()

    if options.scan:
        await scanner.async_scan(timeout=6)
        bulb_info_list = scanner.getBulbInfo()
        # we have a list of buld info dicts
        addrs = []
        if options.scanresults and len(bulb_info_list) > 0:
            for b in bulb_info_list:
                addrs.append(b["ipaddr"])
        else:
            print(f"{len(bulb_info_list)} bulbs found")
            for b in bulb_info_list:
                print("  {} {}".format(b["id"], b["ipaddr"]))
            return
    else:
        if options.info:
            for addr in args:
                await scanner.async_scan(timeout=6, address=addr)
            bulb_info_list = scanner.getBulbInfo()
        else:
            bulb_info_list = []
        found_addrs = {discovery[ATTR_IPADDR] for discovery in bulb_info_list}
        for addr in args:
            if addr in found_addrs:
                continue
            bulb_info_list.append(
                FluxLEDDiscovery({ATTR_IPADDR: addr, ATTR_ID: "Unknown ID"})  # type: ignore[typeddict-item]
            )

    # now we have our bulb list, perform same operation on all of them
    tasks = [_async_process_bulb(info, options) for info in bulb_info_list]
    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )
    for idx, info in enumerate(bulb_info_list):
        if isinstance(results[idx], Exception):
            msg = str(results[idx]) or type(results[idx])
            print(f"Error while processing {info}: {msg}")
    return


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
    sys.exit(0)
