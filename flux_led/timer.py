from __future__ import annotations

import datetime

from .pattern import PresetPattern
from .utils import utils


class BuiltInTimer:
    sunrise = 0xA1
    sunset = 0xA2

    @staticmethod
    def valid(byte_value: int) -> bool:
        return byte_value == BuiltInTimer.sunrise or byte_value == BuiltInTimer.sunset

    @staticmethod
    def valtostr(pattern: int) -> str:
        for key, value in list(BuiltInTimer.__dict__.items()):
            if type(value) is int and value == pattern:
                return key.replace("_", " ").title()
        raise ValueError(f"{pattern} must be 0xA1 or 0xA2")


class LedTimer:
    Mo = 0x02
    Tu = 0x04
    We = 0x08
    Th = 0x10
    Fr = 0x20
    Sa = 0x40
    Su = 0x80
    Everyday = Mo | Tu | We | Th | Fr | Sa | Su
    Weekdays = Mo | Tu | We | Th | Fr
    Weekend = Sa | Su

    @staticmethod
    def dayMaskToStr(mask: int) -> str:
        for key, value in LedTimer.__dict__.items():
            if type(value) is int and value == mask:
                return key
        raise ValueError(
            f"{mask} must be one of 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80"
        )

    def __init__(
        self, bytes: bytes | bytearray | None = None, length: int = 14
    ) -> None:
        self.cold_level = 0
        self.pattern_code = 0
        self.delay = 0
        if bytes is not None:
            self.length = len(bytes)
            self.fromBytes(bytes)
            return

        self.length = length
        the_time = datetime.datetime.now() + datetime.timedelta(hours=1)
        self.setTime(the_time.hour, the_time.minute)
        self.setDate(the_time.year, the_time.month, the_time.day)
        self.setModeTurnOff()
        self.setActive(False)

    def setActive(self, active: bool = True) -> None:
        self.active = active

    def isActive(self) -> bool:
        return self.active

    def isExpired(self) -> bool:
        # if no repeat mask and datetime is in past, return True
        if self.repeat_mask != 0:
            return False
        if self.year != 0 and self.month != 0 and self.day != 0:
            dt = datetime.datetime(
                self.year, self.month, self.day, self.hour, self.minute
            )
            if utils.date_has_passed(dt):
                return True
        return False

    def setTime(self, hour: int, minute: int) -> None:
        self.hour = hour
        self.minute = minute

    def setDate(self, year: int, month: int, day: int) -> None:
        self.year = year
        self.month = month
        self.day = day
        self.repeat_mask = 0

    def setRepeatMask(self, repeat_mask: int) -> None:
        self.year = 0
        self.month = 0
        self.day = 0
        self.repeat_mask = repeat_mask

    def setModeDefault(self) -> None:
        self.mode = "default"
        self.pattern_code = 0
        self.turn_on = True
        self.red = 0
        self.green = 0
        self.blue = 0
        self.warmth_level = 0
        self.cold_level = 0

    def setModePresetPattern(self, pattern: int, speed: int) -> None:
        self.mode = "preset"
        self.warmth_level = 0
        self.cold_level = 0
        self.pattern_code = pattern
        self.delay = utils.speedToDelay(speed)
        self.turn_on = True

    def setModeColor(self, r: int, g: int, b: int) -> None:
        self.mode = "color"
        self.warmth_level = 0
        self.cold_level = 0
        self.red = r
        self.green = g
        self.blue = b
        self.pattern_code = 0x61
        self.turn_on = True

    def setModeWarmWhite(self, level: int) -> None:
        self.mode = "ww"
        self.warmth_level = utils.percentToByte(level)
        self.cold_level = 0
        self.pattern_code = 0x61
        self.red = 0
        self.green = 0
        self.blue = 0
        self.turn_on = True

    def setModeSunrise(
        self, startBrightness: int, endBrightness: int, duration: int
    ) -> None:
        self.mode = "sunrise"
        self.turn_on = True
        self.pattern_code = BuiltInTimer.sunrise
        self.brightness_start = utils.percentToByte(startBrightness)
        self.brightness_end = utils.percentToByte(endBrightness)
        self.warmth_level = utils.percentToByte(endBrightness)
        self.cold_level = 0
        self.duration = int(duration)

    def setModeSunset(
        self, startBrightness: int, endBrightness: int, duration: int
    ) -> None:
        self.mode = "sunrise"
        self.turn_on = True
        self.pattern_code = BuiltInTimer.sunset
        self.brightness_start = utils.percentToByte(startBrightness)
        self.brightness_end = utils.percentToByte(endBrightness)
        self.warmth_level = utils.percentToByte(endBrightness)
        self.cold_level = 0
        self.duration = int(duration)

    def setModeTurnOff(self) -> None:
        self.mode = "off"
        self.turn_on = False
        self.pattern_code = 0

    """

    timer are in six 14-byte structs
        f0 0f 08 10 10 15 00 00 25 1f 00 00 00 f0 0f
         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14

        0: f0 when active entry/ 0f when not active
        1: (0f=15) year when no repeat, else 0
        2:  month when no repeat, else 0
        3:  dayofmonth when no repeat, else 0
        4: hour
        5: min
        6: 0
        7: repeat mask, Mo=0x2,Tu=0x04, We 0x8, Th=0x10 Fr=0x20, Sa=0x40, Su=0x80
        8:  61 for solid color or warm, or preset pattern code
        9:  r (or delay for preset pattern)
        10: g
        11: b
        12: warm white level
        13: 0f = turn off, f0 = turn on

    timer are in six 15-byte structs for 9 byte devices
        f0 0f 08 10 10 15 00 00 25 1f 00 00 00 f0 0f
         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14

        0: f0 when active entry/ 0f when not active
        1: (0f=15) year when no repeat, else 0
        2:  month when no repeat, else 0
        3:  dayofmonth when no repeat, else 0
        4: hour
        5: min
        6: 0
        7: repeat mask, Mo=0x2,Tu=0x04, We 0x8, Th=0x10 Fr=0x20, Sa=0x40, Su=0x80
        8:  61 for solid color or warm, or preset pattern code
        9:  r (or delay for preset pattern)
        10: g
        11: b
        12: warm white level
        13: cold white level
        14: 0f = turn off, f0 = turn on
    """

    def fromBytes(self, bytes: bytes | bytearray) -> None:
        self.red = 0
        self.green = 0
        self.blue = 0
        if bytes[0] == 0xF0:
            self.active = True
        else:
            self.active = False
        self.year = bytes[1] + 2000
        self.month = bytes[2]
        self.day = bytes[3]
        self.hour = bytes[4]
        self.minute = bytes[5]
        self.repeat_mask = bytes[7]

        if len(bytes) == 12:  # sockets
            if bytes[8] == 0x23:
                self.turn_on = True
            else:
                self.turn_on = False
                self.mode = "off"
            return

        self.pattern_code = bytes[8]
        if self.pattern_code == 0x00:
            self.mode = "default"
        elif self.pattern_code == 0x61:
            self.mode = "color"
            self.red = bytes[9]
            self.green = bytes[10]
            self.blue = bytes[11]
        elif BuiltInTimer.valid(self.pattern_code):
            self.mode = BuiltInTimer.valtostr(self.pattern_code)
            self.duration = bytes[9]  # same byte as red
            self.brightness_start = bytes[10]  # same byte as green
            self.brightness_end = bytes[11]  # same byte as blue
        elif PresetPattern.valid(self.pattern_code):
            self.mode = "preset"
            self.delay = bytes[9]  # same byte as red
        else:
            self.mode = "unknown"

        self.warmth_level = bytes[12]
        if self.warmth_level != 0:
            self.mode = "ww"

        if len(bytes) == 15:  # 9 byte protocol
            self.cold_level = bytes[13]
            on_byte = bytes[14]
        else:  # 8 byte protocol
            on_byte = bytes[13]

        if on_byte == 0xF0:
            self.turn_on = True
        else:
            self.turn_on = False
            self.mode = "off"

    def toBytes(self) -> bytearray:
        bytes = bytearray(self.length)
        if not self.active:
            bytes[0] = 0x0F
            # quit since all other zeros is good
            return bytes

        bytes[0] = 0xF0

        if self.year >= 2000:
            bytes[1] = self.year - 2000
        else:
            bytes[1] = self.year
        bytes[2] = self.month
        bytes[3] = self.day
        bytes[4] = self.hour
        bytes[5] = self.minute
        # what is 6?
        bytes[7] = self.repeat_mask

        if self.length == 12:
            bytes[8] == 0x23 if self.turn_on else 0x24
            return bytes

        on_byte_num = 14 if self.length == 15 else 13
        if not self.turn_on:
            bytes[on_byte_num] = 0x0F
            return bytes
        bytes[on_byte_num] = 0xF0

        bytes[8] = self.pattern_code
        if PresetPattern.valid(self.pattern_code):
            bytes[9] = self.delay
            bytes[10] = 0
            bytes[11] = 0
        elif BuiltInTimer.valid(self.pattern_code):
            bytes[9] = self.duration
            bytes[10] = self.brightness_start
            bytes[11] = self.brightness_end
        else:
            bytes[9] = self.red
            bytes[10] = self.green
            bytes[11] = self.blue
        bytes[12] = self.warmth_level
        if self.length == 15:
            bytes[13] = self.cold_level

        return bytes

    def __str__(self) -> str:
        txt = ""
        if not self.active:
            return "Unset"

        if self.turn_on:
            txt += "[ON ]"
        else:
            txt += "[OFF]"

        txt += " "

        txt += f"{self.hour:02}:{self.minute:02}  "

        if self.repeat_mask == 0:
            txt += f"Once: {self.year:04}-{self.month:02}-{self.day:02}"
        else:
            bits = [
                LedTimer.Su,
                LedTimer.Mo,
                LedTimer.Tu,
                LedTimer.We,
                LedTimer.Th,
                LedTimer.Fr,
                LedTimer.Sa,
            ]
            for b in bits:
                if self.repeat_mask & b:
                    txt += LedTimer.dayMaskToStr(b)
                else:
                    txt += "--"
            txt += "  "

        txt += "  "
        if self.pattern_code == 0x61:
            if self.warmth_level != 0:
                txt += f"Warm White: {utils.byteToPercent(self.warmth_level)}%"
            else:
                color_str = utils.color_tuple_to_string(
                    (self.red, self.green, self.blue)
                )
                txt += f"Color: {color_str}"

        elif PresetPattern.valid(self.pattern_code):
            pat = PresetPattern.valtostr(self.pattern_code)
            speed = utils.delayToSpeed(self.delay)
            txt += f"{pat} (Speed:{speed}%)"

        elif BuiltInTimer.valid(self.pattern_code):
            type = BuiltInTimer.valtostr(self.pattern_code)

            txt += f"{type} (Duration:{self.duration} minutes, Brightness: {utils.byteToPercent(self.brightness_start)}% -> {utils.byteToPercent(self.brightness_end)}%)"

        return txt
