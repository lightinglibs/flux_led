from __future__ import annotations

from typing import cast

from .const import MultiColorEffects

EFFECT_COLORLOOP = "colorloop"
EFFECT_RED_FADE = "red_fade"
EFFECT_GREEN_FADE = "green_fade"
EFFECT_BLUE_FADE = "blue_fade"
EFFECT_YELLOW_FADE = "yellow_fade"
EFFECT_CYAN_FADE = "cyan_fade"
EFFECT_PURPLE_FADE = "purple_fade"
EFFECT_WHITE_FADE = "white_fade"
EFFECT_RED_GREEN_BLUE_CROSS_FADE = "rgb_cross_fade"
EFFECT_RED_GREEN_CROSS_FADE = "rg_cross_fade"
EFFECT_RED_BLUE_CROSS_FADE = "rb_cross_fade"
EFFECT_GREEN_BLUE_CROSS_FADE = "gb_cross_fade"
EFFECT_COLORSTROBE = "colorstrobe"
EFFECT_RED_STROBE = "red_strobe"
EFFECT_GREEN_STROBE = "green_strobe"
EFFECT_BLUE_STROBE = "blue_strobe"
EFFECT_YELLOW_STROBE = "yellow_strobe"
EFFECT_CYAN_STROBE = "cyan_strobe"
EFFECT_PURPLE_STROBE = "purple_strobe"
EFFECT_WHITE_STROBE = "white_strobe"
EFFECT_CYCLE_RGB = "cycle_rgb"
EFFECT_CYCLE_SEVEN_COLORS = "cycle_seven_colors"
EFFECT_COLORJUMP = "colorjump"
EFFECT_CUSTOM = "custom"

EFFECT_WARM_FLASH = "Warm Flash"
EFFECT_COOL_FLASH = "Cool Flash"
EFFECT_WARM_GRADUAL = "Warm Gradual"
EFFECT_COOL_GRADUAL = "Cool Gradual"

EFFECT_MAP = {
    EFFECT_COLORLOOP: 0x25,
    EFFECT_RED_FADE: 0x26,
    EFFECT_GREEN_FADE: 0x27,
    EFFECT_BLUE_FADE: 0x28,
    EFFECT_YELLOW_FADE: 0x29,
    EFFECT_CYAN_FADE: 0x2A,
    EFFECT_PURPLE_FADE: 0x2B,
    EFFECT_WHITE_FADE: 0x2C,
    EFFECT_RED_GREEN_CROSS_FADE: 0x2D,
    EFFECT_RED_BLUE_CROSS_FADE: 0x2E,
    EFFECT_GREEN_BLUE_CROSS_FADE: 0x2F,
    EFFECT_COLORSTROBE: 0x30,
    EFFECT_RED_STROBE: 0x31,
    EFFECT_GREEN_STROBE: 0x32,
    EFFECT_BLUE_STROBE: 0x33,
    EFFECT_YELLOW_STROBE: 0x34,
    EFFECT_CYAN_STROBE: 0x35,
    EFFECT_PURPLE_STROBE: 0x36,
    EFFECT_WHITE_STROBE: 0x37,
    EFFECT_COLORJUMP: 0x38,
}
EFFECT_MAP_DIMMABLE = {
    EFFECT_RED_GREEN_BLUE_CROSS_FADE: 0x24,
    **EFFECT_MAP,
    EFFECT_CYCLE_RGB: 0x39,
    EFFECT_CYCLE_SEVEN_COLORS: 0x3A,
    # cycle_seven_colors Doesn't work on the bulbs, but no way to tell ahead of time
    # since the firmware version is v9 for both but it seems like only the AK001-ZJ2147
    # model actually has support for it
}

EFFECT_MAP_LEGACY_CCT = {
    EFFECT_WARM_GRADUAL: 0x3A,
    EFFECT_WARM_FLASH: 0x3C,
    EFFECT_COOL_GRADUAL: 0x4A,
    EFFECT_COOL_FLASH: 0x4C,
}

EFFECT_ID_NAME_LEGACY_CCT = {v: k for k, v in EFFECT_MAP_LEGACY_CCT.items()}
EFFECT_ID_NAME = {v: k for k, v in EFFECT_MAP_DIMMABLE.items()}
EFFECT_CUSTOM_CODE = 0x60

EFFECT_LIST = sorted(EFFECT_MAP)
EFFECT_LIST_DIMMABLE = sorted(EFFECT_MAP_DIMMABLE)
EFFECT_LIST_LEGACY_CCT = sorted(EFFECT_MAP_LEGACY_CCT)

ADDRESSABLE_EFFECT_ID_NAME = {
    1: "RBM 1",
    2: "RBM 2",
    3: "RBM 3",
    4: "RBM 4",
    5: "RBM 5",
    6: "RBM 6",
    7: "RBM 7",
    8: "RBM 8",
    9: "RBM 9",
    10: "RBM 10",
    11: "RBM 11",
    12: "RBM 12",
    13: "RBM 13",
    14: "RBM 14",
    15: "RBM 15",
    16: "RBM 16",
    17: "RBM 17",
    18: "RBM 18",
    19: "RBM 19",
    20: "RBM 20",
    21: "RBM 21",
    22: "RBM 22",
    23: "RBM 23",
    24: "RBM 24",
    25: "RBM 25",
    26: "RBM 26",
    27: "RBM 27",
    28: "RBM 28",
    29: "RBM 29",
    30: "RBM 30",
    31: "RBM 31",
    32: "RBM 32",
    33: "RBM 33",
    34: "RBM 34",
    35: "RBM 35",
    36: "RBM 36",
    37: "RBM 37",
    38: "RBM 38",
    39: "RBM 39",
    40: "RBM 40",
    41: "RBM 41",
    42: "RBM 42",
    43: "RBM 43",
    44: "RBM 44",
    45: "RBM 45",
    46: "RBM 46",
    47: "RBM 47",
    48: "RBM 48",
    49: "RBM 49",
    50: "RBM 50",
    51: "RBM 51",
    52: "RBM 52",
    53: "RBM 53",
    54: "RBM 54",
    55: "RBM 55",
    56: "RBM 56",
    57: "RBM 57",
    58: "RBM 58",
    59: "RBM 59",
    60: "RBM 60",
    61: "RBM 61",
    62: "RBM 62",
    63: "RBM 63",
    64: "RBM 64",
    65: "RBM 65",
    66: "RBM 66",
    67: "RBM 67",
    68: "RBM 68",
    69: "RBM 69",
    70: "RBM 70",
    71: "RBM 71",
    72: "RBM 72",
    73: "RBM 73",
    74: "RBM 74",
    75: "RBM 75",
    76: "RBM 76",
    77: "RBM 77",
    78: "RBM 78",
    79: "RBM 79",
    80: "RBM 80",
    81: "RBM 81",
    82: "RBM 82",
    83: "RBM 83",
    84: "RBM 84",
    85: "RBM 85",
    86: "RBM 86",
    87: "RBM 87",
    88: "RBM 88",
    89: "RBM 89",
    90: "RBM 90",
    91: "RBM 91",
    92: "RBM 92",
    93: "RBM 93",
    94: "RBM 94",
    95: "RBM 95",
    96: "RBM 96",
    97: "RBM 97",
    98: "RBM 98",
    99: "RBM 99",
    100: "RBM 100",
    101: "RBM 101",  # Not in the Magic Home App (only set by remote)
    102: "RBM 102",  # Not in the Magic Home App (only set by remote)
    255: "Circulate all modes",  # Cycles all
}
ADDRESSABLE_EFFECT_NAME_ID = {v: k for k, v in ADDRESSABLE_EFFECT_ID_NAME.items()}

ASSESSABLE_MULTI_COLOR_ID_NAME = {
    MultiColorEffects.STATIC.value: "Multi Color Static",
    MultiColorEffects.RUNNING_WATER.value: "Multi Color Running Water",
    MultiColorEffects.STROBE.value: "Multi Color Strobe",
    MultiColorEffects.JUMP.value: "Multi Color Jump",
    MultiColorEffects.BREATHING.value: "Multi Color Breathing",
}


ASSESSABLE_MULTI_COLOR_NAME_ID = {
    v: k for k, v in ASSESSABLE_MULTI_COLOR_ID_NAME.items()
}

ORIGINAL_ADDRESSABLE_EFFECT_ID_NAME = {
    1: "Circulate all modes",
    2: "7 colors change gradually",
    3: "7 colors run in olivary",
    4: "7 colors change quickly",
    5: "7 colors strobe-flash",
    6: "7 colors running, 1 point from start to end and return back",
    7: "7 colors running, multi points from start to end and return back",
    8: "7 colors overlay, multi points from start to end and return back",
    9: "7 colors overlay, multi points from the middle to the both ends and return back",
    10: "7 colors flow gradually, from start to end and return back",
    11: "Fading out run, 7 colors from start to end and return back",
    12: "Runs in olivary, 7 colors from start to end and return back",
    13: "Fading out run, 7 colors start with white color from start to end and return back",
    14: "Run circularly, 7 colors with black background, 1point from start to end",
    15: "Run circularly, 7 colors with red background, 1point from start to end",
    16: "Run circularly, 7 colors with green background, 1point from start to end",
    17: "Run circularly, 7 colors with blue background, 1point from start to end",
    18: "Run circularly, 7 colors with yellow background, 1point from start to end",
    19: "Run circularly, 7 colors with purple background, 1point from start to end",
    20: "Run circularly, 7 colors with cyan background, 1point from start to end",
    21: "Run circularly, 7 colors with white background, 1point from start to end",
    22: "Run circularly, 7 colors with black background, 1point from end to start",
    23: "Run circularly, 7 colors with red background, 1point from end to start",
    24: "Run circularly, 7 colors with green background, 1point from end to start",
    25: "Run circularly, 7 colors with blue background, 1point from end to start",
    26: "Run circularly, 7 colors with yellow background, 1point from end to start",
    27: "Run circularly, 7 colors with purple background, 1point from end to start",
    28: "Run circularly, 7 colors with cyan background, 1point from end to start",
    29: "Run circularly, 7 colors with white background, 1point from end to start",
    30: "Run circularly, 7 colors with black background, 1point from start to end and return back",
    31: "Run circularly, 7 colors with red background, 1point from start to end and return back",
    32: "Run circularly, 7 colors with green background, 1point from start to end and return back",
    33: "Run circularly, 7 colors with blue background, 1point from start to end and return back",
    34: "Run circularly, 7 colors with yellow background, 1point from start to end and return back",
    35: "Run circularly, 7 colors with purple background, 1point from start to end and return back",
    36: "Run circularly, 7 colors with cyan background, 1point from start to end and return back",
    37: "Run circularly, 7 colors with white background, 1point from start to end and return back",
    38: "Run circularly, 7 colors with black background, 1point from middle to both ends",
    39: "Run circularly, 7 colors with red background, 1point from middle to both ends",
    40: "Run circularly, 7 colors with green background, 1point from middle to both ends",
    41: "Run circularly, 7 colors with blue background, 1point from middle to both ends",
    42: "Run circularly, 7 colors with yellow background, 1point from middle to both ends",
    43: "Run circularly, 7 colors with purple background, 1point from middle to both ends",
    44: "Run circularly, 7 colors with cyan background, 1point from middle to both ends",
    45: "Run circularly, 7 colors with white background, 1point from middle to both ends",
    46: "Run circularly, 7 colors with black background, 1point from both ends to middle",
    47: "Run circularly, 7 colors with red background, 1point from both ends to middle",
    48: "Run circularly, 7 colors with green background, 1point from both ends to middle",
    49: "Run circularly, 7 colors with blue background, 1point from both ends to middle",
    50: "Run circularly, 7 colors with yellow background, 1point from both ends to middle",
    51: "Run circularly, 7 colors with purple background, 1point from both ends to middle",
    52: "Run circularly, 7 colors with cyan background, 1point from both ends to middle",
    53: "Run circularly, 7 colors with white background, 1point from both ends to middle",
    54: "Run circularly, 7 colors with black background, 1point from middle to both ends and return back",
    55: "Run circularly, 7 colors with red background, 1point from middle to both ends and return back",
    56: "Run circularly, 7 colors with green background, 1point from middle to both ends and return back",
    57: "Run circularly, 7 colors with blue background, 1point from middle to both ends and return back",
    58: "Run circularly, 7 colors with yellow background, 1point from middle to both ends and return back",
    59: "Run circularly, 7 colors with purple background, 1point from middle to both ends and return back",
    60: "Run circularly, 7 colors with cyan background, 1point from middle to both ends and return back",
    61: "Run circularly, 7 colors with white background, 1point from middle to both ends and return back",
    62: "Overlay circularly, 7 colors with black background from start to end",
    63: "Overlay circularly, 7 colors with red background from start to end",
    64: "Overlay circularly, 7 colors with green background from start to end",
    65: "Overlay circularly, 7 colors with blue background from start to end",
    66: "Overlay circularly, 7 colors with yellow background from start to end",
    67: "Overlay circularly, 7 colors with purple background from start to end",
    68: "Overlay circularly, 7 colors with cyan background from start to end",
    69: "Overlay circularly, 7 colors with white background from start to end",
    70: "Overlay circularly, 7 colors with black background from end to start",
    71: "Overlay circularly, 7 colors with red background from end to start",
    72: "Overlay circularly, 7 colors with green background from end to start",
    73: "Overlay circularly, 7 colors with blue background from end to start",
    74: "Overlay circularly, 7 colors with yellow background from end to start",
    75: "Overlay circularly, 7 colors with purple background from end to start",
    76: "Overlay circularly, 7 colors with cyan background from end to start",
    77: "Overlay circularly, 7 colors with white background from end to start",
    78: "Overlay circularly, 7 colors with black background from start to end and return back",
    79: "Overlay circularly, 7 colors with red background from start to end and return back",
    80: "Overlay circularly, 7 colors with green background from start to end and return back",
    81: "Overlay circularly, 7 colors with blue background from start to end and return back",
    82: "Overlay circularly, 7 colors with yellow background from start to end and return back",
    83: "Overlay circularly, 7 colors with purple background from start to end and return back",
    84: "Overlay circularly, 7 colors with cyan background from start to end and return back",
    85: "Overlay circularly, 7 colors with white background from start to end and return back",
    86: "Overlay circularly, 7 colors with black background from middle to both ends",
    87: "Overlay circularly, 7 colors with red background from middle to both ends",
    88: "Overlay circularly, 7 colors with green background from middle to both ends",
    89: "Overlay circularly, 7 colors with blue background from middle to both ends",
    90: "Overlay circularly, 7 colors with yellow background from middle to both ends",
    91: "Overlay circularly, 7 colors with purple background from middle to both ends",
    92: "Overlay circularly, 7 colors with cyan background from middle to both ends",
    93: "Overlay circularly, 7 colors with white background from middle to both ends",
    94: "Overlay circularly, 7 colors with black background from both ends to middle",
    95: "Overlay circularly, 7 colors with red background from both ends to middle",
    96: "Overlay circularly, 7 colors with green background from both ends to middle",
    97: "Overlay circularly, 7 colors with blue background from both ends to middle",
    98: "Overlay circularly, 7 colors with yellow background from both ends to middle",
    99: "Overlay circularly, 7 colors with purple background from both ends to middle",
    100: "Overlay circularly, 7 colors with cyan background from both ends to middle",
    101: "Overlay circularly, 7 colors with white background from both ends to middle",
    102: "Overlay circularly, 7 colors with black background from middle to both sides and return back",
    103: "Overlay circularly, 7 colors with red background from middle to both sides and return back",
    104: "Overlay circularly, 7 colors with green background from middle to both sides and return back",
    105: "Overlay circularly, 7 colors with blue background from middle to both sides and return back",
    106: "Overlay circularly, 7 colors with yellow background from middle to both sides and return back",
    107: "Overlay circularly, 7 colors with purple background from middle to both sides and return back",
    108: "Overlay circularly, 7 colors with cyan background from middle to both sides and return back",
    109: "Overlay circularly, 7 colors with white background from middle to both sides and return back",
    110: "Fading out run circularly, 1point with black background from start to end",
    111: "Fading out run circularly, 1point with red background from start to end",
    112: "Fading out run circularly, 1point with green background from start to end",
    113: "Fading out run circularly, 1point with blue background from start to end",
    114: "Fading out run circularly, 1point with yellow background from start to end",
    115: "Fading out run circularly, 1point with purple background from start to end",
    116: "Fading out run circularly, 1point with cyan background from start to end",
    117: "Fading out run circularly, 1point with white background from start to end",
    118: "Fading out run circularly, 1point with black background from end to start",
    119: "Fading out run circularly, 1point with red background from end to start",
    120: "Fading out run circularly, 1point with green background from end to start",
    121: "Fading out run circularly, 1point with blue background from end to start",
    122: "Fading out run circularly, 1point with yellow background from end to start",
    123: "Fading out run circularly, 1point with purple background from end to start",
    124: "Fading out run circularly, 1point with cyan background from end to start",
    125: "Fading out run circularly, 1point with white background from end to start",
    126: "Fading out run circularly, 1point with black background from start to end and return back",
    127: "Fading out run circularly, 1point with red background from start to end and return back",
    128: "Fading out run circularly, 1point with green background from start to end and return back",
    129: "Fading out run circularly, 1point with blue background from start to end and return back",
    130: "Fading out run circularly, 1point with yellow background from start to end and return back",
    131: "Fading out run circularly, 1point with purple background from start to end and return back",
    132: "Fading out run circularly, 1point with cyan background from start to end and return back",
    133: "Fading out run circularly, 1point with white background from start to end and return back",
    134: "Flows in olivary circularly, 7 colors with black background from start to end",
    135: "Flows in olivary circularly, 7 colors with red background from start to end",
    136: "Flows in olivary circularly, 7 colors with green background from start to end",
    137: "Flows in olivary circularly, 7 colors with blue background from start to end",
    138: "Flows in olivary circularly, 7 colors with yellow background from start to end",
    139: "Flows in olivary circularly, 7 colors with purple background from start to end",
    140: "Flows in olivary circularly, 7 colors with cyan background from start to end",
    141: "Flows in olivary circularly, 7 colors with white background from start to end",
    142: "Flows in olivary circularly, 7 colors with black background from end to start",
    143: "Flows in olivary circularly, 7 colors with red background from end to start",
    144: "Flows in olivary circularly, 7 colors with green background from end to start",
    145: "Flows in olivary circularly, 7 colors with blue background from end to start",
    146: "Flows in olivary circularly, 7 colors with yellow background from end to start",
    147: "Flows in olivary circularly, 7 colors with purple background from end to start",
    148: "Flows in olivary circularly, 7 colors with cyan background from end to start",
    149: "Flows in olivary circularly, 7 colors with white background from end to start",
    150: "Flows in olivary circularly, 7 colors with black background from start to end and return back",
    151: "Flows in olivary circularly, 7 colors with red background from start to end and return back",
    152: "Flows in olivary circularly, 7 colors with green background from start to end and return back",
    153: "Flows in olivary circularly, 7 colors with blue background from start to end and return back",
    154: "Flows in olivary circularly, 7 colors with yellow background from start to end and return back",
    155: "Flows in olivary circularly, 7 colors with purple background from start to end and return back",
    156: "Flows in olivary circularly, 7 colors with cyan background from start to end and return back",
    157: "Flows in olivary circularly, 7 colors with white background from start to end and return back",
    158: "7 colors run circularly, each color in every 1 point with black background from start to end",
    159: "7 colors run circularly, each color in every 1 point with red background from start to end",
    160: "7 colors run circularly, each color in every 1 point with green background from start to end",
    161: "7 colors run circularly, each color in every 1 point with blue background from start to end",
    162: "7 colors run circularly, each color in every 1 point with yellow background from start to end",
    163: "7 colors run circularly, each color in every 1 point with purple background from start to end",
    164: "7 colors run circularly, each color in every 1 point with cyan background from start to end",
    165: "7 colors run circularly, each color in every 1 point with white background from start to end",
    166: "7 colors run circularly, each color in every 1 point with black background from end to start",
    167: "7 colors run circularly, each color in every 1 point with red background from end to start",
    168: "7 colors run circularly, each color in every 1 point with green background from end to start",
    169: "7 colors run circularly, each color in every 1 point with blue background from end to start",
    170: "7 colors run circularly, each color in every 1 point with yellow background from end to start",
    171: "7 colors run circularly, each color in every 1 point with purple background from end to start",
    172: "7 colors run circularly, each color in every 1 point with cyan background from end to start",
    173: "7 colors run circularly, each color in every 1 point with white background from end to start",
    174: "7 colors run circularly, each color in every 1 point with black background from start to end and return back",
    175: "7 colors run circularly, each color in every 1 point with red background from start to end and return back",
    176: "7 colors run circularly, each color in every 1 point with green background from start to end and return back",
    177: "7 colors run circularly, each color in every 1 point with blue background from start to end and return back",
    178: "7 colors run circularly, each color in every 1 point with yellow background from start to end and return back",
    179: "7 colors run circularly, each color in every 1 point with purple background from start to end and return back",
    180: "7 colors run circularly, each color in every 1 point with cyan background from start to end and return back",
    181: "7 colors run circularly, each color in every 1 point with white background from start to end and return back",
    182: "7 colors run circularly, each color in multi points with red background from start to end",
    183: "7 colors run circularly, each color in multi points with green background from start to end",
    184: "7 colors run circularly, each color in multi points with blue background from start to end",
    185: "7 colors run circularly, each color in multi points with yellow background from start to end",
    186: "7 colors run circularly, each color in multi points with purple background from start to end",
    187: "7 colors run circularly, each color in multi points with cyan background from start to end",
    188: "7 colors run circularly, each color in multi points with white background from start to end",
    189: "7 colors run circularly, each color in multi points with red background from end to start",
    190: "7 colors run circularly, each color in multi points with green background from end to start",
    191: "7 colors run circularly, each color in multi points with blue background from end to start",
    192: "7 colors run circularly, each color in multi points with yellow background from end to start",
    193: "7 colors run circularly, each color in multi points with purple background from end to start",
    194: "7 colors run circularly, each color in multi points with cyan background from end to start",
    195: "7 colors run circularly, each color in multi points with white background from end to start",
    196: "7 colors run circularly, each color in multi points with red background from start to end and return back",
    197: "7 colors run circularly, each color in multi points with green background from start to end and return back",
    198: "7 colors run circularly, each color in multi points with blue background from start to end and return back",
    199: "7 colors run circularly, each color in multi points with yellow background from start to end and return back",
    200: "7 colors run circularly, each color in multi points with purple background from start to end and return back",
    201: "7 colors run circularly, each color in multi points with cyan background from start to end and return back",
    202: "7 colors run circularly, each color in multi points with white background from start to end and return back",
    203: "Fading out run circularly, 7 colors each in red fading from start to end",
    204: "Fading out run circularly, 7 colors each in green fading from start to end",
    205: "Fading out run circularly, 7 colors each in blue fading from start to end",
    206: "Fading out run circularly, 7 colors each in yellow fading from start to end",
    207: "Fading out run circularly, 7 colors each in purple fading from start to end",
    208: "Fading out run circularly, 7 colors each in cyan fading from start to end",
    209: "Fading out run circularly, 7 colors each in white fading from start to end",
    210: "Fading out run circularly, 7 colors each in red fading from end to start",
    211: "Fading out run circularly, 7 colors each in green fading from end to start",
    212: "Fading out run circularly, 7 colors each in blue fading from end to start",
    213: "Fading out run circularly, 7 colors each in yellow fading from end to start",
    214: "Fading out run circularly, 7 colors each in purple fading from end to start",
    215: "Fading out run circularly, 7 colors each in cyan fading from end to start",
    216: "Fading out run circularly, 7 colors each in white fading from end to start",
    217: "Fading out run circularly, 7 colors each in red fading from start to end and return back",
    218: "Fading out run circularly, 7 colors each in green fading from start to end and return back",
    219: "Fading out run circularly, 7 colors each in blue fading from start to end and return back",
    220: "Fading out run circularly, 7 colors each in yellow fading from start to end and return back",
    221: "Fading out run circularly, 7 colors each in purple fading from start to end and return back",
    222: "Fading out run circularly, 7 colors each in cyan fading from start to end and return back",
    223: "Fading out run circularly, 7 colors each in white fading from start to end and return back",
    224: "7 colors each in red run circularly, multi points from start to end",
    225: "7 colors each in green run circularly, multi points from start to end",
    226: "7 colors each in blue run circularly, multi points from start to end",
    227: "7 colors each in yellow run circularly, multi points from start to end",
    228: "7 colors each in purple run circularly, multi points from start to end",
    229: "7 colors each in cyan run circularly, multi points from start to end",
    230: "7 colors each in white run circularly, multi points from start to end",
    231: "7 colors each in red run circularly, multi points from end to start",
    232: "7 colors each in green run circularly, multi points from end to start",
    233: "7 colors each in blue run circularly, multi points from end to start",
    234: "7 colors each in yellow run circularly, multi points from end to start",
    235: "7 colors each in purple run circularly, multi points from end to start",
    236: "7 colors each in cyan run circularly, multi points from end to start",
    237: "7 colors each in white run circularly, multi points from end to start",
    238: "7 colors each in red run circularly, multi points from start to end and return back",
    239: "7 colors each in green run circularly, multi points from start to end and return back",
    240: "and return back7 colors each in blue run circularly, multi points from start to end",
    241: "7 colors each in yellow run circularly, multi points from start to end and return back",
    242: "7 colors each in purple run circularly, multi points from start to end and return back",
    243: "7 colors each in cyan run circularly, multi points from start to end and return back",
    244: "7 colors each in white run circularly, multi points from start to end and return back",
    245: "Flows gradually and circularly, 6 colors with red background from start to end",
    246: "Flows gradually and circularly, 6 colors with green background from start to end",
    247: "Flows gradually and circularly, 6 colors with blue background from start to end",
    248: "Flows gradually and circularly, 6 colors with yellow background from start to end",
    249: "Flows gradually and circularly, 6 colors with purple background from start to end",
    250: "Flows gradually and circularly, 6 colors with cyan background from start to end",
    251: "Flows gradually and circularly, 6 colors with white background from start to end",
    252: "Flows gradually and circularly, 6 colors with red background from end to start",
    253: "Flows gradually and circularly, 6 colors with green background from end to start",
    254: "Flows gradually and circularly, 6 colors with blue background from end to start",
    255: "Flows gradually and circularly, 6 colors with yellow background from end to start",
    256: "Flows gradually and circularly, 6 colors with purple background from end to start",
    257: "Flows gradually and circularly, 6 colors with cyan background from end to start",
    258: "Flows gradually and circularly, 6 colors with white background from end to start",
    259: "Flows gradually and circularly, 6 colors with red background from start to end and return back",
    260: "Flows gradually and circularly, 6 colors with green background from start to end and return back",
    261: "Flows gradually and circularly, 6 colors with blue background from start to end and return back",
    262: "Flows gradually and circularly, 6 colors with yellow background from start to end and return back",
    263: "Flows gradually and circularly, 6 colors with purple background from start to end and return back",
    264: "Flows gradually and circularly, 6 colors with cyan background from start to end and return back",
    265: "Flows gradually and circularly, 6 colors with white background from start to end and return back",
    266: "7 colors run with black background from start to end",
    267: "7 colors run with red background from start to end",
    268: "7 colors run with green background from start to end",
    269: "7 colors run with blue background from start to end",
    270: "7 colors run with yellow background from start to end",
    271: "7 colors run with purple background from start to end",
    272: "7 colors run with cyan background from start to end",
    273: "7 colors run with white background from start to end",
    274: "7 colors run with black background from end to start",
    275: "7 colors run with red background from end to start",
    276: "7 colors run with green background from end to start",
    277: "7 colors run with blue background from end to start",
    278: "7 colors run with yellow background from end to start",
    279: "7 colors run with purple background from end to start",
    280: "7 colors run with cyan background from end to start",
    281: "7 colors run with white background from end to start",
    282: "7 colors run with black background from start to end and return back",
    283: "7 colors run with red background from start to end and return back",
    284: "7 colors run with green background from start to end and return back",
    285: "7 colors run with blue background from start to end and return back",
    286: "7 colors run with yellow background from start to end and return back",
    287: "7 colors run with purple background from start to end and return back",
    288: "7 colors run with cyan background from start to end and return back",
    289: "7 colors run with white background from start to end and return back",
    290: "7 colors run gradually + 7 colors run in olivary",
    291: "7 colors run gradually + 7 colors change quickly",
    292: "7 colors run gradually + 7 colors flash",
    293: "7 colors run in olivary + 7 colors change quickly",
    294: "7 colors run in olivary + 7 colors flash",
    295: "7 colors change quickly + 7 colors flash",
    296: "7 colors run gradually + 7 colors run in olivary + 7 colors change quickly",
    297: "7 colors run gradually + 7 colors run in olivary + 7 colors flash",
    298: "7 colors run gradually + 7 colors change quickly + 7 colors flash",
    299: "7 colors run in olivary + 7 colors change quickly + 7 colors flash",
    300: "7 colors run gradually + 7 colors run in olivary + 7 colors change quickly + 7 color flash",
}
ORIGINAL_ADDRESSABLE_EFFECT_NAME_ID = {
    v: k for k, v in ORIGINAL_ADDRESSABLE_EFFECT_ID_NAME.items()
}


CHRISTMAS_ADDRESSABLE_EFFECT_ID_NAME = {
    1: "Random Jump Async",
    2: "Random Gradient Async",
    3: "Fill-in Red, Green",
    4: "Fill-in Green, Blue",
    5: "Fill-in Green, Yellow",
    6: "Fill-in Green, Cyan",
    7: "Fill-in Green, White",
    8: "Fill-in Green, Red",
    9: "Twinkle Red",
    10: "Twinkle Green",
    11: "Twinkle Yellow",
    12: "Twinkle Blue",
    13: "Twinkle Purple",
    14: "Twinkle Cyan",
    15: "Twinkle White",
    16: "Alternating Flash Red, Green, Blue",
    17: "Alternating Flash Red, Green",
    18: "Alternating Flash Green, Blue",
    19: "Alternating Flash Blue, Yellow",
    20: "Alternating Flash Yellow, Cyan",
    21: "Alternating Flash Cyan, Purple",
    22: "Alternating Flash Purple, White",
    23: "Wave Seven-Color",
    24: "Wave RGB",
    25: "Wave Red",
    26: "Wave Green",
    27: "Wave Yellow",
    28: "Wave Blue",
    29: "Wave Purple",
    30: "Wave Cyan",
    31: "Wave White",
    32: "Breathe Sync Seven-Color",
    33: "Breathe Sync RGB",
    34: "Breathe Sync Red, Green",
    35: "Breathe Sync Red, Blue",
    36: "Breathe Sync Green, Blue",
    37: "Breathe Sync Red",
    38: "Breathe Sync Green",
    39: "Breathe Sync Yellow",
    40: "Breathe Sync Blue",
    41: "Breathe Sync Purple",
    42: "Breathe Sync Cyan",
    43: "Breathe Sync White",
    44: "Fill-in and Reset Red, Green",
    45: "Fill-in and Reset Green, Blue",
    46: "Fill-in and Reset Blue, Yellow",
    47: "Fill-in and Reset Yellow, Cyan",
    48: "Fill-in and Reset Cyan, Purple",
    49: "Fill-in and Reset Purple, White",
    50: "Fill-in and Reset Red, Green, Blue, Yellow",
    51: "Fill-in and Reset Red, Blue, Green, White",
    52: "Fill-in and Reset Blue, White, Purple, Yellow",
    53: "Fill-in and Reset White, Purple, Cyan, Green",
    54: "Strobe Red, Green, Blue, Yellow, Cyan, Purple, White",
    55: "Strobe Green, Red, Blue, Yellow, Cyan, Purple, White",
    56: "Strobe Blue, Green, Red, Yellow, Cyan, Purple, White",
    57: "Strobe Yellow, Green, Blue, Red, Cyan, Purple, White",
    58: "Strobe Cyan, Green, Blue, Yellow, Red, Purple, White",
    59: "Strobe Purple, Green, Blue, Yellow, Cyan, Red, White",
    60: "Strobe White, Green, Blue, Yellow, Cyan, Purple, Red",
    61: "Strobe Red, Green",
    62: "Strobe Green, Blue",
    63: "Strobe Blue, Yellow",
    64: "Strobe Yellow, Cyan",
    65: "Strobe Cyan, Purple",
    66: "Strobe Purple, White",
    67: "Strobe Black, White",
    68: "Flash Sync Seven-color",
    69: "Flash Sync RGB",
    70: "Flash Sync Red",
    71: "Flash Sync Green",
    72: "Flash Sync Yellow",
    73: "Flash Sync Blue",
    74: "Flash Sync Purple",
    75: "Flash Sync Cyan",
    76: "Jump Sync Seven-color",
    77: "Jump Sync RGB",
    78: "Jump Sync Red",
    79: "Jump Sync Green",
    80: "Jump Sync Yellow",
    81: "Jump Sync Blue",
    82: "Jump Sync Purple",
    83: "Jump Sync Cyan",
    84: "Red Wave, Breathe Sync, Flash, Jump",
    85: "Green Wave, Breathe Sync, Flash, Jump",
    86: "Yellow Wave, Breathe Sync, Flash, Jump",
    87: "Blue Wave, Breathe Sync, Flash, Jump",
    88: "Purple Wave, Breathe Sync, Flash, Jump",
    89: "Cyan Wave, Breathe Sync, Flash, Jump",
    90: "White Wave, Breathe Sync, Flash, Jump",
    91: "Seven-color Wave, Breathe Sync",
    92: "Seven-color Breathe Sync, Flash Sync",
    93: "Seven-color Flash Sync, Jump Sync",
    94: "Seven-color Wave, Breathe Sync, Flash Sync, Jump Sync",
    95: "Overlap Red, Green, Blue",
    96: "Overlap Red, Green, Blue, Cyan, Purple, White",
    97: "Overlap Green, Blue, Cyan",
    98: "Overlap Blue, Cyan, Purple",
    99: "Overlap Cyan, Purple, White",
    100: "Overlap White, Black, Red",
}

CHRISTMAS_ADDRESSABLE_EFFECT_NAME_ID = {
    v: k for k, v in CHRISTMAS_ADDRESSABLE_EFFECT_ID_NAME.items()
}


class PresetPattern:
    _instance = None
    warm_flash = EFFECT_MAP_LEGACY_CCT[EFFECT_WARM_FLASH]
    cool_flash = EFFECT_MAP_LEGACY_CCT[EFFECT_COOL_FLASH]
    warm_gradual = EFFECT_MAP_LEGACY_CCT[EFFECT_WARM_GRADUAL]
    cool_gradual = EFFECT_MAP_LEGACY_CCT[EFFECT_COOL_GRADUAL]
    seven_color_cross_fade = EFFECT_MAP[EFFECT_COLORLOOP]
    red_gradual_change = EFFECT_MAP[EFFECT_RED_FADE]
    green_gradual_change = EFFECT_MAP[EFFECT_GREEN_FADE]
    blue_gradual_change = EFFECT_MAP[EFFECT_BLUE_FADE]
    yellow_gradual_change = EFFECT_MAP[EFFECT_YELLOW_FADE]
    cyan_gradual_change = EFFECT_MAP[EFFECT_CYAN_FADE]
    purple_gradual_change = EFFECT_MAP[EFFECT_PURPLE_FADE]
    white_gradual_change = EFFECT_MAP[EFFECT_WHITE_FADE]
    red_green_cross_fade = EFFECT_MAP[EFFECT_RED_GREEN_CROSS_FADE]
    red_blue_cross_fade = EFFECT_MAP[EFFECT_RED_BLUE_CROSS_FADE]
    green_blue_cross_fade = EFFECT_MAP[EFFECT_GREEN_BLUE_CROSS_FADE]
    seven_color_strobe_flash = EFFECT_MAP[EFFECT_COLORSTROBE]
    red_strobe_flash = EFFECT_MAP[EFFECT_RED_STROBE]
    green_strobe_flash = EFFECT_MAP[EFFECT_GREEN_STROBE]
    blue_strobe_flash = EFFECT_MAP[EFFECT_BLUE_STROBE]
    yellow_strobe_flash = EFFECT_MAP[EFFECT_YELLOW_STROBE]
    cyan_strobe_flash = EFFECT_MAP[EFFECT_CYAN_STROBE]
    purple_strobe_flash = EFFECT_MAP[EFFECT_PURPLE_STROBE]
    white_strobe_flash = EFFECT_MAP[EFFECT_WHITE_STROBE]
    seven_color_jumping = EFFECT_MAP[EFFECT_COLORJUMP]
    cycle_rgb = EFFECT_MAP_DIMMABLE[EFFECT_CYCLE_RGB]
    cycle_seven_colors = EFFECT_MAP_DIMMABLE[EFFECT_CYCLE_SEVEN_COLORS]
    red_green_blue_cross_fade = EFFECT_MAP_DIMMABLE[EFFECT_RED_GREEN_BLUE_CROSS_FADE]

    def __init__(self) -> None:
        self._value_to_str: dict[int, str] = {
            **EFFECT_ID_NAME_LEGACY_CCT,
            **{
                v: k.replace("_", " ").title()
                for k, v in PresetPattern.__dict__.items()
                if type(v) is int
            },
        }
        self._hex_str_valid_values = {f"0x{byte:02X}" for byte in self._value_to_str}

    @classmethod
    def instance(cls) -> PresetPattern:
        """Get preset pattern instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def valid(pattern: int) -> bool:
        instance = PresetPattern.instance()
        return pattern in instance._value_to_str

    @staticmethod
    def valid_or_raise(pattern: int) -> None:
        instance = PresetPattern.instance()
        if pattern not in instance._value_to_str:
            raise ValueError(f"Pattern must be one of {instance._hex_str_valid_values}")

    @staticmethod
    def valtostr(pattern: int) -> str | None:
        instance = PresetPattern.instance()
        return instance._value_to_str.get(pattern)

    @staticmethod
    def str_to_val(effect: str) -> int:
        if effect in EFFECT_MAP_DIMMABLE:
            return EFFECT_MAP_DIMMABLE[effect]
        mapped_effect = effect.replace(" ", "_").lower()
        if hasattr(PresetPattern, mapped_effect):
            return cast("int", getattr(PresetPattern, mapped_effect))
        raise ValueError(f"{effect} is not a known effect name.")
