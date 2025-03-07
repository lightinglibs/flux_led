#!/usr/bin/env python

"""
This is an example script that can be used to set on and off timers based on the sunrise/sunset times.

Specifically, it will set times on an outside porch light
to turn on at dusk and off at dawn.  It will set the timers for
inside light to turn on at sunset, and off at a fixed time.

A script like this is best used with an /etc/crontab entry that might
run every day or every few days. For example:
-----------------

# Sync up the bulb clocks a few times a day, in case of manual power toggles
00 3,12,17,22 * * * username /path/to/scripts/flux_led.py -Ss --setclock

# Set the sun timers everyday at 3am
00 3 * * * username /path/to/scripts/sun_timers.py


-----------------

The python file with the Flux LED wrapper classes should live in
the same folder as this script
"""

import datetime
import os
import sys
import syslog

from flux_led import BulbScanner, LedTimer, WifiLedBulb

try:
    from astral import Astral
except ImportError:
    print("Error:  Need to install python package: astral")
    sys.exit(-1)


this_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_folder)

debug = False


def main():
    syslog.openlog(sys.argv[0])

    # Change location to nearest city.
    location = "San Diego"

    # Get the local sunset/sunrise times
    a = Astral()
    a.solar_depression = "civil"
    city = a[location]
    timezone = city.timezone
    sun = city.sun(date=datetime.datetime.now(), local=True)

    if debug:
        print(f"Information for {location}/{city.region}\n")
        print(f"Timezone: {timezone}")

        print(f"Latitude: {city.latitude:.02f}; Longitude: {city.longitude:.02f}\n")

        print("Dawn:    {}".format(sun["dawn"]))
        print("Sunrise: {}".format(sun["sunrise"]))
        print("Noon:    {}".format(sun["noon"]))
        print("Sunset:  {}".format(sun["sunset"]))
        print("Dusk:    {}".format(sun["dusk"]))

    # Find the bulbs on the LAN
    scanner = BulbScanner()
    scanner.scan(timeout=4)

    # Specific ID/MAC of the bulbs to set
    porch_info = scanner.getBulbInfoByID("ACCF235FFFEE")
    livingroom_info = scanner.getBulbInfoByID("ACCF235FFFAA")

    if porch_info:
        bulb = WifiLedBulb(porch_info["ipaddr"])
        bulb.refreshState()

        timers = bulb.getTimers()

        # Set the porch bulb to turn on at dusk using timer idx 0
        syslog.syslog(
            syslog.LOG_ALERT,
            "Setting porch light to turn on at {}:{:02d}".format(
                sun["dusk"].hour, sun["dusk"].minute
            ),
        )
        dusk_timer = LedTimer()
        dusk_timer.setActive(True)
        dusk_timer.setRepeatMask(LedTimer.Everyday)
        dusk_timer.setModeWarmWhite(35)
        dusk_timer.setTime(sun["dusk"].hour, sun["dusk"].minute)
        timers[0] = dusk_timer

        # Set the porch bulb to turn off at dawn using timer idx 1
        syslog.syslog(
            syslog.LOG_ALERT,
            "Setting porch light to turn off at {}:{:02d}".format(
                sun["dawn"].hour, sun["dawn"].minute
            ),
        )
        dawn_timer = LedTimer()
        dawn_timer.setActive(True)
        dawn_timer.setRepeatMask(LedTimer.Everyday)
        dawn_timer.setModeTurnOff()
        dawn_timer.setTime(sun["dawn"].hour, sun["dawn"].minute)
        timers[1] = dawn_timer

        bulb.sendTimers(timers)

    else:
        print("Can't find porch bulb")

    if livingroom_info:
        bulb = WifiLedBulb(livingroom_info["ipaddr"])
        bulb.refreshState()

        timers = bulb.getTimers()

        # Set the living room bulb to turn on at sunset using timer idx 0
        syslog.syslog(
            syslog.LOG_ALERT,
            "Setting LR light to turn on at {}:{:02d}".format(
                sun["sunset"].hour, sun["sunset"].minute
            ),
        )
        sunset_timer = LedTimer()
        sunset_timer.setActive(True)
        sunset_timer.setRepeatMask(LedTimer.Everyday)
        sunset_timer.setModeWarmWhite(50)
        sunset_timer.setTime(sun["sunset"].hour, sun["sunset"].minute)
        timers[0] = sunset_timer

        # Set the living room bulb to turn off at a fixed time
        off_timer = LedTimer()
        off_timer.setActive(True)
        off_timer.setRepeatMask(LedTimer.Everyday)
        off_timer.setModeTurnOff()
        off_timer.setTime(23, 30)
        timers[1] = off_timer

        bulb.sendTimers(timers)
    else:
        print("Can't find living room bulb")


if __name__ == "__main__":
    main()
