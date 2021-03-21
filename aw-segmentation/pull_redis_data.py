#!/usr/bin/env python3
import requests
import urllib
import re
import os
# pyquery is also required if you want to scrape the redis list
BASE_URL = "http://67.58.52.158:8081/frame"
from datetime import datetime, timezone, timedelta
valid_ts = re.compile("([0-9]+(?:[.][0-9]+)?)[.]jpg")


def yjh2ts(y, j, h):
    "Convert year/julian-day/hour to a unix timestamp"
    s = "/".join(map(str, [y, j, h]))
    d = datetime.strptime(s, '%Y/%j/%H').replace(tzinfo=timezone.utc)
    return d.timestamp()


def get_frame_sequence(camera, timestamp, count):
    """
    Connects to the server to fetch 'count' sequential frames
    from camera starting at the approximate timestamp provided.  Returns a list of [(timestamp, bytes)] where timestamp is a string
    like "1602273600.197393968" and bytes contains the bytes of the JPEG.
    """
    results = []
    while len(results) < count:
        args = {
            "id": camera,
            "timestamp": timestamp,
            "direction": "after"}
        if len(results) == 0:
            # don't skip to next frame on first request      
            del args["direction"]

        params = urllib.parse.urlencode(args)      
        r = requests.get(BASE_URL + "?" + params)
        r.raise_for_status()
        timestamp = valid_ts.match(r.url.rsplit("/")[-1]).groups()[0]
        results.append((timestamp, r.content))
        return results


def get_list_of_all_cameras():
    r = requests.get("http://67.58.52.158:8081/camera_sites")
    result = [x["name"] for x in r.json()]
    result.sort()
    return result


def get_list_of_redis_cameras():
    import pyquery
    r = requests.get("http://67.58.52.158/fireframes3/redis/")
    pq = pyquery.PyQuery(r.content)
    result = [x.text.replace("/", "")
              for x in pq("a") if x.text.startswith("Axis-")]
    result.sort()
    return result


def run_fetch(year, month, day):
    cameras = get_list_of_redis_cameras()
    for hour in range(8, 4+12+1):
        hour_ts = datetime(year, month, day, hour).timestamp()
        for camera in cameras:
            print("Fetching (hour, camera):", (hour, camera))
            try:
                frames = get_frame_sequence(camera, hour_ts, 20)
                path = os.path.join(
                    "/data/field/AlertWildfire/redis_nonsmoke_dataset/", str(year)+str(month).zfill(2)+str(day).zfill(2)+str(hour).zfill(2))
                os.makedirs(path, exist_ok=True)
                for ts, frame in frames:
                    out_fname = camera + "-" + ts + ".jpg"
                    with open(os.path.join(path, out_fname), "wb") as f:
                        f.write(frame)
            except requests.exceptions.HTTPError as e:
                print("  ^^ failed:", e)


if __name__ == '__main__':

    year = 2020
    for julian_days in range(236, 335+1):
        d = datetime(year, 1, 1) + timedelta(days=julian_days - 1)
        run_fetch(d.year, d.month, d.day)
        print("{}/{}\r".format(julian_days,335), end='')
