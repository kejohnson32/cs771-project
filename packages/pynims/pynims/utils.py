import os
import pytz
from datetime import datetime, timezone, timedelta
from pathlib import Path


##### NIMS image path / date time conversion utilities
def get_cam_id_from_nims_image_name(filename):
    imageName = os.path.basename(filename)
    imageName = os.path.splitext(filename)[0]  # remove file ext
    return imageName.split("___")[0]  # separate timestamp from name


def get_nims_image_timestamp(filename):
    imageName = os.path.basename(filename)
    imageName = os.path.splitext(filename)[0]  # remove file ext
    return imageName.split("___")[1]  # separate timestamp from name


def get_nims_image_name_from_date_time_and_cam_id(dateTime, camId):
    if dateTime is None or camId is None:
        raise ValueError("Expecting dateTime and camId")
    if not isinstance(dateTime, datetime):
        raise TypeError("Expecting dateTime to be of type 'datetime'")
    if not isinstance(camId, str):
        raise TypeError("Expecting camId to be of type 'str'")
    if dateTime.tzinfo != "UTC" and dateTime.tzinfo is not None:
        dateTime = dateTime.astimezone(timezone.utc)
    dateTime = (
        dateTime.isoformat(timespec="seconds").replace("+00:00", "Z").replace(":", "-")
    )
    return f"{camId}___{dateTime}.jpg"


def get_image_time_from_nims_image_name(filename):
    imageName = os.path.basename(filename)
    imageName = os.path.splitext(filename)[0]  # remove file ext
    nameSplit = imageName.split("___")[1]  # separate timestamp from name
    date, time = nameSplit.split("T")  # separate date and time from timestamp
    time = time[:-1]  # remove Z from time
    h, m, s = time.split("-")  # separate hours minutes seconds
    dateTimeString = date + " " + h + ":" + m + ":" + s  # set dateTimeString
    return dateTimeString


def convert_nims_image_name_to_utc_date(imageName):
    dateTimeString = get_image_time_from_nims_image_name(imageName)
    imgDate = datetime.strptime(dateTimeString, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc
    )
    return imgDate


##### date conversion utilities
def convert_utc_date_to_tz_date(utcDate, tz):
    return utcDate.astimezone(pytz.timezone(tz))


def convert_tz_date_to_utc_date(tzDate):
    return tzDate.astimezone(pytz.utc)


##### Image List utilities
def get_downloaded_image_list_within_directory(directory):
    return [p.name for p in Path(directory).iterdir() if p.is_file()]


def get_downloaded_image_list_for_cam_id(camId, downloadedImageList):
    return [img for img in downloadedImageList if camId in img]


def compare_image_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    commonFilenames = list(set1.intersection(set2))
    uniqueList1 = list(set1.difference(set2))
    uniqueList2 = list(set2.difference(set1))
    return commonFilenames, uniqueList1, uniqueList2


##### Basic file path utilities
def get_folder_images(folder_path, ext=(".JPG", ".jpg")):
    if len(os.listdir(folder_path)) == 0:
        return None
    imgs = []
    for file in os.listdir(folder_path):
        if file.endswith(ext):
            file_path = folder_path + file
            if os.path.getsize(file_path) > 0:
                imgs.append(folder_path + file)
    if len(imgs) == 0:
        return None
    return imgs


##### Exif conversion utilities
def convert_image_name_to_exif_date(imageName, tz, format):
    if format.lower() == "nims":
        utcDate = convert_nims_image_name_to_utc_date(imageName)
        tzDate = convert_utc_date_to_tz_date(utcDate, tz)
        exifDate = convert_tz_date_to_exif_date(tzDate)
    if format.lower() == "tl1":
        utcDate = get_utc_date_from_tl1_image_name(imageName)
        tzDate = convert_utc_date_to_tz_date(utcDate, tz)
        exifDate = convert_tz_date_to_exif_date(tzDate)
    if format.lower() == "vivotek":
        exifDate = get_exif_date_from_vivotek_image_name(imageName)
    if format.lower() == "rise":
        exifDate = get_exif_date_from_rise_image_name(imageName)
    return exifDate


def get_utc_date_from_tl1_image_name(imageName):
    imageName = os.path.basename(imageName)
    imageName = os.path.splitext(imageName)[0]  # remove file ext
    nameSplit = imageName.split("___")[1]  # separate timestamp from name
    tz = nameSplit[-5:]
    tzSign = nameSplit[-6]
    tzOffsetMinutes = int(tz[1]) * 60 + int(tz[-2:])
    if tzSign == "+":
        tzOffsetMinutes = tzOffsetMinutes * -1
    dt = nameSplit[:-6]
    date, time = dt.split("_")  # separate date and time from timestamp
    y, mo, d = date.split("-")
    h, m, s, ms = time.split("-")  # separate hours minutes seconds
    dt = datetime(int(y), int(mo), int(d), int(h), int(m), int(s)) + timedelta(
        minutes=tzOffsetMinutes
    )
    utcDate = dt.replace(tzinfo=timezone.utc)
    return utcDate


def get_exif_date_from_vivotek_image_name(imageName):
    # format: YYYmmdd_HHMMSS.jpg
    # Note: no tz in filename. Need to verify time in filename is in appropriate tz.
    imageName = os.path.basename(imageName)
    imageName = os.path.splitext(imageName)[0]  # remove file ext
    nameSplit = imageName.split("_")  # separate timestamp from name
    date = nameSplit[len(nameSplit) - 2]  # separate date from timestamp
    time = nameSplit[len(nameSplit) - 1]  # separate time from timestamp
    dIdxs = [0, 4, 6]  # date split indices for format: yyyymmdd
    dSplit = [date[i:j] for i, j in zip(dIdxs, dIdxs[1:] + [None])]  # split date
    date = dSplit[0] + ":" + dSplit[1] + ":" + dSplit[2]  # set date string
    tIdxs = [0, 2, 4]  # time split indices for format: hhmmss
    tSplit = [time[i:j] for i, j in zip(tIdxs, tIdxs[1:] + [None])]  # split time
    time = tSplit[0] + ":" + tSplit[1] + ":" + tSplit[2]  # set time string
    exifDate = date + " " + time  # set dateTimeString
    return exifDate


def get_exif_date_from_rise_image_name(imageName):
    # format: stationId_YYYYmmdd-HHMMSS.jpg
    # Note: no tz in filename. Need to verify time in filename is in appropriate tz.
    imageName = os.path.basename(imageName)
    imageName = os.path.splitext(imageName)[0]  # remove file ext
    stationId, dateTime = imageName.split("_")  # separate timestamp from name
    date, time = dateTime.split("-")
    dIdxs = [0, 4, 6]  # date split indices for format: yyyymmdd
    dSplit = [date[i:j] for i, j in zip(dIdxs, dIdxs[1:] + [None])]  # split date
    date = dSplit[0] + ":" + dSplit[1] + ":" + dSplit[2]  # set date string
    tIdxs = [0, 2, 4]  # time split indices for format: hhmmss
    tSplit = [time[i:j] for i, j in zip(tIdxs, tIdxs[1:] + [None])]  # split time
    time = tSplit[0] + ":" + tSplit[1] + ":" + tSplit[2]  # set time string
    exifDate = date + " " + time  # set dateTimeString
    return exifDate


def convert_exif_date_to_tz_date(exifDate, tz):
    imageDate = datetime.strptime(exifDate, "%Y:%m:%d %H:%M:%S")
    imageDateTz = pytz.timezone(tz).localize(imageDate)
    return imageDateTz


def convert_tz_date_to_exif_date(tzDate):
    return tzDate.strftime("%Y:%m:%d %H:%M:%S")
