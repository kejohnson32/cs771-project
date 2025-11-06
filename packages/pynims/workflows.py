from pathlib import Path
from typing import Optional, Union
from datetime import datetime
import json

from pynims.client import NIMSClient
from pynims.utils import get_nims_image_timestamp


def get_camera_list(ids_only: bool = True):
    """Get list of all cameras"""
    client = NIMSClient()
    cameras = client.get_cameras()
    return [d["camId"] for d in cameras] if ids_only else cameras


def save_image_list_to_file(
    camera_id: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    recursive: Optional[bool] = None,
    max_results: Optional[int] = None,
    save_dir: Union[str, Path] = "image_lists",
) -> None:
    """Save image list to a file."""

    client = NIMSClient()
    image_list = client.get_image_list(camera_id, start, end, recursive, max_results)

    first_image_time = get_nims_image_timestamp(image_list[0])
    last_image_time = get_nims_image_timestamp(image_list[-1])

    file_name = f"{camera_id}_{first_image_time}---{last_image_time}.json"
    target_dir = Path(save_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(target_dir).joinpath(file_name)

    Path(output_path).write_text(json.dumps(image_list, indent=2))
    client.close()


def download_images_for_camera(
    camera_id: str,
    start: Optional[Union[str, datetime]] = None,
    end: Optional[Union[str, datetime]] = None,
    recursive: Optional[bool] = None,
    max_results: Optional[int] = None,
    save_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Get images for a camera and download them all."""
    with NIMSClient() as client:
        image_list = client.get_image_list(
            camera_id, start, end, recursive, max_results
        )
        if len(image_list) > 0:
            for idx, image in enumerate(image_list):
                print(f"image # {idx + 1} of {len(image_list)}")
                client.download_image(image, save_dir)
