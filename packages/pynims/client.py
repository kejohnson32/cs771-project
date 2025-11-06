import httpx
import time
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
from datetime import datetime, timezone
from dateutil.parser import parse
import pytz

from .config import (
    NIMS_PROD_CAMERA_BASE_URL,
    NIMS_DEV_CAMERA_BASE_URL,
    NIMS_IMAGE_BASE_URL,
    NIMS_IMAGE_LIST_LIMIT,
    NIMS_DEFAULT_OLDEST_IMAGE_TIME,
    DEFAULT_HTTP_TIMEOUT,
    DEFAULT_RETRIES,
    DEFAULT_SLEEP_MULTIPLIER,
)

from .utils import (
    get_cam_id_from_nims_image_name,
    convert_nims_image_name_to_utc_date,
)


class NIMSClient:
    """A client for interacting with the USGS NIMS (National Imagery Management System) API."""

    def __init__(self, env: str = "prod", timeout: float = DEFAULT_HTTP_TIMEOUT):
        """Initialize the NIMSClient."""

        if env not in ["prod", "dev"]:
            raise ValueError("env must be 'prod' or 'dev'")

        self.camera_base_url = (
            NIMS_PROD_CAMERA_BASE_URL if env == "prod" else NIMS_DEV_CAMERA_BASE_URL
        )
        self.image_base_url = NIMS_IMAGE_BASE_URL
        self.client = httpx.Client(timeout=timeout)
        self.default_image_limit = NIMS_IMAGE_LIST_LIMIT

    def get_cameras(self):
        """Retrieve a list of all cameras."""
        url = f"{self.camera_base_url}cameras"
        return self._make_request(url)

    def get_camera(self, camera_id: str):
        """Retrieve a specific camera by its ID."""
        url = f"{self.camera_base_url}cameras"
        params = {"camId": camera_id}
        response = self._make_request(url, params=params)
        if response and len(response) > 0:
            return response[0]
        raise ValueError(f"Camera with ID '{camera_id}' not found")

    def get_camera_attribute(self, camera_id=None, attr=None):
        """Retrieve a specific attribute of a camera by its ID."""
        if camera_id is None or attr is None:
            raise ValueError("Expecting a camera_id and an attribute retrieve")
        return self.get_camera(camera_id)[attr]

    def get_image_list(
        self,
        camera_id: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        recursive: Optional[bool] = None,
        max_results: Optional[int] = None,
    ) -> List[str]:
        """Get a list of images for a camera within a time range."""
        # Validate inputs
        if not camera_id:
            raise ValueError("camera_id is required")
        if not all(
            isinstance(obj, (str, datetime)) or obj is None for obj in [start, end]
        ):
            raise TypeError(
                "expecting type 'str' or 'datetime.datetime' for 'after' and 'before'"
            )

        # Set default recursive behavior
        if recursive is None:
            recursive = start is not None and end is not None

        if recursive:
            return self._fetch_image_list_recursively(
                camera_id, start, end, max_results, []
            )
        else:
            return self._fetch_image_list_single(camera_id, start, end, max_results)

    def _fetch_image_list_single(
        self,
        camera_id: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        max_results: Optional[int] = None,
    ) -> List[str]:
        url = f"{self.camera_base_url}listFiles"
        params = {"camId": camera_id}
        params["recent"] = "false" if start else "true"
        params["limit"] = max_results or self.default_image_limit
        if start:
            start = self._format_date_range_input(start, camera_id)
            params["after"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")
        if end:
            end = self._format_date_range_input(end, camera_id)
            params["before"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")
        image_list = self._make_request(url, params)
        return sorted(image_list)

    def _fetch_image_list_recursively(
        self,
        camera_id: str,
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None,
        max_results: Optional[int] = None,
        whole_image_list: Optional[List[str]] = None,
    ) -> List[str]:
        if whole_image_list is None:
            whole_image_list = []

        url = f"{self.camera_base_url}listFiles"
        params = {"camId": camera_id, "recent": "false"}

        if not start:
            start = NIMS_DEFAULT_OLDEST_IMAGE_TIME
        start = self._format_date_range_input(start, camera_id)
        params["after"] = start.strftime("%Y-%m-%dT%H:%M:%SZ")

        if not end:
            end = datetime.now(timezone.utc)
        end = self._format_date_range_input(end, camera_id)
        params["before"] = end.strftime("%Y-%m-%dT%H:%M:%SZ")

        image_list = self._make_request(url, params)

        # Stop if there are no new images
        if len(image_list) <= 1:
            return whole_image_list

        latest_image_timestamp = convert_nims_image_name_to_utc_date(image_list[-1])

        # Extend results
        if whole_image_list:
            whole_image_list.extend(image_list[1:])
        else:
            whole_image_list = image_list

        if max_results is not None and len(whole_image_list) >= max_results:
            return whole_image_list[:max_results]

        # Continue recursively
        return self._fetch_image_list_recursively(
            camera_id=camera_id,
            start=latest_image_timestamp,
            end=end,
            max_results=max_results,
            whole_image_list=whole_image_list,
        )

    def _format_date_range_input(self, datetimeInput, cameraId):
        if isinstance(datetimeInput, str):
            datetimeInput = parse(datetimeInput)
        if (
            datetimeInput.tzinfo is None
        ):  # assume if user doesn't pass any tz info, they want to query in local camera time
            tz = pytz.timezone(self.get_camera_attribute(cameraId, "tz"))
            datetimeInput = tz.localize(datetimeInput)
            datetimeInput = datetimeInput.astimezone(
                pytz.utc
            )  # after setting to camera tz, convert to utc
        return datetimeInput

    def download_image(self, image_name: str, save_dir: Optional[str] = None):
        """Download a NIMS image by its name and save it to the specified directory."""
        # Set save_dir to default if None provided
        if save_dir is None:
            save_dir = "downloaded_images/"

        # Create directory if it doesn't exist
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / image_name
        if save_path.exists():
            print(f"{image_name} already exists -- skipping")
            return save_path

        camera_id = get_cam_id_from_nims_image_name(image_name)
        url = f"{self.image_base_url}overlay/{camera_id}/{image_name}"

        # Download and save
        response = self.client.get(url)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return save_path  # Return where it was saved

    def _make_request(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        max_retries: int = DEFAULT_RETRIES,
    ) -> Dict[str, Any]:
        for attempt in range(max_retries + 1):
            try:
                resp = self.client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()

            except httpx.TimeoutException:
                if attempt == max_retries:
                    raise
                time.sleep(
                    DEFAULT_SLEEP_MULTIPLIER**attempt
                )  # Exponential backoff: 1s, 2s, 4s

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500 and attempt < max_retries:
                    time.sleep(
                        DEFAULT_SLEEP_MULTIPLIER**attempt
                    )  # Retry on server errors
                    continue
                raise  # Don't retry on client errors (4xx)

            except httpx.RequestError:
                if attempt == max_retries:
                    raise
                time.sleep(DEFAULT_SLEEP_MULTIPLIER**attempt)

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
