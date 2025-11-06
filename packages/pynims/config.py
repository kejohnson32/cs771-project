# ENDPOINTS
NIMS_PROD_CAMERA_BASE_URL = (
    "https://jj5utwupk5.execute-api.us-east-1.amazonaws.com/prod/"
)
NIMS_DEV_CAMERA_BASE_URL = "https://wnzcqxlz38.execute-api.us-east-1.amazonaws.com/dev/"
NIMS_IMAGE_BASE_URL = "https://usgs-nims-images.s3.amazonaws.com/"

# IMAGE LIST REQUEST CONSTANTS
NIMS_IMAGE_LIST_LIMIT = 1000
NIMS_DEFAULT_OLDEST_IMAGE_TIME = "2000-01-01T00:00Z"

# HTTP REQUEST CONSTANTS
DEFAULT_HTTP_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
DEFAULT_SLEEP_MULTIPLIER = 2
