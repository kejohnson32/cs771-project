# PyNIMS

Python client and CLI for interacting with the USGS NIMS (National Imagery Management System) API.

---

## Features

- Query camera metadata
- List available images within a time range
- Download images by name or for a full time window
- CLI for scripting and automation
- Workflows for high-level tasks like downloading image batches

---

## Installation Options

There are two main ways to use `pynims`:

1. [Clone the repository and install locally](#1️⃣-clone-the-repo-and-install-locally) — ideal for development
2. [Install directly from USGS GitLab](#2️⃣-install-directly-from-usgs-gitlab-no-cloning) — ideal for use as a package

Each method supports either `uv` or `pip`.

---

### Prerequisites

You must have **Python 3.6+** installed.

### To use `uv` (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### To use `pip`:
```bash
python -m ensurepip --upgrade
python -m pip install --upgrade pip
```
---

## 1️⃣ Clone the Repo and Install Locally

Use this method if you want to contribute, modify, or explore the source code.

### 🔧 Using `uv`

```bash
git clone https://code.usgs.gov/usgs-imagery-packages/pynims.git  
cd pynims

uv venv                # optional: creates a local .venv  
uv sync                # installs dependencies from pyproject.toml

source .venv/bin/activate        # macOS/Linux  
.venv\Scripts\activate           # Windows
```

### 🔧 Using `pip`

```bash
git clone https://code.usgs.gov/usgs-imagery-packages/pynims.git  
cd pynims

python -m venv .venv  
source .venv/bin/activate        # macOS/Linux  
.venv\Scripts\activate           # Windows

pip install -e .                 # install in editable mode
```
> 📌 `-e .` lets you make live changes to the code without reinstalling.

---

## 2️⃣ Install Directly from USGS GitLab (No Cloning)

Use this if you just want to install `pynims` as a package in your own project.

### 🚀 Using `uv`

    uv pip install git+https://code.usgs.gov/usgs-imagery-packages/pynims.git 

### 🚀 Using `pip`

    pip install git+https://code.usgs.gov/usgs-imagery-packages/pynims.git 

> This installs the latest version of `pynims` from the `main` branch.

---
## ✅ Verifying the Installation

To check that `pynims` is installed:

    python -c "import pynims; print(pynims.__version__)"

Or, using the CLI:

    pynims --help

---

## Usage
### As a Python package
```python
from pynims.client import NIMSClient

client = NIMSClient()
cameras = client.get_cameras()
print(cameras)
```
### As a CLI tool

After installing, you'll have access to the nims CLI:

```bash
pynims cameras       # lists all camera IDs
pynims image-list --camera-id=cam123 --start=2023-06-01 --end=2023-06-10
pynims download-images --camera-id=cam123 --start=2023-06-01 --end=2023-06-10
```
Use --help to explore options:

```bash
pynims --help
pynims save-images --help
```