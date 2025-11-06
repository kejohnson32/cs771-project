import typer
from typing import Optional
from pathlib import Path

from pynims.workflows import (
    get_camera_list,
    save_image_list_to_file,
    download_images_for_camera,
)

app = typer.Typer(help="NIMS Workflow CLI")


@app.command()
def cameras(ids_only: bool = True):
    """List cameras (IDs or full info)."""
    result = get_camera_list(ids_only=ids_only)
    for item in result:
        typer.echo(item)


@app.command()
def image_list(
    camera_id: str,
    start: Optional[str] = typer.Option(None, help="Start datetime (ISO 8601)"),
    end: Optional[str] = typer.Option(None, help="End datetime (ISO 8601)"),
    recursive: Optional[bool] = typer.Option(None, help="Use recursive fetching"),
    max_results: Optional[int] = typer.Option(
        None, help="Max number of images to include in list"
    ),
    save_dir: Path = typer.Option("image_lists", help="Directory to save image list"),
):
    """Fetch and save image list for a camera."""
    save_image_list_to_file(
        camera_id=camera_id,
        start=start,
        end=end,
        recursive=recursive,
        max_results=max_results,
        save_dir=save_dir,
    )
    typer.echo(f"Saved image list for {camera_id} to {save_dir}")


@app.command()
def download_images(
    camera_id: str,
    start: Optional[str] = typer.Option(None, help="Start datetime (ISO 8601)"),
    end: Optional[str] = typer.Option(None, help="End datetime (ISO 8601)"),
    recursive: bool = typer.Option(False, help="Use recursive fetching"),
    max_results: Optional[int] = typer.Option(
        None, help="Max number of images to download"
    ),
    save_dir: Optional[Path] = typer.Option(
        None, help="Parent directory to save images"
    ),
):
    """Download images for a camera."""
    download_images_for_camera(
        camera_id=camera_id,
        start=start,
        end=end,
        recursive=recursive,
        max_results=max_results,
        save_dir=save_dir,
    )
    typer.echo(f"Downloaded images for {camera_id}")


if __name__ == "__main__":
    app()
