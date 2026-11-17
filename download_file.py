"""Download public project data with resumable partial files."""

import time
import urllib.request
from pathlib import Path


def download_file(url, destination, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if expected_size is not None and destination.stat().st_size != expected_size:
            raise ValueError(f"Unexpected size for existing file {destination}")
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                append = offset > 0 and response.status == 206
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise ValueError("Download resumed at an unexpected offset")
                content_length = response.headers.get("Content-Length")
                response_size = int(content_length) if content_length else None
                received = 0
                with partial.open("ab" if append else "wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        received += len(chunk)
                if response_size is not None and received != response_size:
                    raise OSError(f"Incomplete response for {url}")
            if expected_size is not None and partial.stat().st_size != expected_size:
                raise OSError(f"Incomplete download of {destination}")
            partial.replace(destination)
            print(f"Downloaded {destination} ({destination.stat().st_size} bytes)", flush=True)
            return destination
        except (OSError, TimeoutError) as error:
            if attempt == 4:
                raise
            print(f"Download retry {attempt + 1} after {error}", flush=True)
            time.sleep(2)
    raise RuntimeError("Download attempts exhausted")
