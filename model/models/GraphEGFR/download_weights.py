"""Download the published GraphEGFR state dictionaries from Zenodo."""

import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from urllib.request import urlopen


ARCHIVE_URL = "https://zenodo.org/records/11122146/files/GraphEGFR.tar.gz"
ARCHIVE_MD5 = "94b053a5e067a73b60ecf460e90b8a7d"
WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "data" / "weights" / "graphegfr"
CHUNK_SIZE = 1024 * 1024


def download_archive(archive_path):
    digest = hashlib.md5(usedforsecurity=False)
    with urlopen(ARCHIVE_URL, timeout=60) as response, archive_path.open("wb") as output:
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            digest.update(chunk)
            output.write(chunk)

    if digest.hexdigest() != ARCHIVE_MD5:
        raise ValueError("Archive checksum mismatch")


def extract_weights(archive_path, staging_dir):
    weights = []
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.parent.name != "state_dict" or path.suffix != ".pt":
                continue

            # Use only the filename so archive paths cannot escape the staging directory.
            if "\\" in path.name or ":" in path.name:
                raise ValueError(f"Invalid weight filename: {path.name}")

            destination = staging_dir / path.name
            with archive.extractfile(member) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, CHUNK_SIZE)
            weights.append(destination)

    if not weights:
        raise ValueError("No state_dict/*.pt files found in the archive")
    return weights


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading GraphEGFR archive from Zenodo (2.9 GB)...", flush=True)
    try:
        with TemporaryDirectory(dir=WEIGHTS_DIR) as temporary_dir:
            staging_dir = Path(temporary_dir)
            archive_path = staging_dir / "GraphEGFR.tar.gz"
            download_archive(archive_path)
            weights = extract_weights(archive_path, staging_dir)
            for weight in weights:
                weight.replace(WEIGHTS_DIR / weight.name)
    except (OSError, EOFError, tarfile.TarError, ValueError) as error:
        raise SystemExit(f"Could not install GraphEGFR weights: {error}") from None

    print(f"Saved {len(weights)} weight files to {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
