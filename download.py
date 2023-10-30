"""Utility for downloading gladostts models"""
import argparse
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Union
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import urlopen

DEFAULT_URL = "https://github.com/nalf3in/glados-tts/releases/download/v0.1.0-alpha/{file}"
DEFAULT_MODEL_DIR = "./models"

_LOGGER = logging.getLogger(__name__)


def _quote_url(url: str) -> str:
    """Quote file part of URL in case it contains UTF-8 characters."""
    parts = list(urlsplit(url))
    parts[2] = quote(parts[2])
    return urlunsplit(parts)


def get_file_hash(path: Union[str, Path], bytes_per_chunk: int = 8192) -> str:
    """Hash a file in chunks using md5."""
    path_hash = hashlib.md5()
    with open(path, "rb") as path_file:
        chunk = path_file.read(bytes_per_chunk)
        while chunk:
            path_hash.update(chunk)
            chunk = path_file.read(bytes_per_chunk)

    return path_hash.hexdigest()


def ensure_model_exists(download_dir: Union[str, Path]):
    download_dir = Path(download_dir)

    # Define the list of model files and their checksums
    model_files = [
        {"filename": "glados-new.pt", "md5": "d6945ffd96ee0619d0d49a581b5b83ad"},
        {"filename": "glados.pt", "md5": "11383a00f7ddfc8f80285ce3aba2ebb0"},
        {"filename": "en_us_cmudict_ipa_forward.pt", "md5": "33887f7f579f010ce4463534306120b0"},
        {"filename": "emb/glados_p2.pt", "md5": "ff2ad1438e9acb1f8e8607864c239ffc"},
        {"filename": "emb/glados_p1.pt", "md5": "e0ffe67a6f53c4ff0b3952fc678946d9"},
        {"filename": "vocoder-gpu.pt", "md5": "d35c13c01d2cacd348aa216649bbfac3"},
        {"filename": "vocoder-cpu-hq.pt", "md5": "e8842210dc989e351c2e50614ff55f46"},
        {"filename": "vocoder-cpu-lq.pt", "md5": "cfd048af8bb8190995eac7b95bf7367e"},
    ]

    for model in model_files:
        model_file = model["filename"]
        model_file_path = download_dir / model_file
        model_file_path.parent.mkdir(parents=True, exist_ok=True)

        # If file exists and is too small or has incorrect checksum, delete it
        if model_file_path.exists():
            md5_hash = get_file_hash(model_file_path)
        
            if model_file_path.stat().st_size < 1024:  
                model_file_path.unlink()
            elif md5_hash != model["md5"]:
                _LOGGER.warning("WARNING md5 hash failed for %s, this file may be corrupted. md5: %s", model_file_path, md5_hash)

        # If file does not exist (or was deleted), download it
        if not model_file_path.exists():
            try:
                filename = model_file.split("/")[-1]
                model_url = URL.format(file=filename)
                _LOGGER.warning("Downloading %s to %s", model_url, model_file_path)
                with urlopen(_quote_url(model_url)) as response, open(
                    model_file_path, "wb"
                ) as download_file:
                    shutil.copyfileobj(response, download_file)
                _LOGGER.info("Downloaded %s (%s)", model_file_path, model_url)
            except:
                _LOGGER.exception("Unexpected error while downloading file: %s\nURL: %s", model_file, _quote_url(model_url))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Downloader')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR, help='Directory for the models')
    parser.add_argument('--url', type=str, default=DEFAULT_URL, help='URL for downloading models')
    args = parser.parse_args()

    URL = args.url
    ensure_model_exists(args.model_dir)