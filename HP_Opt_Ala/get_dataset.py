import functools
import pathlib
import shutil
import requests
import os
from tqdm.auto import tqdm
from warnings import warn

base_url = "http://ftp.imp.fu-berlin.de/pub/cmb-data/"
files = [
    "alanine-dipeptide-3x250ns-backbone-dihedrals.npz",
    "alanine-dipeptide-3x250ns-heavy-atom-distances.npz",
    "alanine-dipeptide-3x250ns-heavy-atom-positions.npz",
]

def download(url, filename):
    """
    Download a file from a URL to a local path
    Args:
        url: str, URL to download from
        filename: str, local path to save the file
    Returns:
        pathlib.Path, local path to the downloaded file
    """
    #Adapted from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)
    return path

def main(folder_path):

    data_path = os.path.join(folder_path, 'data/')
    print(data_path)
    if not os.path.exists(data_path):  
        os.makedirs(data_path)
    else:
        warn("Warning: data folder already exists. Removing it and downloading new data.")
        shutil.rmtree(data_path)
        #Remove existing files
    for file in files:
        download(base_url + file, os.path.join(data_path, file))

if __name__ == '__main__':
    main()
