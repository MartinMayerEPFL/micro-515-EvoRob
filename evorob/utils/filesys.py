import os
import re
from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent


def search_file_list(rootname, file_name) -> list:
    file_list = []
    for root, dirs, files in os.walk(rootname):
        for file in files:
            if file_name in file:
                file_list.append(os.path.join(root, file))
    file_list.sort(key=natural_keys)
    return file_list


def atoi(text) -> int | str:
    return int(text) if text.isdigit() else text


def natural_keys(text) -> list:
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def get_last_checkpoint_dir(path: str, required_files: tuple[str, ...] = ()) -> str:
    """
    Get the last checkpoint directory from a given path. Directories are numbered.
    Args:
        path (str): The path to search for checkpoint directories.
        required_files (tuple[str, ...]): Files that must exist inside a checkpoint
            directory for it to be considered valid.
    """
    if not os.path.isdir(path):
        print(f"Path {path} is not a directory.")
        return ""

    dirs = [
        d
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d.isdigit()
    ]
    if not dirs:
        print(f"No checkpoint directories found in {path}.")
        return ""

    valid_dirs = []
    for d in dirs:
        checkpoint_dir = os.path.join(path, d)
        if all(os.path.isfile(os.path.join(checkpoint_dir, file)) for file in required_files):
            valid_dirs.append(d)

    if required_files and not valid_dirs:
        print(
            f"No valid checkpoint directories found in {path} "
            f"with required files {required_files}."
        )
        return ""

    candidate_dirs = valid_dirs if required_files else dirs
    last_checkpoint = max(candidate_dirs, key=lambda d: int(d))
    return os.path.join(path, last_checkpoint)


def get_distinct_filename(filename: str) -> str:
    """
    If the filename already exists, append a number to make it distinct.
    Args:
        filename (str): The original filename.
    """
    if not os.path.exists(filename):
        return filename

    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = f"{base}_{counter}{ext}"
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{base}_{counter}{ext}"
    return new_filename
