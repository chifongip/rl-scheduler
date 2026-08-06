from __future__ import annotations

import os
import pwd


CONDA_DIRS = ("miniconda3", "anaconda3", "miniforge3")


def get_user_home(username: str) -> str | None:
    try:
        entry = pwd.getpwnam(username)
    except KeyError:
        return None
    return entry.pw_dir if os.path.isdir(entry.pw_dir) else None


def is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((os.path.realpath(path), os.path.realpath(root))) == os.path.realpath(root)
    except ValueError:
        return False


def list_conda_environments(username: str) -> dict[str, str]:
    home = get_user_home(username)
    if home is None:
        return {}
    environments: dict[str, str] = {}
    for directory in CONDA_DIRS:
        conda_root = os.path.join(home, directory)
        base_python = os.path.join(conda_root, "bin", "python")
        if os.path.isfile(base_python):
            environments.setdefault("base", base_python)
        envs_root = os.path.join(conda_root, "envs")
        if not os.path.isdir(envs_root):
            continue
        try:
            names = os.listdir(envs_root)
        except OSError:
            continue
        for name in names:
            python_path = os.path.join(envs_root, name, "bin", "python")
            if not name.startswith(".") and os.path.isfile(python_path):
                environments.setdefault(name, python_path)

    env_file = os.path.join(home, ".conda", "environments.txt")
    try:
        with open(env_file) as file_handle:
            paths = [line.strip() for line in file_handle if line.strip()]
    except OSError:
        paths = []
    for env_path in paths:
        python_path = os.path.join(env_path, "bin", "python")
        name = os.path.basename(os.path.normpath(env_path))
        if name and os.path.isfile(python_path):
            environments.setdefault(name, python_path)
    return environments


def resolve_python(username: str, env_name: str, env_type: str | None) -> str | None:
    home = get_user_home(username)
    if home is None:
        return None
    if env_type == "venv":
        if not is_within(env_name, home):
            return None
        python_path = os.path.join(os.path.realpath(env_name), "bin", "python")
        return python_path if os.path.isfile(python_path) else None
    return list_conda_environments(username).get(env_name)
