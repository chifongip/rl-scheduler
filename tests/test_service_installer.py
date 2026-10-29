import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest


INSTALLER = Path(__file__).resolve().parents[1] / "install-service.sh"


def run_installer(*args, installer=INSTALLER, env=None):
    return subprocess.run(
        ["/bin/bash", str(installer), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )


def test_dry_run_selects_active_environment_and_preserves_python_symlink(tmp_path):
    bin_dir = tmp_path / "conda-env" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python3"
    python.symlink_to(sys.executable)
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"}

    result = run_installer("--dry-run", env=env)

    assert result.returncode == 0, result.stderr
    assert f'ExecStart=:/usr/bin/env "{python}" -u ' in result.stdout
    assert f'Environment="PATH={bin_dir}:' in result.stdout
    assert "User=root\n" in result.stdout
    assert "EnvironmentFile=/etc/rl-scheduler.env\n" in result.stdout
    assert "Restart=on-failure\n" in result.stdout
    assert "WantedBy=multi-user.target\n" in result.stdout


def test_paths_with_spaces_and_systemd_metacharacters_pass_validation(tmp_path):
    project = tmp_path / 'project "100%" $SCHEDULER_TEST'
    project.mkdir()
    installer = project / "install-service.sh"
    shutil.copyfile(INSTALLER, installer)
    (project / "main.py").touch()
    bin_dir = tmp_path / 'env "100%" $SCHEDULER_TEST' / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.symlink_to(sys.executable)

    result = run_installer("--python", str(python), "--dry-run", installer=installer)

    assert result.returncode == 0, result.stderr
    assert f"WorkingDirectory={str(project).replace('%', '%%')}/\n" in result.stdout
    assert '\\"100%%\\" $SCHEDULER_TEST' in result.stdout
    assert "ExecStart=:" in result.stdout  # Disable dollar expansion in command paths.
    if shutil.which("systemd-analyze"):
        unit = tmp_path / "rl-scheduler.service"
        unit.write_text(result.stdout)
        validation = subprocess.run(
            ["systemd-analyze", "verify", str(unit)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert validation.returncode == 0, validation.stderr
        # verify may return success after silently ignoring an invalid setting.
        assert "EnvironmentFile=" not in validation.stderr


@pytest.mark.parametrize("args, message", [
    (["--python"], "requires an interpreter path"),
    (["--python", "/nonexistent/rl-scheduler-python"], "Python not found"),
    (["--unknown"], "Unknown argument"),
])
def test_invalid_arguments_fail_before_installing(args, message):
    result = run_installer(*args)
    assert result.returncode != 0
    assert message in result.stderr
    assert not result.stdout


def test_control_characters_in_paths_are_rejected(tmp_path):
    python = tmp_path / "python\nRestart=always"
    python.symlink_to(sys.executable)

    result = run_installer("--python", str(python), "--dry-run")

    assert result.returncode != 0
    assert "control characters" in result.stderr
    assert not result.stdout


@pytest.mark.skipif(os.geteuid() == 0, reason="Exercises sudo from a normal user")
def test_sudo_receives_selected_python_and_no_start_without_preserving_environment(tmp_path):
    bin_dir = tmp_path / "active env" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python3"
    python.symlink_to(sys.executable)
    sudo = bin_dir / "sudo"
    sudo.write_text('#!/bin/bash\nprintf "%s\\0" "$@"\n')
    sudo.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"}

    result = run_installer("--no-start", env=env)

    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\0")[:-1] == [
        "--", "/bin/bash", str(INSTALLER), "--python", str(python), "--no-start",
    ]


# Exercise the real install functions against temporary paths. Only privileged
# tools are simulated; the preflight runs real Bash/Python with a clean service
# environment, and creates the real settings file in the sandbox directory.
@pytest.fixture
def installation(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "main.py").touch()
    environment = tmp_path / 'conda env $LITERAL "100%"'
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", "--without-pip", str(environment)],
        check=True,
        capture_output=True,
        timeout=20,
    )
    python = environment / "bin" / "python"
    site_packages = environment / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    trace = tmp_path / "commands.jsonl"
    fake_tool = tools_dir / "fake-tool"
    fake_tool.write_text(f"#!{sys.executable} -I\n" + r'''
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["INSTALL_TRACE"], "a") as log:
    log.write(json.dumps([name, *args]) + "\n")
if name == "systemd-run":
    assert all(option in args for option in (
        "--wait", "--pipe", "--collect", "--service-type=oneshot", "--uid=root",
    ))
    env = {"HOME": str(Path(os.environ["INSTALL_TRACE"]).parent / "root-home")}
    for arg in args:
        if arg.startswith("--setenv="):
            key, value = arg[len("--setenv="):].split("=", 1)
            env[key] = value
    property_arg = next(arg for arg in args if arg.startswith("--property=EnvironmentFile="))
    # systemd-run's EnvironmentFile property accepts a raw path, without unit-file
    # quotes or percent escaping (unlike EnvironmentFile in a .service file).
    config_path = property_arg.split("=", 2)[2].removeprefix("-")
    config = Path(config_path)
    if config.exists():
        # Only the simple quoted assignment syntax used in these test fixtures.
        # Production delegates all parsing to systemd, not to this test stub.
        for line in config.read_text().splitlines():
            if line and not line.startswith("#"):
                key, value = shlex.split(line)[0].split("=", 1)
                env[key] = value
    cwd = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--working-directory="))
    result = subprocess.run(["/bin/bash"], input=sys.stdin.read(), text=True, env=env, cwd=cwd)
    sys.exit(result.returncode)
if name == "install":
    shutil.copyfile(args[-2], args[-1])
    os.chmod(args[-1], int(args[args.index("-m") + 1], 8))
if name == "systemctl" and args[0] == "restart":
    sys.exit(int(os.environ.get("INSTALL_RESTART_FAIL", "0")))
''')
    fake_tool.chmod(0o755)
    for name in ("systemd-run", "systemctl", "systemd-analyze", "install", "chown"):
        (tools_dir / name).symlink_to(fake_tool)

    class Installation:
        env_path = tmp_path / 'settings env "100%" $LITERAL'
        unit_path = tmp_path / "rl-scheduler.service"
        packages = site_packages

        def add_dependencies(self, target=None):
            target = target or self.packages
            target.mkdir(parents=True, exist_ok=True)
            for module in ("fastapi", "uvicorn", "aiosqlite", "pynvml", "pydantic"):
                (target / f"{module}.py").write_text("")

        def run(self, start=True, extra_env=None):
            return subprocess.run(
                ["/bin/bash", "-c", '''
source "$1"
PROJECT_DIR=$2
PYTHON=$3
PYTHON_BIN=${PYTHON%/*}
ENV_PATH=$4
UNIT_PATH=$5
START=$6
SERVICE_ENV=(PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 "PATH=$PYTHON_BIN:/usr/bin:/bin")
install_service
''', "installer-test", str(INSTALLER), str(project), str(python),
                 str(self.env_path), str(self.unit_path), str(start).lower()],
                env={**os.environ, "PATH": f"{tools_dir}:/usr/bin:/bin", "INSTALL_TRACE": str(trace),
                     **(extra_env or {})},
                capture_output=True,
                text=True,
                timeout=20,
            )

        def commands(self, name):
            if not trace.exists():
                return []
            return [command[1:] for line in trace.read_text().splitlines()
                    if (command := json.loads(line))[0] == name]

    return Installation()


def test_install_creates_private_settings_and_enables_service(installation):
    installation.add_dependencies()

    result = installation.run()

    assert result.returncode == 0, result.stderr
    settings = dict(line.split("=", 1) for line in installation.env_path.read_text().splitlines()
                    if not line.startswith("#"))
    assert len(settings["ADMIN_PASSWORD"]) >= 40
    assert settings["ADMIN_SESSION_TIMEOUT_SECONDS"] == "300"
    assert installation.env_path.stat().st_mode & 0o777 == 0o600
    assert installation.unit_path.stat().st_mode & 0o777 == 0o644
    assert f"EnvironmentFile={str(installation.env_path).replace('%', '%%')}\n" in installation.unit_path.read_text()
    assert installation.commands("systemctl") == [
        ["daemon-reload"], ["enable", "rl-scheduler.service"], ["restart", "rl-scheduler.service"],
    ]
    assert settings["ADMIN_PASSWORD"] not in result.stdout + result.stderr


def test_preflight_uses_saved_environment_and_preserves_settings(installation, tmp_path):
    packages = tmp_path / "custom packages $LITERAL"
    installation.add_dependencies(packages)
    marker = tmp_path / "must-not-exist"
    settings = (
        "ADMIN_PASSWORD='saved password $LITERAL'\n"
        "ADMIN_SESSION_TIMEOUT_SECONDS=900\n"
        f"PYTHONPATH='{packages}'\n"
        "LD_LIBRARY_PATH='/custom/lib $LITERAL'\n"
        f"DESCRIPTION='$(touch {marker})'\n"
    )
    installation.env_path.write_text(settings)
    (packages / "fastapi.py").write_text(
        "import os\n"
        "assert os.environ['LD_LIBRARY_PATH'] == '/custom/lib $LITERAL'\n"
        "assert os.environ['ADMIN_SESSION_TIMEOUT_SECONDS'] == '900'\n"
        "assert 'CALLER_ONLY' not in os.environ\n"
    )

    result = installation.run(extra_env={"CALLER_ONLY": "must not leak"})

    assert result.returncode == 0, result.stderr
    assert installation.env_path.read_text() == settings
    assert installation.env_path.stat().st_mode & 0o777 == 0o600
    assert not marker.exists()
    assert "saved password" not in result.stdout + result.stderr
    assert "saved password" not in str(installation.commands("systemd-run"))


def test_failed_preflight_keeps_previous_unit_and_suggests_environment_local_pip(installation):
    installation.unit_path.write_text("previous service\n")

    result = installation.run()

    assert result.returncode != 0
    assert installation.unit_path.read_text() == "previous service\n"
    assert not installation.env_path.exists()
    assert not installation.commands("install")
    assert not installation.commands("systemctl")
    command_line = next(line.strip() for line in result.stderr.splitlines() if "-m pip" in line)
    command = shlex.split(command_line)
    assert command[1:] == [
        "-I", "-m", "pip", "--isolated", "install", "--no-user", "-r", command[-1],
    ]
    assert Path(command[0]).name == "python"


def test_no_start_still_checks_dependencies_and_enables_service(installation):
    installation.add_dependencies()

    result = installation.run(start=False)

    assert result.returncode == 0, result.stderr
    assert installation.commands("systemd-run")
    assert installation.commands("systemctl") == [["daemon-reload"], ["enable", "rl-scheduler.service"]]


def test_restart_failure_is_reported_without_success_message(installation):
    installation.add_dependencies()

    result = installation.run(extra_env={"INSTALL_RESTART_FAIL": "1"})

    assert result.returncode != 0
    assert "Installed and enabled" not in result.stdout
