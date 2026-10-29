#!/bin/bash
# Install the scheduler as a system service, using the caller's Python environment.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./install-service.sh [--python PATH] [--no-start] [--dry-run]

  --python PATH  Python interpreter to use (default: python3 from the current PATH).
  --no-start     Install and enable at boot without starting/restarting the app now.
  --dry-run      Print the unit without sudo or changes to the system.
  -h, --help     Show this help.

Activate your conda/venv environment before running this script, or pass its
absolute Python path. The script requests sudo automatically when installing.
Existing /etc/rl-scheduler.env settings are preserved; a new installation gets
a random admin password in that root-only file.
EOF
}

die() {
    printf 'Error: %s\n' "$*" >&2
    exit 1
}

# systemd has its own quoting and % specifiers, even inside double quotes.
unit_quote() {
    local value=$1
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    value=${value//%/%%}
    printf '"%s"' "$value"
}

render_unit() {
    cat <<EOF
[Unit]
Description=RL Scheduler GPU workload manager
Wants=network-online.target
After=network-online.target
RequiresMountsFor=$(unit_quote "$PROJECT_DIR") $(unit_quote "$PYTHON_BIN")
StartLimitIntervalSec=0

[Service]
Type=exec
User=root
# WorkingDirectory is a single unquoted path; '/' prevents a trailing backslash
# in a directory name from becoming a line continuation.
WorkingDirectory=${PROJECT_DIR//%/%%}/
# Like WorkingDirectory, EnvironmentFile takes a single unquoted path.
EnvironmentFile=${ENV_PATH//%/%%}
$(for setting in "${SERVICE_ENV[@]}"; do printf 'Environment=%s\n' "$(unit_quote "$setting")"; done)
# The ':' prefix disables dollar expansion in paths. No shell activation is needed.
# env allows interpreter paths with characters systemd forbids in executable names.
ExecStart=:/usr/bin/env $(unit_quote "$PYTHON") -u $(unit_quote "$PROJECT_DIR/main.py")
Restart=on-failure
RestartSec=10s
TimeoutStopSec=30s
StandardOutput=journal
StandardError=journal
SyslogIdentifier=rl-scheduler

[Install]
WantedBy=multi-user.target
EOF
}

# Use the system manager for EnvironmentFile parsing and a clean root environment.
# This also avoids treating the settings file as an executable shell script.
check_environment() {
    local setting
    local -a environment_args=()
    for setting in "${SERVICE_ENV[@]}"; do
        environment_args+=("--setenv=$setting")
    done
    local check_code
    check_code=$(cat <<'PYCODE'
import os
import secrets
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Python 3.10+ is required")
import fastapi, uvicorn, aiosqlite, pynvml, pydantic

# Generate settings only after imports succeed. Preserve existing settings.
config_path = sys.argv[1]
if not os.path.exists(config_path):
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as config:
        config.write("# Settings for rl-scheduler.service (systemd EnvironmentFile syntax).\n")
        config.write(f"ADMIN_PASSWORD={secrets.token_urlsafe(32)}\n")
        config.write("ADMIN_SESSION_TIMEOUT_SECONDS=300\n")
    print(f"Created {config_path} with a random admin password.")
PYCODE
)
    # Pass the command on stdin, with Bash quoting, so systemd cannot expand '$'
    # in interpreter paths or Python code. No config values enter the command line.
    if ! printf 'exec /usr/bin/env %q -c %q %q\n' "$PYTHON" "$check_code" "$ENV_PATH" |
        systemd-run --quiet --wait --pipe --collect --service-type=oneshot \
            --uid=root --working-directory="$PROJECT_DIR" \
            --property="EnvironmentFile=-$ENV_PATH" \
            --property=TimeoutStartSec=60s "${environment_args[@]}" /bin/bash; then
        printf '\nDependency check failed using the service environment.\n' >&2
        printf 'Check /etc/rl-scheduler.env if dependencies need PYTHONPATH or LD_LIBRARY_PATH.\n' >&2
        printf 'Install into a writable conda/venv environment, excluding user packages:\n  ' >&2
        printf '%q -I -m pip --isolated install --no-user -r %q\n' \
            "$PYTHON" "$PROJECT_DIR/requirements.txt" >&2
        die 'Service installation stopped; the existing service unit was not replaced'
    fi
}

install_service() {
    for target in "$UNIT_PATH" "$ENV_PATH"; do
        [[ ! -L "$target" ]] || die "Refusing to replace or follow symlink: $target"
        [[ ! -e "$target" || -f "$target" ]] || die "Expected a regular file: $target"
    done

    TEMP_DIR=$(mktemp -d)
    trap 'rm -rf -- "$TEMP_DIR"' EXIT
    render_unit > "$TEMP_DIR/rl-scheduler.service"
    systemd-analyze verify "$TEMP_DIR/rl-scheduler.service"

    # Restrict an existing settings file before passing it to the service manager.
    if [[ -e "$ENV_PATH" ]]; then
        chown root:root "$ENV_PATH"
        chmod 600 "$ENV_PATH"
    fi
    check_environment
    chown root:root "$ENV_PATH"
    chmod 600 "$ENV_PATH"
    install -o root -g root -m 644 "$TEMP_DIR/rl-scheduler.service" "$UNIT_PATH"
    systemctl daemon-reload
    systemctl enable rl-scheduler.service
    if "$START"; then
        systemctl restart rl-scheduler.service
    fi

    printf '\nInstalled and enabled rl-scheduler.service using %s\n' "$PYTHON"
    printf 'Admin settings/password: sudo cat %s\n' "$ENV_PATH"
    printf 'Edit settings: sudoedit %s (then sudo systemctl restart rl-scheduler)\n' "$ENV_PATH"
    printf 'Status: sudo systemctl status rl-scheduler\n'
    printf 'Logs: sudo journalctl -u rl-scheduler -f\n'
    if ! "$START"; then
        printf 'Start when ready: sudo systemctl start rl-scheduler\n'
    fi
}

main() {
    PYTHON=python3
    START=true
    DRY_RUN=false
    while (($#)); do
        case "$1" in
            --python)
                (($# >= 2)) || die '--python requires an interpreter path'
                PYTHON=$2
                shift 2
                ;;
            --no-start) START=false; shift ;;
            --dry-run) DRY_RUN=true; shift ;;
            -h|--help) usage; exit 0 ;;
            *) die "Unknown argument: $1 (see --help)" ;;
        esac
    done

    PROJECT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
    [[ -f "$PROJECT_DIR/main.py" ]] || die 'main.py must be beside this installer'
    PYTHON=$(command -v -- "$PYTHON") || die 'Python not found; activate your environment or use --python'
    [[ -x "$PYTHON" && ! -d "$PYTHON" ]] || die "Not an executable interpreter: $PYTHON"
    # Resolve relative paths, but keep the interpreter symlink: resolving bin/python
    # itself would lose a venv's identity and select the system Python instead.
    PYTHON=$(cd -- "$(dirname -- "$PYTHON")" && printf '%s/%s' "$PWD" "$(basename -- "$PYTHON")")
    PYTHON_BIN=$(dirname -- "$PYTHON")
    [[ ! "$PROJECT_DIR$PYTHON" =~ [[:cntrl:]] ]] || die 'Paths must not contain control characters'

    UNIT_PATH=/etc/systemd/system/rl-scheduler.service
    ENV_PATH=/etc/rl-scheduler.env
    SERVICE_ENV=(
        PYTHONUNBUFFERED=1
        PYTHONNOUSERSITE=1
        "PATH=$PYTHON_BIN:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )

    if "$DRY_RUN"; then
        render_unit
        exit 0
    fi

    # Select Python before sudo resets PATH. Never depend on conda being initialized
    # in root's login shell, and do not copy the caller's environment into the service.
    if ((EUID != 0)); then
        SUDO_ARGS=(--python "$PYTHON")
        if ! "$START"; then SUDO_ARGS+=(--no-start); fi
        exec sudo -- /bin/bash "$PROJECT_DIR/install-service.sh" "${SUDO_ARGS[@]}"
    fi

    [[ -d /run/systemd/system ]] || die 'A running systemd system manager is required'
    [[ -x /usr/bin/env ]] || die '/usr/bin/env is required'
    for tool in systemctl systemd-run systemd-analyze install; do
        command -v "$tool" >/dev/null || die "Required command is missing: $tool"
    done
    install_service
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
