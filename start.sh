#!/bin/bash
cd "$(dirname "$0")"
# Replace the default password before exposing the service to other users.
export ADMIN_PASSWORD="${ADMIN_PASSWORD:-admin}"
export ADMIN_SESSION_TIMEOUT_SECONDS="${ADMIN_SESSION_TIMEOUT_SECONDS:-300}"
sudo -E "$(which python3)" main.py "$@"
