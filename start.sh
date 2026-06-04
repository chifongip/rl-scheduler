#!/bin/bash
cd "$(dirname "$0")"
export ADMIN_PASSWORD="admin"
sudo -E "$(which python3)" main.py "$@"
