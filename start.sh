#!/bin/bash
cd "$(dirname "$0")"
sudo -E "$(which python3)" main.py "$@"
