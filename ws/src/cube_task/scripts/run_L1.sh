#!/usr/bin/env bash
# L1 真机一轮。先在另一个终端运行  bash scripts/camera.sh
# 用法: bash scripts/run_L1.sh --targets red,blue,green [--no-wait] [--dry-run] [--speed 20] ...
exec bash "$(dirname "${BASH_SOURCE[0]}")/_task.sh" L1 "$@"
