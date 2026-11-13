#!/usr/bin/env bash
# run_L1.sh / run_L2.sh 共用: 找到装了 pyAgxArm 的 Python -> 确认相机在出图 -> 拉起 CAN
# -> scripts/start.sh <任务> (check_setup 自检, 然后 run_task --mode real)。
# 直接用: bash scripts/_task.sh <L1|L2> [run_task 参数...]
TASK="${1:-}"
case "$TASK" in
    L1|L2) shift ;;
    *) echo "_task.sh: 第一个参数必须是 L1 或 L2" >&2; exit 2 ;;
esac

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

# Python: PYTHON > PIPER_VENV > ~/.venvs/piper-runtime > ~/.venvs/* > /usr/bin/python3,
# 取第一个能 import pyAgxArm 的
if [ -z "${PYTHON:-}" ]; then
    for cand in "${PIPER_VENV:+$PIPER_VENV/bin/python}" "$HOME/.venvs/piper-runtime/bin/python" \
                "$HOME"/.venvs/*/bin/python /usr/bin/python3; do
        if [ -n "$cand" ] && [ -x "$cand" ] && "$cand" -c 'import pyAgxArm' 2>/dev/null; then
            PYTHON="$cand"; break
        fi
    done
fi
if [ -z "${PYTHON:-}" ]; then
    fail '找不到能 import pyAgxArm 的 Python: 设置 PYTHON=<python> 或 PIPER_VENV=<venv 目录>'
    exit 1
fi
export PYTHON
export PYTHONPATH="$PKG${PYTHONPATH:+:$PYTHONPATH}"
ok "Python: $PYTHON"

say "相机 ($COLOR_TOPIC)"
if ! camera_running || ! wait_topic "$COLOR_TOPIC" 5; then
    fail '相机没有出图: 先在另一个终端运行  bash scripts/camera.sh'
    exit 1
fi
ok '相机在出图'

arm_safety_notice
CHANNEL="$("$PYTHON" -c 'from cube_task.core.config import load_config
print(load_config().get("robot.can_channel", "can0"))' 2>/dev/null || echo can0)"
say "CAN $CHANNEL (agx 机械臂)"
if ! bash "$PKG/scripts/can_up.sh" "$CHANNEL" 1000000; then
    fail "CAN $CHANNEL 没起来: 检查 USB-CAN 适配器和机械臂电源"
    exit 1
fi

say "任务 $TASK"
exec bash "$PKG/scripts/start.sh" "$TASK" "$@"
