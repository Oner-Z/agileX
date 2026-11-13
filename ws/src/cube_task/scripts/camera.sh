#!/usr/bin/env bash
# 启动大白相机驱动 (前台运行, 单独开一个终端, Ctrl-C 停止)。L1 / L2 运行期间一直开着。
#
# 用法: bash scripts/camera.sh [LAUNCH_FILE [额外 launch 参数...]]
#   默认 dabai_dcl.py + DaBai DC1 在 USB 2.0 下可用的流:
#     彩色 640x480 15fps MJPG, 深度 640x400 15fps Y11, 关 IR / IMU / 点云
#   换 launch 文件时不加这些默认流参数, 例如: bash scripts/camera.sh dabai_dcw.launch.py
#   驱动工作空间默认 ~/orbbec_ws, 不在那里时: ORBBEC_WS=<目录> bash scripts/camera.sh
# 话题: /camera/color/image_raw, /camera/depth/image_raw (深度已对齐彩色)
case "${1:-}" in
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

if [ $# -gt 0 ]; then
    LAUNCH="$1"; shift
    ARGS=(depth_registration:=true "$@")
else
    # 11/12 在这台 DC1 上跑通过的参数
    LAUNCH=dabai_dcl.py
    ARGS=(depth_registration:=true "sync_mode:=''" enable_sync_output_accel_gyro:=false
          enable_ir:=false enable_point_cloud:=false enable_colored_point_cloud:=false
          color_width:=640 color_height:=480 color_fps:=15 color_format:=MJPG
          depth_width:=640 depth_height:=400 depth_fps:=15 depth_format:=Y11)
fi

# shellcheck disable=SC1091
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

ORBBEC_WS="${ORBBEC_WS:-$HOME/orbbec_ws}"
if [ -f "$ORBBEC_WS/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$ORBBEC_WS/install/setup.bash"
fi
if ! ros2 pkg prefix orbbec_camera >/dev/null 2>&1; then
    fail "找不到 orbbec_camera 驱动 (没有 $ORBBEC_WS/install/setup.bash); 设置 ORBBEC_WS=<目录>"
    exit 1
fi

DEV="$(lsusb 2>/dev/null | grep -i 'ID 2bc5:' | head -n1)"
if [ -z "$DEV" ]; then
    fail '没检测到奥比中光相机 (lsusb 里没有 2bc5), 插到蓝色 USB 3 口'
    exit 1
fi
ok "$DEV"

if camera_running; then
    ok "相机驱动已经在运行 ($COLOR_TOPIC), 不再重复启动"
    exit 0
fi

say "启动 orbbec_camera $LAUNCH (Ctrl-C 停止)"
exec ros2 launch orbbec_camera "$LAUNCH" "${ARGS[@]}"
