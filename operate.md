下面是一会儿上机的步骤，按队友的上机指南顺序，换上了我写的脚本。一小时做不完全部，第 1–6 步是今天的重点。所有命令我都在仿真里试过，但都还没在真机上跑过：每步先对照"期望"，对不上就停。

指南里有两处已经过时，我改好并提交了（`6db04b1`）：
- L1 的标签要写 `red`，不是 `cube_red`。用 `cube_red` 会提示"没有放置区"，机械臂不动。
- 外参赛方不提供，要用 `handeye.sh` 自己标。

**开始前**
- 上电前确认 J1 零点刻度线对齐，安全员手放在急停上，桌面清空。
- 接线：航空插头接 XT30 电源（底座灯绿色闪烁），再接 USB 转 CAN 模块到电脑；大白相机插电脑的蓝色 USB 3 口。
- 每开一个新终端都先执行：
```bash
cd ~/vision/ws/src/cube_task
source scripts/env.sh          # 期望最后一行：pyAgxArm: 可用
```

**第 1 步：CAN 和只读状态（5 分钟）**
```bash
bash scripts/arm_check.sh --plan-only
```
- 脚本依次做三件事：拉起 CAN（要输 sudo 密码）、确认能收到机械臂的反馈帧、只读打印关节角、TCP 和夹爪宽度。然后打开 RViz，只规划不动。
- 期望：打印的关节角和真机姿态一致，RViz 里的模型姿态也和真机一致。
- 看完按 `Ctrl-C` 退出。

**第 2 步：夹爪（5 分钟，第一次使能）**
```bash
ct motion_ctl gripper --mode real --width 0.05 --speed 20
ct motion_ctl gripper --mode real --width 0.08 --speed 20
ct motion_ctl gripper --mode real --width 0.02 --speed 20
```
- 每条会先问 `yes` 才使能。期望夹爪开到约 50 / 80 / 20 mm。
- 不要加 `--force`，夹爪力的单位还没确认。

**第 3 步：六个关节小步（12 分钟）**
```bash
ct motion_ctl joints --mode real --joint 1 --delta 0.05 --dry-run
ct motion_ctl joints --mode real --joint 1 --delta 0.05 --speed 20
ct motion_ctl joints --mode real --joint 1 --delta -0.05 --speed 20
# --joint 2 到 6 照做
```
- **动的关节编号或方向不对，立刻停，后面都不做。**

**第 4 步：量 TCP，然后 RViz 拖动（15 分钟）**
1. 用尺子量法兰端面到闭合指尖中心的距离，手册图纸读数是 142.5 mm。
2. 写进 `config/site/local.yaml`，在现有内容后面加：
   ```yaml
   robot:
     tcp_offset: [0.0, 0.0, 0.1425, 0.0, 0.0, 0.0]   # 第 3 个数换成实测值（米）
   ```
3. `scripts/moveit_rviz.sh` 里的 `tcp_offset:=[0.0, 0.0, 0.140, ...]` 也改成同一个值。
4. 检查配置，期望 `tcp_offset` 那一行通过：
   ```bash
   ct check_setup --task L1 --skip can,ros
   ```
5. 打开 RViz 真实执行：
   ```bash
   bash scripts/arm_check.sh          # 第 4 步会问是否允许 RViz 动臂，输入 y，速度 20%
   ```
   第一次只拖很小的幅度，确认动作平滑，结束按 `Ctrl-C`。

到这里运控就验证完了，大约用掉 40 分钟。

**第 5 步：相机（5 分钟，不动臂）**
```bash
bash scripts/camera_check.sh
```
- 第一次会问要不要装 udev 规则（要 sudo）。装完把相机拔下再插上，按回车。
- 期望：打印的帧率不低于 5 fps，内参来自 camera_info，深度有效。RViz 里能看到 RGB、深度和点云。
- 脚本会打印用的是哪个 launch 文件，记下来，后面要用。关掉 RViz，驱动也会一起停。
- 后面的工具需要驱动一直开着，所以单独开一个终端运行：
  ```bash
  ros2 launch orbbec_camera <刚才打印的launch文件> depth_registration:=true
  ```

**第 6 步：手眼标定（15 分钟，工具不动臂）**

先确认第 4 步已经量好 TCP，并且 `arm_check.sh` 已经退出。同一时间只能有一个程序控制机械臂。
```bash
bash scripts/handeye.sh
```
- 桌上放一个 50 mm 方块。按机械臂上的示教按钮进入拖动示教，然后每个点这样做：
  1. 拖臂让相机看到方块，手离开、臂静止后按回车测量。
  2. 拖臂让闭合的指尖正好碰到方块顶面中心，按回车。
- 一共 8 个点，铺满整个取物区，每个点换个高度或角度。
- 期望：残差 RMS 不超过 3 mm，结果写进 `config/site/extrinsics.yaml`。
- 做完再按一次示教按钮，退出拖动示教。

**第 7 步：示教取物区、卡纸和观察位（15 分钟，工具不动臂）**
```bash
cp config/examples/tasks.example.yaml config/site/tasks.yaml
ct teach_poses --task L1
```
按顺序输入：
1. `p`：用指尖依次碰取物区的 4 个角。这一步同时记录桌面高度，一定要做。
2. `z zone_green`、`z zone_yellow`、`z zone_red`：碰每张卡纸的中心，尺寸填 `0.2 0.2`，围边高度填 `0`。
3. `o`：观察位，相机朝下能看到取物区。
4. `v` 做 3 次：L2 用的几个观察视角，相邻视角之间要有重叠。
5. `w` 写入，`q` 退出。

之后检查，目标是没有 FAIL：
```bash
ct check_setup --task L1
```

**第 8 步：核对标定，再试视觉抓取（10–15 分钟，会动臂）**
```bash
bash scripts/handeye.sh --verify-only                          # 指尖悬停在每个方块上方 3 cm，看偏差
ct motion_ctl vision --mode real --task L1 --list-only          # 只打印检测结果，不动臂
ct motion_ctl vision --mode real --task L1 --dry-run --speed 20 # 只悬停，不闭合夹爪
ct motion_ctl vision --mode real --task L1 --label red --speed 20   # 真抓一个
```
- L1 现在默认用队友的新识别（`l1_new`，没在真机上验证过）。识别不对时，在命令后面加 `--set perception.backend=pipeline` 换回原来的识别。原来的识别要先做颜色标定 `ct color_calib --task L1`。

**有空就做：采数据集**
```bash
bash scripts/capture_dataset.sh l2_mixed
```
按 `s` 存一帧，`a` 开或关自动存，`q` 退出。两帧之间拖动机械臂换角度、换物体摆法。

**结束前：备份现场配置**
```bash
cd ~/vision && git add ws/src/cube_task/config/site && git commit -m "11/12 现场配置"
```

**安全规则**
- 运行 `vision` 或 `pick` 时，在那个终端按 `p` 或空格是原地停住（首选），`e` 是电子急停，硬件急停由安全员负责。
- 同一时间只开一个会动臂的程序。
- 出错就停，记下终端输出，以及 `~/vision_runs/` 里最新的日志目录。

**时间不够时**
- 今天至少做完第 1–6 步，也就是运控、相机、TCP 和手眼标定。
- 第 7–8 步、颜色标定、L2 果蔬学习和整轮 `run_task` 留到 11/13。


# 终端 1：相机，一直开着
cd ~/projects/agileX/ws/src/cube_task
conda deactivate
bash scripts/camera.sh

# 终端 2：任务
cd ~/projects/agileX/ws/src/cube_task
bash scripts/run_L1.sh --targets red,yellow,green
bash scripts/run_L2.sh
