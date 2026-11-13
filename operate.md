# 终端 1：相机，一直开着
cd ~/projects/agileX/ws/src/cube_task
conda deactivate
bash scripts/camera.sh

# 终端 2：任务
cd ~/projects/agileX/ws/src/cube_task
bash scripts/run_L1.sh --targets red,yellow,green
bash scripts/run_L2.sh
