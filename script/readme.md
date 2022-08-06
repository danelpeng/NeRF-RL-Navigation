# environment
conda create -n nerf-rl-nav python = 3.8
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

## 测试环境
python test.py

## 复位环境
python reset.py     ##启动gazebo和ros后 若非正常关闭，手动复位

# 训练VAE
## 激活工作空间
source devel/setup.bash

## 收集数据
cd envs/WorldModels
python generate_data.py