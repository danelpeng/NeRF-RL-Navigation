# environment
1. conda create -n nerf-rl-nav python = 3.8
2. pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

## 测试环境
python test.py

## 复位环境
python reset.py     ##启动gazebo和ros后 若非正常关闭，手动复位

# 训练VAE

## 激活工作空间
source devel/setup.bash

## 设置gazebo模型环境变量

export GAZEBO_MODEL_PATH=~/ros_project/catkin_ws/src/models

## 收集数据
1. cd encoder
2. python generate_data.py