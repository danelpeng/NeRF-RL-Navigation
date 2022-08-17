# environment
## python环境
1. conda create -n nerf-rl-nav python = 3.8
2. pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

## 仿真环境
![](https://github.com/zedwind/NeRF-RL-Navigation/blob/master/script/images/recons_VAE_Epoch_46.png)
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

![](https://github.com/zedwind/NeRF-RL-Navigation/blob/master/script/images/recons_VAE_Epoch_46.png)

## 训练
1. python run.py

## 重建效果

![](https://github.com/zedwind/NeRF-RL-Navigation/blob/master/script/images/recons_VAE_Epoch_46.png)


## 加载保存的模型

	vae_model = VAE(in_channels=3, latent_dim=128)	#初始化VAE
	model_dict = torch.load("/checkpoint/**.ckpt", map_location=torch.device("cpu"))	#加载模型
	new_state_dict = OrderedDict()	#消除模型key值不匹配
	for k,v in state_dict.items():
		name = k[6:]
		new_state_dict[name] = v
		
	vae_model.load_state_dict(new_state_dict)	#加载预训练模型，用于策略网络的前端
