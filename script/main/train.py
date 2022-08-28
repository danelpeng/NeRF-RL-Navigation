import argparse
from pprint import pformat
import time
import collections
from datetime import datetime
import uuid
import yaml
import sys
import os
import torch 
import torch.nn as nn 
import numpy as np 
from collections import OrderedDict
import gym
from torchvision.datasets.folder import default_loader
import GPUtil
from tensorboardX import SummaryWriter
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import encoder.vae
import envs.registration
from rl_algo.td3 import Actor, Critic, TD3, ReplayBuffer
from rl_algo.collector import LocalCollector
from models.net import MLP


vae_pretrain_model_path = '/home/lkq/ros_project/acc/acc_ws/src/script/encoder/logs/VAE/version_10/chechpoints/last.ckpt'
vae_model = encoder.vae.VAE(in_channels=3, latent_dim=128)


def initialize_config(config_path, save_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    config["env_config"]["save_path"] = save_path
    config["env_config"]["config_path"] = config_path

    return config

def initialize_logging(config):
    now = datetime.now()
    dt_str = now.strftime("%Y_%m_%d_%H_%M")
    save_path = os.path.join(
        config["env_config"]["save_path"],
        config["env_config"]["env_id"],
        "TD3",
        dt_str,
        uuid.uuid4().hex[:4]
    )
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    shutil.copyfile(
        config["env_config"]["config_path"],
        os.path.join(save_path, "config.yaml")
    )

    return save_path, writer


def initialize_policy(config, env, vae_pretrain_model_path, vae_model):
    state_dim = env.observation_space.shape #(96, 128, 3)
    action_dim = np.prod(env.action_space.shape)
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high


    # devices = GPUtil.getAvailability(order = 'first', limit = 4, maxLoad=0.8, maxMemory=0.8, includeNan=False, excludeID=[], excludeUUID=[])
    # device = "cuda:%d"%(devices[0]) if len(devices) >0 else "cpu"
    device = "cuda:0"
    print(">>>>>>>> Running on device %s"%(device))

    vae = load_pretrain_model(vae_pretrain_model_path, vae_model)
    # initialize actor
    input_dim = 128
    actor = Actor(
        encoder=vae, 
        head=MLP(input_dim=input_dim, num_layers=2, hidden_layer_size=128),
        action_dim=action_dim,
        ).to(device)

    actor_optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad,actor.parameters()),
        lr=config["training_config"]["actor_lr"]
    )
    print("Total number of parameters of actor: %d" %sum(p.numel() for p in actor.parameters()))    #total params
    print("Trainable number of parameters of actor: %d" %sum(p.numel() for p in actor.parameters() if p.requires_grad))    #trainable params

    # initialize critic
    input_dim += np.prod(action_dim)
    critic = Critic(
        encoder=vae,
        head=MLP(input_dim=input_dim, num_layers=2, hidden_layer_size=128),
    ).to(device)
    critic_optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad,critic.parameters()),
        lr=config["training_config"]["critic_lr"]
    )
    print("Total number of parameters of critic: %d" %sum(p.numel() for p in critic.parameters()))   #total params
    print("Trainable number of parameters of critic: %d" %sum(p.numel() for p in critic.parameters() if p.requires_grad))    #trainable params

    # initialize agents
    policy = TD3(
        actor,actor_optim,
        critic, critic_optim,
        action_range=[action_space_low, action_space_high],
        device=device
    )

    # initialize buffer
    replay_buffer = ReplayBuffer(
        state_dim, action_dim, max_size=8,
        device=device
    )

    return policy, replay_buffer


def load_pretrain_model(model_path, model):

    model_dict = torch.load(model_path,map_location=torch.device('cpu'))
    state_dict = model_dict['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[6:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

    return model

# def load_img(img):
#     img = np.transpose(img, (2,0,1))
#     img = torch.tensor(img).reshape((1,3,96,128)).to(torch.float32)
#     return img

# def test(img, vae_pretrain_model_path, vae_model):
#     img = load_img(img)
#     vae = load_pretrain_model(vae_pretrain_model_path, vae_model)
#     [mu, log_var] = vae.encode(img)
#     vae_feature = vae.reparameterize(mu, log_var)
#     print(vae_feature) #[1, 128]
# test(img, vae_pretrain_model_path, vae_model)

def train(env, policy, replay_buffer, config):
    save_path, writer = initialize_logging(config)
    print(">>>> initialized logging")

    collector = LocalCollector(policy, env, replay_buffer)
    # print(">>>> Pre-collect experience")
    # collector.collect(n_steps=8)
    print(">>>>Start training")

    n_steps = 0
    n_iter = 0
    n_ep = 0
    epinfo_buf = collections.deque(maxlen=8)    #300
    t0 = time.time()

    while n_steps < config["training_config"]["max_step"]:
        policy.exploration_noise = - (0.1 - 0.0999) * n_steps / config["training_config"]["max_step"] + 0.1
        steps, epinfo = collector.collect(n_steps=config["training_config"]["collect_per_step"])

        n_steps += steps
        n_iter += 1
        n_ep += len(epinfo)
        epinfo_buf.extend(epinfo)

        loss_infos = []
        for _ in range(config["training_config"]["update_per_step"]):
            loss_info = policy.train(replay_buffer, config["training_config"]["batch_size"])
            loss_infos.append(loss_info)
        
        loss_info = {}
        for k in loss_infos[0].keys():
            loss_info[k] = np.mean([li[k] for li in loss_infos if li[k] is not None])
        
        t1 = time.time()
        log = {
            "Episode_return": np.mean([epinfo["ep_rew"] for epinfo in epinfo_buf]),
            "Episode_length": np.mean([epinfo["ep_len"] for epinfo in epinfo_buf]),
            "Success": np.mean([epinfo["success"] for epinfo in epinfo_buf]),
            "Time": np.mean([epinfo["ep_time"] for epinfo in epinfo_buf]),
            "fps": n_steps / (t1 -t0),
            "n_episode": n_ep,
            "Steps": n_steps,
            "Exploration_noise": policy.exploration_noise,
        }
        log.update(loss_info)
        print(pformat(log))

        if n_iter % config["training_config"]["log_intervals"] == 0:
            for k in log.keys():
                writer.add_scalar('train/' + k, log[k], global_step=n_steps)
            policy.save(save_path, "last_policy")
            print("Logging to %s" %save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest='config_path', default='main/configs/default.yaml')
    parser.add_argument('--device', dest='device', default=None)
    args = parser.parse_args()
    CONFIG_PATH = args.config_path
    SAVE_PATH = "main/logging/"
    print(">>>>>>>>>>>>>>>>> Loading the configuration")
    config = initialize_config(CONFIG_PATH, SAVE_PATH)
    np.random.seed(config["env_config"]["seed"])
    torch.manual_seed(config["env_config"]["seed"])

    print(">>>>>>>>>>>>>>>>> Creating the environments")
    env = gym.make(id = 'motion_control_continuous_laser-v0',)

    print(">>>>>>>>>>>>>>>>> Initializing the policy")
    policy, replay_buffer = initialize_policy(config, env, vae_pretrain_model_path, vae_model)

    print(">>>>>>>>>>>>>>>>>> Start training <<<<<<<<<<<<<<<<<<<<<<<")
    train(env, policy, replay_buffer, config)