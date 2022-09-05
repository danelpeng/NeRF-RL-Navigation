import copy
import pickle
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, encoder, head, action_dim):
        super(Actor, self).__init__()

        self.encoder = encoder
        self.head = head
        self.fc = nn.Linear(self.head.feature_dim, action_dim)

    def forward(self, obs, last_act, relative_pos):
        last_act = torch.tensor(last_act).reshape((-1,2)).to(torch.float32)
        relative_pos = torch.tensor(relative_pos).reshape((-1,2)).to(torch.float32)
        # last_act = torch.tensor(last_act).reshape((-1,2)).to(torch.float64)
        # relative_pos = torch.tensor(relative_pos).reshape((-1,2)).to(torch.float64)

        [mu, log_var] = self.encoder.encode(obs) 
        latent_vec = self.encoder.reparameterize(mu, log_var)    
        """
        add (v, w) and goal_pos to state
        """
        state = np.concatenate([latent_vec.cpu(), relative_pos.cpu(), last_act.cpu()], axis=1)
        state = torch.tensor(state).to(torch.float32).to(latent_vec.device) #34
        # state = torch.tensor(state).to(torch.float64).to(latent_vec.device) #34
        
        state = self.head(state)    
        return torch.tanh(self.fc(state))   #[-1,1]
    
class Critic(nn.Module):
    def __init__(self, encoder, head):
        super(Critic, self).__init__()

        #Q1 arch
        self.encoder1 = encoder
        self.head1 = head
        self.fc1 = nn.Linear(self.head1.feature_dim, 1)

        #Q2 arch
        self.encoder2 = copy.deepcopy(encoder)
        self.head2 = copy.deepcopy(head)
        self.fc2 = nn.Linear(self.head2.feature_dim, 1)
    
    def forward(self, obs, last_act, relative_pos, action):
        last_act = torch.tensor(last_act).reshape((-1,2)).to(torch.float32)
        relative_pos = torch.tensor(relative_pos).reshape((-1,2)).to(torch.float32)
        # last_act = torch.tensor(last_act).reshape((-1,2)).to(torch.float64)
        # relative_pos = torch.tensor(relative_pos).reshape((-1,2)).to(torch.float64)

        #critic1
        if self.encoder1:
            [mu1, log_var1] = self.encoder1.encode(obs)
            latent_vec_1 = self.encoder1.reparameterize(mu1, log_var1)
        else:
            latent_vec_1 = obs

        state1_concat = np.concatenate([latent_vec_1.cpu(), relative_pos.cpu(), last_act.cpu()], axis=1)
        state1_concat = torch.tensor(state1_concat).to(torch.float32).to(latent_vec_1.device) #34
        # state1_concat = torch.tensor(state1_concat).to(torch.float64).to(latent_vec_1.device) #34

        sa1 = torch.cat([state1_concat, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)

        #critic2
        if self.encoder2:
            [mu2, log_var2] = self.encoder2.encode(obs)
            latent_vec_2 = self.encoder2.reparameterize(mu2, log_var2)
        else:
            latent_vec_2 = obs
        
        state2_concat = np.concatenate([latent_vec_2.cpu(), relative_pos.cpu(), last_act.cpu()], axis=1)
        state2_concat = torch.tensor(state2_concat).to(torch.float32).to(latent_vec_2.device) #34
        # state2_concat = torch.tensor(state2_concat).to(torch.float64).to(latent_vec_2.device) #34

        sa2 = torch.cat([state2_concat, action], 1)

        q2 = self.head2(sa2)
        q2 = self.fc2(q2)

        return q1, q2
    
    def Q1(self, obs, last_act, relative_pos, action):
        last_act = torch.tensor(last_act).reshape((-1,2)).to(torch.float32)
        relative_pos = torch.tensor(relative_pos).reshape((-1,2)).to(torch.float32)
        # last_act = torch.tensor(last_act).reshape((-1,2)).to(torch.float64)
        # relative_pos = torch.tensor(relative_pos).reshape((-1,2)).to(torch.float64)
        if self.encoder1:
            [mu1, log_var1] = self.encoder1.encode(obs)
            latent_vec_1 = self.encoder1.reparameterize(mu1, log_var1)
        else:
            latent_vec_1 = obs

        state1_concat = np.concatenate([latent_vec_1.cpu(), relative_pos.cpu(), last_act.cpu()], axis=1)
        state1_concat = torch.tensor(state1_concat).to(torch.float32).to(latent_vec_1.device) #34
        # state1_concat = torch.tensor(state1_concat).to(torch.float64).to(latent_vec_1.device) #34

        sa1 = torch.cat([state1_concat, action], 1)

        q1 = self.head1(sa1)
        q1 = self.fc1(q1)
        return q1


class TD3(object):
    def __init__(
        self,actor,actor_optim,critic,critic_optim,action_range,
        gamma=0.99,tau=0.05,policy_noise=0.2,noise_clip=0.5,n_step=1,update_actor_freq=2,exploration_noise=0.1,
        device="cpu",
    ):
        self.actor = actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = actor_optim

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = critic_optim

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.update_actor_freq = update_actor_freq
        self.exploration_noise = exploration_noise
        self.device = device
        self.n_step = n_step

        self.total_it = 0
        self.action_range = action_range
        self._action_scale = torch.tensor((action_range[1] - action_range[0]) / 2.0, device=self.device)
        self._action_bias = torch.tensor((action_range[1] + action_range[0]) / 2.0, device=self.device)

    def select_action(self, obs, last_act, relative_pos):
        obs = torch.FloatTensor(obs).to(self.device)    #(96,128,3), just obs

        if len(obs.shape) < 3:
            obs = obs[None, :, :]
        
        #reshape for VAE input
        obs = obs.permute(2,0,1)
        obs = obs.reshape((1,3,48,64)).to(torch.float32)
        # obs = obs.reshape((1,3,48,64)).to(torch.float64)

        action = self.actor(obs, last_act, relative_pos).cpu().data.numpy().flatten()
        action += np.random.normal(0, self.exploration_noise, size=action.shape)

        action *= self._action_scale.cpu().data.numpy()
        action += self._action_bias.cpu().data.numpy()
        return action
    
    def sample_transition(self, replay_buffer, batch_size=256):
        #Sample replay buffer
        last_act, obs, action, relative_pos, next_obs, reward, next_relative_pos, not_done, ind = replay_buffer.sample(batch_size)
        #next_state, reward, not_done, gammas = replay_buffer.n_step_return(self.n_step, ind, self.gamma)
        return last_act, obs, action, relative_pos, next_obs, reward, next_relative_pos, not_done, 1.0
        #return state, action, next_state, reward, not_done, gammas
    
    def train_rl(self, last_act, obs, relative_pos, action, next_obs, reward, next_relative_pos, not_done, gammas):
        self.total_it += 1

        with torch.no_grad():
            noise = (torch.rand_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)

            #reshape for VAE input
            next_obs = next_obs.permute(0,3,1,2)

            next_action = (self.actor_target(next_obs, action, next_relative_pos) + noise).clamp(-1,1)

            #compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_obs, action, next_relative_pos, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done*self.gamma*target_Q# TD target
        
        #Get current Q estimates
        action -= self._action_bias
        action /= self._action_scale

        #reshape for VAE input
        obs = obs.permute(0,3,1,2)
        # state = state.reshape((-1,3,96,128)).to(torch.float32)

        current_Q1, current_Q2 = self.critic(obs, last_act, relative_pos, action)#TD prediction

        #Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        #Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        #Delayed policy updates
        if self.total_it % self.update_actor_freq ==0 :

            #Compute actor loss
            actor_loss = -self.critic.Q1(obs, last_act, relative_pos, self.actor(obs, last_act, relative_pos)).mean()

            #Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            #Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau)*target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau)*target_param.data)
            
        actor_loss = actor_loss.item() if actor_loss is not None else None
        critic_loss = critic_loss.item()
        return {
            #"Actor_grad_norm": self.grad_norm(self.actor),
            #"Critic_grad_norm": self.grad_norm(self.critic),
            "Actor_loss": actor_loss,
            "Critic_loss": critic_loss
        }
    
    def train(self, replay_buffer, batch_size=256):
        last_act, obs, action, relative_pos, next_obs, reward, next_relative_pos, not_done, gammas = self.sample_transition(replay_buffer, batch_size)
        loss_info = self.train_rl(last_act, obs, relative_pos, action, next_obs, reward, next_relative_pos, not_done, gammas)
        return loss_info
    
    def grad_norm(self, model):
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.data.norm(2).item() if p.grad is not None else 0 
            total_norm += param_norm ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm
    
    def save(self, dir, filename):
        self.actor.to("cpu")
        with open(join(dir, filename + "_actor"), "wb") as f:
            pickle.dump(self.actor.state_dict(), f)
        with open(join(dir, filename + "_noise"), "wb") as f:
            pickle.dump(self.exploration_noise, f)
        self.actor.to(self.device)
    
    def load(self, dir, filename):
        with open(join(dir, filename + "_actor"), "rb") as f:
            self.actor.load_state_dict(pickle.load(f))
            self.actor_target = copy.deepcopy(self.actor)
        with open(join(dir, filename + "_noise"), "rb") as f:
            self.exploration_noise = pickle.load(f)

class ReplayBuffer(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        device="cpu",
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.mean, self.std = 0.0, 1.0

        self.last_action = np.zeros((max_size, action_dim))
        self.obs = np.zeros((max_size, *state_dim))   
        self.relative_pos = np.zeros((max_size, 2))
        self.action = np.zeros((max_size, action_dim))  
        self.next_obs = np.zeros((max_size, *state_dim))  
        self.reward = np.zeros((max_size, 1))   
        self.next_relative_pos = np.zeros((max_size, 2))
        self.not_done = np.zeros((max_size, 1))

        #Reward normalization
        self.mean = None
        self.std = None

        self.device = device

    def add(self, last_action, obs, action, relative_pos, next_obs, reward, next_relative_pos, done):
        self.last_action[self.ptr] = last_action
        self.obs[self.ptr] = obs
        self.relative_pos[self.ptr] = relative_pos
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.next_relative_pos[self.ptr] = next_relative_pos
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        self.mean, self.std = 0.0, 1.0
    
    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)##ind == index

        return (
            torch.FloatTensor(self.last_action[ind]).to(self.device),
            torch.FloatTensor(self.obs[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.relative_pos[ind]).to(self.device),
            torch.FloatTensor(self.next_obs[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_relative_pos[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind
        )
        
    # def n_step_return(self, n_step, ind, gamma):
    #     reward = []
    #     not_done = []
    #     next_state = []
    #     gammas = []
        
    #     for i in ind:
    #         n = 0
    #         r = 0
    #         c = 0
    #         for _ in range(n_step):
    #             idx = (i+n) % self.size
    #             assert self.mean is not None
    #             assert self.std is not None
    #             r += (self.reward[idx] - self.mean) / self.std * gamma ** n
    #             if not self.not_done[idx]:
    #                 break
    #             n = n + 1
    #         next_state.append(self.next_state[idx])
    #         not_done.append(self.not_done[idx])
    #         reward.append(r)
    #         gammas.append([gamma ** (n+1)])
        
    #     next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
    #     not_done = torch.FloatTensor(np.array(not_done)).to(self.device)
    #     reward = torch.FloatTensor(np.array(reward)).to(self.device)
    #     gammas = torch.FloatTensor(np.array(gammas)).to(self.device)

    #     return next_state, reward, not_done, gammas