import numpy as np

class LocalCollector(object):
    def __init__(self, policy, env, replaybuffer):
        self.policy = policy
        self.env = env
        self.buffer = replaybuffer

        self.last_obs = None
        self.last_act = np.array([0.0, 0.0])
        self.last_relative_pos = np.array([-3.8, -3.8])-np.array([0.0, -3.0])
        
        self.global_episodes = 0
        self.global_steps = 0

    def collect(self, n_steps):
        env = self.env
        policy = self.policy
        

        n_steps_curr = 0
        ep_rew = 0
        ep_len = 0
        results = []

        if self.last_obs is not None:
            obs = self.last_obs
            last_act = self.last_act
            relative_pos = self.last_relative_pos
        else:
            print("last_obs is None")
            obs = env.reset()
            last_act = np.array([0.0, 0.0])
            relative_pos = np.array([-3.8, -3.8])-np.array([0.0, -3.0])
        
        while n_steps_curr < n_steps:
            act = policy.select_action(obs, last_act, relative_pos)#state = [obs, last_act, relative_pos]
            #print("action is : ", act)

            obs_new, rew, done, info = env.step(act)
            relative_pos_new = info['relative_position']

            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1

            self.buffer.add(last_act, obs, act, relative_pos, obs_new, rew, relative_pos_new, done,)

            obs = obs_new
            last_act = act
            relative_pos = relative_pos_new

            if done:
                obs = env.reset()
                last_act = np.array([0.0, 0.0])
                relative_pos = np.array([-3.8, -3.8])-np.array([0.0, -3.0])
                results.append(dict(
                    ep_rew=ep_rew,
                    ep_len=ep_len,
                    success=info['success'],
                    ep_time=info['time'],
                    collision=info['collided']
                ))
                ep_rew = 0
                ep_len = 0
                self.global_episodes += 1
            print("n_episode: %d, n_steps: %d"%(self.global_episodes, self.global_steps), end="\r")
            
        self.last_act = last_act
        self.last_relative_pos = relative_pos
        self.last_obs = obs
        return n_steps_curr, results