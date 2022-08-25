import numpy as np

class LocalCollector(object):
    def __init__(self, policy, env, replaybuffer):
        self.policy = policy
        self.env = env
        self.buffer = replaybuffer

        self.last_obs = None
        
        self.global_episodes = 0
        self.global_steps = 0

    def collect(self, n_steps):
        n_steps_curr = 0
        env = self.env
        policy = self.policy
        results = []

        ep_rew = 0
        ep_len = 0

        if self.last_obs is not None:
            obs = self.last_obs
        else:
            print("last_obs is None")
            obs = env.reset()
        while n_steps_curr < n_steps:
            act = policy.select_action(obs)
            obs_new, rew, done, info = env.step(act)
            obs = obs_new
            ep_rew += rew
            ep_len += 1
            n_steps_curr += 1
            self.global_steps += 1

            self.buffer.add(
                obs, act,
                obs_new, rew,
                done
            )

            if done:
                obs = env.reset()
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
        self.last_obs = obs
        return n_steps_curr, results
