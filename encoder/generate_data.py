import os
import sys
import argparse
import numpy as np
import gym
import cv2

encoder_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(encoder_dir,".."))   #for import envs
import envs.registration

np.random.seed(888)

def main(args):
    env = gym.make(
        id = 'motion_control_continuous_laser-v0',
    )

    total_episodes = args.total_episodes
    max_size = args.max_size

    high = env.action_space.high 
    low = env.action_space.low
    bias = (high + low) /2
    scale = (high - low) /2

    init_position = [[2.0, -2.0, np.pi],[0.0, -2.0, np.pi/2],[-3.0, -3.0, np.pi/2], [-1.0, 0.0, 0.0],[2.0, 0.0, 0.0],[2.0, 2.0, np.pi],
    [0.0, 2.0, -np.pi/2], [-2.0,3.0, 0.0]]  #for office_small
    #init_position = [[0.0, -3.0, np.pi/2], [-2.0, -3.0, np.pi/2]]  #for maze1

    save_dir = os.path.join(encoder_dir, "dataset")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episode = 0
    total_time_steps = 0
    start_idx = 0   #for choosing the init_positon
    while episode < total_episodes:
        print('---------------------------')
        start_idx = start_idx % len(init_position)
        print('init postion is ', init_position[start_idx])
        env.reset_init_model_state(init_position[start_idx])
        obs = env.reset()

        time_step = 0
        done = False
        while not done:
            actions = 2*(np.random.rand(env.action_space.shape[0]) - 0.5)
            actions *= scale
            actions += bias

            time_step += 1

            obs, rew, done, info = env.step(actions)
            image_name = 'episode_' + str(episode) + '_time_step_' + str(time_step) + '.jpg'
            print("Saving {} to dataset ...".format(image_name))
            cv2.imwrite(save_dir + "/" + image_name, obs)
            
        total_time_steps += time_step
        if (total_time_steps >= max_size):
            break
        episode += 1
        start_idx +=1

        print("Current dataset contains {} observations".format(total_time_steps))
        
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate new training data')
    parser.add_argument('--total_episodes', type=int, default=10000, help='total record episodes')
    parser.add_argument('--max_size', type=int, default=10000, help='save image nums')

    args = parser.parse_args()
    main(args)