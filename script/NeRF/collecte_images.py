import os
import sys
import argparse
import numpy as np
import gym
import cv2

nerf_dir = os.path.dirname(__file__)
sys.path.append(os.path.join(nerf_dir,".."))   #for import envs
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

    init_position = [np.array([0.0, -3.0, np.pi])]

    save_dir = os.path.join(nerf_dir, "datasets")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    episode = 0
    total_time_steps = 0
    start_idx = 0   #for choosing the init_positon
    image_infos = []
    while episode < total_episodes:
        print('---------------------------')
        start_idx = start_idx % len(init_position)
        # print('init postion is ', init_position[start_idx])
        env.reset_init_model_state(init_position[start_idx])
        obs = env.reset()

        time_step = 0
        done = False
        
        while not done:
            actions = 2*(np.random.rand(env.action_space.shape[0]) - 0.5)
            actions *= scale
            actions += bias

            actions /= 50
            time_step += 1

            obs, rew, done, info = env.step(actions)
            image_name = 'episode_' + str(episode) + '_time_step_' + str(time_step) + '.jpg'

            if (episode >= 10):
                image_info = np.concatenate([info['camera_pos'], info['camera_quat']], axis=0)
                image_info = image_info.reshape((1,7))
                image_infos.append(image_info)
                f_id = open(nerf_dir+"/image_id.txt", "a")
                f_id.write(image_name + "\r\n")
                print("Saving {} to dataset ...".format(image_name))
                cv2.imwrite(save_dir + "/" + image_name, obs)
                
        
        if (episode >= 10):    
            total_time_steps += time_step

        if (total_time_steps >= max_size):
            break
        episode += 1
        start_idx +=1

        print("Current dataset contains {} observations".format(total_time_steps))
    
    env.close()
    for image_info in image_infos:
        with open(nerf_dir+"/image_info.txt", "a") as f:
            np.savetxt(f, image_info, newline='\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate new training data')
    parser.add_argument('--total_episodes', type=int, default=1000, help='give a big nums')
    parser.add_argument('--max_size', type=int, default=100, help='real image nums you want to save')

    args = parser.parse_args()
    main(args)