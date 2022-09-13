#Environment 
maze_id = 1
deterministic = False
continuous = True
#Image
input_dim = (48, 64, 3)
#reward params
r_arrive = 100
r_collision = -50
r_move = 100
r_time_penalty = -0.05
#goal
goal_space_maze1 = [[-3.8, -3.8], [-1.5, -4], [-3, -2.2]]
start_space_maze1 = [[-0.4, -2.6]]
Cd = 0.345/2
#max linear velocity
v_max = 0.3 #m/s
#max angular velocity
w_max = 1.0

#VAE
latent_dim = 32
vae_weight = ''

#RNN
dir_name = 'maze1_rnn_data'
rnn_weight  = ''
episode = 2
frame = 25