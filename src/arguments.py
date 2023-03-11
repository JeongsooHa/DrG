import argparse
import numpy as np
from numpy.lib.index_tricks import _fill_diagonal_dispatcher
from torch.nn import functional as F

def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
	parser = argparse.ArgumentParser()

	# environment
	parser.add_argument('--domain_name', default='walker') # 'walker / cheetah'
	parser.add_argument('--task_name', default='walk')		# 'walk / run'
	parser.add_argument('--frame_stack', default=1, type=int) 
	parser.add_argument('--action_repeat', default=2, type=int) 
	parser.add_argument('--episode_length', default=1000, type=int)
	parser.add_argument('--train_mode', default='train', type=str)
	parser.add_argument('--eval_mode', default='video_hard', type=str)
	
	# agent
	parser.add_argument('--algorithm', default='drg', type=str) 
	parser.add_argument('--train_steps', default='800k', type=str)
	parser.add_argument('--discount', default=0.99, type=float)
	parser.add_argument('--init_steps', default=1000, type=int)
	parser.add_argument('--batch_size', default=128, type=int)
	parser.add_argument('--hidden_dim', default=1024, type=int)

	# actor
	parser.add_argument('--actor_lr', default=1e-3, type=float)
	parser.add_argument('--actor_beta', default=0.9, type=float)
	parser.add_argument('--actor_log_std_min', default=-10, type=float)
	parser.add_argument('--actor_log_std_max', default=2, type=float)
	parser.add_argument('--actor_update_freq', default=2, type=int)

	# critic
	parser.add_argument('--critic_lr', default=1e-3, type=float)
	parser.add_argument('--critic_beta', default=0.9, type=float)
	parser.add_argument('--critic_tau', default=0.01, type=float)
	parser.add_argument('--critic_target_update_freq', default=2, type=int)

	# architecture
	parser.add_argument('--num_shared_layers', default=11, type=int)
	parser.add_argument('--num_head_layers', default=0, type=int)
	parser.add_argument('--num_filters', default=32, type=int)
	parser.add_argument('--projection_dim', default=100, type=int)
	parser.add_argument('--encoder_tau', default=0.05, type=float)
	
	# entropy maximization
	parser.add_argument('--init_temperature', default=0.1, type=float)
	parser.add_argument('--alpha_lr', default=1e-4, type=float)
	parser.add_argument('--alpha_beta', default=0.5, type=float)

	# auxiliary tasks
	parser.add_argument('--aux_lr', default=1e-3, type=float)
	parser.add_argument('--aux_beta', default=0.9, type=float)
	parser.add_argument('--aux_update_freq', default=2, type=int)

	# soda
	parser.add_argument('--soda_batch_size', default=256, type=int)
	parser.add_argument('--soda_tau', default=0.005, type=float)

	# svea
	parser.add_argument('--svea_alpha', default=0.5, type=float)
	parser.add_argument('--svea_beta', default=0.5, type=float)

	# eval
	parser.add_argument('--save_freq', default='100k', type=str)
	parser.add_argument('--eval_freq', default='20k', type=str, help='drg')
	parser.add_argument('--eval_episodes', default=30, type=int)
	parser.add_argument('--eval_num', default=1, type=int)
	parser.add_argument('--test_interval', type=int, default=20, metavar='I', help='Test interval (episodes) original 10')
	parser.add_argument('--test_episodes', type=int, default=10, metavar='E', help='Number of test episodes original 10, drg')
	parser.add_argument('--distracting_cs_intensity', default=0., type=float)

	# misc
	parser.add_argument('--seed', default=1, type=int) # None
	parser.add_argument('--log_dir', default='logs/debug2', type=str)
	parser.add_argument('--save_video', default=True, action='store_true')

	# drg
	parser.add_argument('--observation_loss', type=str2bool, default=False)
	parser.add_argument('--back_transition_loss', type=str2bool, default=False)
	
	parser.add_argument('--inv_posterior_loss', type=str2bool, default=False)
	parser.add_argument('--inv_belief_loss', type=str2bool, default=False)
	parser.add_argument('--contrast_post_loss', type=str2bool, default=False)
	parser.add_argument('--contrast_belief_loss', type=str2bool, default=False)

	parser.add_argument('--state_poster_cos_sim_loss', type=str2bool, default=False)
	parser.add_argument('--latent_cosine_similarity', type=str2bool, default=False)
	parser.add_argument('--latent_loss_type', type=int, default=1)
	
	# main loss arguments
	parser.add_argument('--worldmodel_LogProbLoss', type=str2bool, default=True, help='use LogProb loss for observation_model and reward_model training')
	parser.add_argument('--inv_belief_post_loss', type=str2bool, default=True)
	parser.add_argument('--contrast_encoder_loss', type=str2bool, default=True)
	parser.add_argument('--contrast_belief_post_loss', type=str2bool, default=True)
	parser.add_argument('--aug_loss', type=str2bool, default=True)
	parser.add_argument('--cross', type=str2bool, default=False)
	
	parser.add_argument('--contrast_obs_target_loss', type=str2bool, default=False)
	parser.add_argument('--target_transition_loss', type=str2bool, default=False)

	parser.add_argument('--test_model', type=int, default=1000)

    # main augmentations arguments
	parser.add_argument('--same_action', type=str2bool, default=False)
	parser.add_argument('--ran', type=str2bool, default=False)
	parser.add_argument('--ran_2', type=str2bool, default=False)
	parser.add_argument('--ran_3', type=str2bool, default=False)
	parser.add_argument('--ran_4', type=str2bool, default=False)
	parser.add_argument('--cc', type=str2bool, default=False, help='cutout_color')
	parser.add_argument('--cj', type=str2bool, default=False, help='color_jitter')
	parser.add_argument('--ol', type=str2bool, default=False, help='overlay')
	parser.add_argument('--con', type=str2bool, default=False, help='random_conv')
	parser.add_argument('--gs', type=str2bool, default=False, help='gray_scale')
	
	parser.add_argument('--hard_2', type=str2bool, default=False, help='***_hard')
	parser.add_argument('--none_2', type=str2bool, default=False, help='***_none')

	# kl loss	
	parser.add_argument('--use_adv_loss', type=str2bool, default=False)
	parser.add_argument('--use_kl_balancing', type=str2bool, default=True)
	parser.add_argument('--kl_balance_scale', type=float, default=0.8)
	parser.add_argument('--kl_loss_scaling', type=float, default=0.1)
	parser.add_argument('--use_free_nats', type=str2bool, default=False)
	parser.add_argument('--free_nats', type=float, default=3, metavar='F', help='Free nats')

	# parser.add_argument('--max_episode_length', type=int, default=1000, metavar='T', help='Max episode length')
	parser.add_argument('--experience_size', type=int, default=100000, metavar='D', help='Experience replay size original 1000000')  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
	parser.add_argument('--cnn_activation_function', type=str, default='relu', choices=dir(F), help='Model activation function for a convolution layer')
	parser.add_argument('--dense_activation_function', type=str, default='elu', choices=dir(F), help='Model activation function a dense layer')
	parser.add_argument('--embedding_size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
	
	parser.add_argument('--hidden_size', type=int, default=200, metavar='H', help='Hidden size')
	parser.add_argument('--belief_size', type=int, default=200, metavar='H', help='Belief/hidden size $$deterministic state$$')
	parser.add_argument('--state_size', type=int, default=30, metavar='Z', help='State/latent size $$stochasitc state$$')
	
	# parser.add_argument('--dreamer_action_repeat', type=int, default=2, metavar='R', help='Action repeat')
	parser.add_argument('--action_noise', type=float, default=0.3, metavar='ε', help='Action noise')
	# parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
	parser.add_argument('--seed_episodes', type=int, default=5, metavar='S', help='Seed episodes  original 5')
	parser.add_argument('--collect_interval', type=int, default=100, metavar='C', help='Collect interval')
	parser.add_argument('--dreamer_batch_size', type=int, default=50, metavar='B', help='Batch size / original = 50, debug = 10')
	parser.add_argument('--chunk_size', type=int, default=50, metavar='L', help='Chunk size')
	
	parser.add_argument('--overshooting_distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
	parser.add_argument('--overshooting_kl_beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
	parser.add_argument('--overshooting_reward_scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
	parser.add_argument('--global_kl_beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
	parser.add_argument('--bit_depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
	parser.add_argument('--model_learning_rate', type=float, default=1e-3, metavar='α', help='Learning rate') 
	parser.add_argument('--actor_learning_rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
	parser.add_argument('--value_learning_rate', type=float, default=8e-5, metavar='α', help='Learning rate') 
	parser.add_argument('--learning_rate_schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
	parser.add_argument('--adam_epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value') 
	# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
	parser.add_argument('--grad_clip_norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
	parser.add_argument('--planning_horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
	# parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
	parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
	parser.add_argument('--optimisation_iters', type=int, default=10, metavar='I', help='Planning optimisation iterations')
	parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
	parser.add_argument('--top_candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
	parser.add_argument('--test', action='store_true', help='Test only')
	
	parser.add_argument('--checkpoint_interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
	parser.add_argument('--checkpoint_experience', action='store_true', help='Checkpoint experience replay')
	parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
	parser.add_argument('--experience_replay', type=str, default='', metavar='ER', help='Load experience replay')
	parser.add_argument('--render', action='store_true', help='Render environment')

	args = parser.parse_args()

	assert args.algorithm in {
		'sac', 'rad', 'curl', 'pad', 'soda', 'drq', 'svea', \
		'dreamer', 'drg'}, f'specified algorithm "{args.algorithm}" is not supported'

	assert args.train_mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'video_hard_train', 'video_hard_test', 'video_driving_train', 'video_driving_test', 'distracting_cs', 'none'}, f'specified mode "{args.train_mode}" is not supported'
	assert args.eval_mode in {'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'video_hard_train', 'video_hard_test', 'video_driving_train', 'video_driving_test', 'distracting_cs', 'none'}, f'specified mode "{args.eval_mode}" is not supported'
	assert args.seed is not None, 'must provide seed for experiment'
	assert args.log_dir is not None, 'must provide a log directory for experiment'

	intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
	assert args.distracting_cs_intensity in intensities, f'distracting_cs has only been implemented for intensities: {intensities}'

	args.train_steps = int(args.train_steps.replace('k', '000'))
	args.save_freq = int(args.save_freq.replace('k', '000'))
	args.eval_freq = int(args.eval_freq.replace('k', '000'))

	if args.eval_mode == 'none':
		args.eval_mode = None

	if args.domain_name == "finger" and args.task_name == "spin":
		args.action_repeat = 2
	elif args.domain_name == "cartpole" and args.task_name == "swingup":
		args.action_repeat = 8
	elif args.domain_name == 'walker' and args.task_name == 'walk' and args.algorithm == 'curl':
		args.action_repeat = 2
	else:
		args.action_repeat = 4

	if args.use_adv_loss:
		args.use_kl_balancing = False
	
	if args.algorithm in {'rad', 'curl', 'pad', 'soda'}:
		args.image_size = 100
		args.image_crop_size = 84
		if args.algorithm == 'curl':
			args.aux_update_freq = 1
		else:
			args.aux_update_freq = 2
			
	elif args.algorithm == 'dreamer' or args.algorithm == 'drg':
		args.image_size = 64
		args.image_crop_size = 64
		args.frame_stack = 1
		if args.same_action:
			args.action_repeat = 2
		else:
			if args.domain_name == 'cartpole':
				args.action_repeat = 4
			elif args.domain_name == 'reacher':
				args.action_repeat = 4
			elif args.domain_name == 'cheetah':
				args.action_repeat = 4
			elif args.domain_name == 'finger':
				args.action_repeat = 1
			elif args.domain_name == 'ball_in_cup':
				args.action_repeat = 4
			elif args.domain_name == 'walker':
				args.action_repeat = 2
		
		if args.use_kl_balancing:
			args.kl_loss_scaling = 0.1
			args.use_free_nats = False
		else:
			args.kl_loss_scaling = 1
			args.use_free_nats = True
	else:
		args.image_size = 84
		args.image_crop_size = 84

	return args
