import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder

def evaluate(env, agent, video, num_episodes, L, step, test_env):
	episode_rewards = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, _ = env.step(action)
			video.record(env)
			episode_reward += reward

		if L is not None:
			# _test_env = '_test_env' if test_env else ''
			video.save(f'{step}_{test_env}.mp4')
			L.log(f'eval/episode_reward_{test_env}', episode_reward, step)
		episode_rewards.append(episode_reward)
	
	return np.mean(episode_rewards)


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		frame_stack=args.frame_stack,
		mode='train'
	)
	test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		frame_stack=args.frame_stack,
		mode='video_hard_test',
		intensity=args.distracting_cs_intensity
	)
	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')), 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	log_dir = utils.make_dir(os.path.join(work_dir, 'logs'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		args,
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.experience_size, #args.train_steps,
		batch_size=args.batch_size,
		bit_depth=args.bit_depth,
		path_length = args.episode_length // args.action_repeat
	)
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	test_env_lst = [test_env]#, test_vn_env]
	# test_env_lst = [test_de_env, test_ch_env, test_ve_env, test_vn_env]#, test_vn_env]
	start_step, episode, episode_reward, done = 0, 0, 0, True
	L = Logger(work_dir)
	start_time = time.time()
	if args.algorithm == 'dreamer' or args.algorithm == 'drg':
		agent.dreamer_train(L, start_time, env, test_env_lst, replay_buffer, work_dir)
	else:
		# 'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs'
		
		for step in range(start_step, args.train_steps+1):
			if done:
				if step > start_step:
					L.log('train/duration', time.time() - start_time, step)
					start_time = time.time()
					L.dump(step)

				# Evaluate agent periodically
				if step % args.eval_freq == 0:
					print('Evaluating:', work_dir)
					L.log('eval/episode', episode, step)

					evaluate(test_env, agent, video, args.eval_episodes, L, step, "de")
					# evaluate(test_vn_env, agent, video, args.eval_episodes, L, step, "vn")

					L.dump(step)

				# Save agent periodically
				if step > start_step and step % args.save_freq == 0:
					torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

				L.log('train/episode_reward', episode_reward, step)

				obs = env.reset()
				done = False
				episode_reward = 0
				episode_step = 0
				episode += 1

				L.log('train/episode', episode, step)

			# Sample action for data collection
			if step < args.init_steps:
				action = env.action_space.sample()
			else:
				with utils.eval_mode(agent):
					action = agent.sample_action(obs)

			# Run training update
			if step >= args.init_steps:
				num_updates = args.init_steps if step == args.init_steps else 1
				for _ in range(num_updates):
					agent.update(replay_buffer, L, step)

			# Take step
			next_obs, reward, done, _ = env.step(action)
			done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)
			replay_buffer.add(obs, action, reward, next_obs, done_bool)
			episode_reward += reward
			obs = next_obs

			episode_step += 1

		print('Completed training for', work_dir)


if __name__ == '__main__':
	args = parse_args()
	main(args)
