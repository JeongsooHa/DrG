import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations

def evaluate_drg(args, env, agent, video, eval_mode):
	episode_reward_lst = []
	# for i in range(args.test_episodes):
	for i in range(1):
		action_lst = []
		observation, episode_reward, video_frames = env.reset(), 0, []
		video.init(enabled=True)
		belief, posterior_state, action = \
			torch.zeros(1, args.belief_size).cuda(), \
			torch.zeros(1, args.state_size).cuda(), \
			torch.zeros(1, env.action_space.shape[0]).cuda()

		for step in range(args.episode_length // args.action_repeat):
			belief, posterior_state, action, next_observation, reward, done = agent.update_belief_and_act(env, belief, posterior_state, action, observation)
			video.record(env, eval_mode)
			episode_reward += reward
			action_lst.append(action.detach().cpu().numpy())
			observation = next_observation
		np.save('graduate_ppt/action_lst.npy',np.array(action_lst))
		video.save(f'eval_{eval_mode}_{i}.mp4')
		print(episode_reward)
		episode_reward_lst.append(episode_reward)

	return np.mean(episode_reward_lst), episode_reward_lst

def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False):
	episode_rewards = []
	for i in tqdm(range(num_episodes)):
		if adapt:
			ep_agent = deepcopy(agent)
			ep_agent.init_pad_optimizer()
		else:
			ep_agent = agent
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(ep_agent):
				action = ep_agent.select_action(obs)
			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			if adapt:
				ep_agent.update_inverse_dynamics(*augmentations.prepare_pad_batch(obs, next_obs, action))
			obs = next_obs

		video.save(f'eval_{eval_mode}_{i}.mp4')
		episode_rewards.append(episode_reward)

	return np.mean(episode_rewards)

def evaluate_random(env, video, num_episodes, eval_mode):
	episode_rewards = []
	for i in tqdm(range(num_episodes)):
		
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			action = env.action_space.sample()
			next_obs, reward, done, _ = env.step(action)
			video.record(env, eval_mode)
			episode_reward += reward
			obs = next_obs

		video.save(f'eval_{eval_mode}_{i}.mp4')
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
		seed=args.seed+42,
		episode_length=args.episode_length,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		frame_stack=args.frame_stack,
		mode=args.eval_mode,
		# mode='train',
		intensity=args.distracting_cs_intensity
	)
	# Set working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, str(args.seed))
	print('Working directory:', work_dir)
	assert os.path.exists(work_dir), 'specified working directory does not exist'
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video_driving'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

	# Check if evaluation has already been run
	if args.eval_mode == 'distracting_cs':
		results_fp = os.path.join(work_dir, args.eval_mode+'_'+str(args.distracting_cs_intensity).replace('.', '_')+'.pt')
	else:
		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
	# assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	print('Observations:', env.observation_space.shape)
	print('Cropped observations:', cropped_obs_shape)
	agent = make_agent(
		obs_shape=cropped_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)
	if args.algorithm == 'drg':
		agent.load_model(os.path.join(model_dir, 'models_'+str(args.test_model)+'.pth'))
		agent.eval()
	else:
		model_dir = utils.make_dir(os.path.join(work_dir, 'logs'))
		agent = torch.load(os.path.join(model_dir, str(args.train_steps)+'.pt'))
		agent.train(False)

	print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
	if args.algorithm == 'drg':
		reward, reward_lst = evaluate_drg(args, env, agent, video, args.eval_mode)
		print("min {}, max {}".format(np.min(reward_lst), np.max(reward_lst)))
		np.save(os.path.join(work_dir, args.eval_mode+'_'+str(args.distracting_cs_intensity).replace('.', '_')+'.npy'), np.array(reward_lst))
	else:
		reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode)
	print('Reward:', int(reward))

	adapt_reward = None
	if args.algorithm == 'pad':
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+42,
			episode_length=args.episode_length,
			action_repeat=args.action_repeat,
			mode=args.eval_mode
		)
		adapt_reward = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, adapt=True)
		print('Adapt reward:', int(adapt_reward))

	# Save results
	torch.save({
		'args': args,
		'reward': reward,
		'adapt_reward': adapt_reward
	}, results_fp)
	print('Saved results to', results_fp)


if __name__ == '__main__':
	args = parse_args()
	main(args)
