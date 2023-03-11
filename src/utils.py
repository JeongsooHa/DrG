import torch
import numpy as np
import os
import glob
import json
import random
import augmentations
import subprocess
from datetime import datetime
from typing import Iterable
from torch.nn import Module
import cv2
from env.wrappers import make_env
import time
class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)

def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def listdir(dir_path, filetype='jpg', sort=True):
	fpath = os.path.join(dir_path, f'*.{filetype}')
	fpaths = glob.glob(fpath, recursive=True)
	if sort:
		return sorted(fpaths)
	return fpaths


def prefill_memory(obses, capacity, obs_shape):
	"""Reserves memory for replay buffer"""
	c,h,w = obs_shape
	for _ in range(capacity):
		frame = np.ones((3,h,w), dtype=np.uint8)
		obses.append(frame)
	return obses

# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
	observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)  # Quantise to given bit depth and centre
	observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)

# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
	return np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1).astype(np.uint8)


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, args, obs_shape, action_shape, capacity, batch_size, bit_depth, prefill=True, path_length=None):
		self.args = args
		self.capacity = capacity
		self.batch_size = batch_size
		self._path_len = path_length
		self._obses = []
		if prefill:
			self._obses = prefill_memory(self._obses, capacity, obs_shape)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)
		
		if self.args.algorithm == 'drg' and (self.args.ol or self.args.ran or self.args.ran_2 or self.args.ran_4):
			augmentations._load_places(batch_size=self.args.dreamer_batch_size*self.args.chunk_size, image_size=self.args.image_size)
		
		self.idx = 0
		self.full = False
		self.bit_depth = bit_depth


	def add(self, obs, action, reward, next_obs, done):
		obses = (obs, next_obs)
		if self.idx >= len(self._obses):
			self._obses.append(obses)
		else:
			self._obses[self.idx] = (obses)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def _encode_obses(self, idxs):
		obses, next_obses = [], []
		for i in idxs:
			obs, next_obs = self._obses[i]
			obses.append(np.array(obs, copy=False))
			next_obses.append(np.array(next_obs, copy=False))
		return np.array(obses), np.array(next_obses)

	def sample_soda(self, n=None):
		idxs = self._get_idxs(n)
		obs, _ = self._encode_obses(idxs)
		return torch.as_tensor(obs).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample_drq(self, n=None, pad=4):
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_shift(obs, pad)
		next_obs = augmentations.random_shift(next_obs, pad)

		return obs, actions, rewards, next_obs, not_dones
	
	def sample(self, n=None):
		
		idxs = self._get_idxs(n)

		obs, next_obs = self._encode_obses(idxs)
		obs = torch.as_tensor(obs).cuda().float()
		next_obs = torch.as_tensor(next_obs).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones

	

	def _retrieve_batch(self, idxs, n, L):
		vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
		observations, _ = self._encode_obses(vec_idxs)
		# observations = torch.as_tensor(observations.astype(np.float32))
		# observations = torch.as_tensor(self._obses[vec_idxs].astype(np.float32))
		# if not self.symbolic_env:
		# preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations
		return observations.reshape(L, n, *observations.shape[1:]).astype(np.float32), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n), self.not_dones[vec_idxs].reshape(L, n, 1)
	
	def _sample_idx(self, L):
		valid_idx = False
		while not valid_idx:
			idx = np.random.randint(0, self.capacity if self.full else self.idx - L)
			idxs = np.arange(idx, idx + L) % self.capacity
			valid_idx = not self.idx in idxs[1:]  # Make sure data does not cross the memory index
		return idxs

	# Returns a batch of sequence chunks uniformly sampled from the memory
	def dreamer_sample(self, n, L):
		batch = self._retrieve_batch(np.asarray([self._sample_idx(L) for _ in range(n)]), n, L)
		# print(np.asarray([self._sample_idx(L) for _ in range(n)]))
		# [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
		# [1049 1050 1051 ... 1096 1097 1098]
		# [1236 1237 1238 ... 1283 1284 1285]
		# ...
		# [2199 2200 2201 ... 2246 2247 2248]
		# [ 686  687  688 ...  733  734  735]
		# [1377 1378 1379 ... 1424 1425 1426]]
		
		observations = torch.as_tensor(batch[0]).cuda().float()
		actions = torch.as_tensor(batch[1]).cuda().float()
		rewards = torch.as_tensor(batch[2]).cuda().float()
		nonterminals = torch.as_tensor(batch[3]).cuda().float()
		
		# return_batch = [torch.as_tensor(item).cuda().float() for item in batch]
		preprocess_observation_(observations, self.bit_depth)
		return observations, actions, rewards, nonterminals
	
	def sequential_augmentation(self, aug_function, observations):
		seq, batch, channel, w,h = observations.shape
		for batch_idx in range(batch):
			observations[:,batch_idx] = aug_function(observations[:,batch_idx])
		return observations
	
	def _sample_sequential_idx(self, n, L):
        # Returns an index for a valid single chunk uniformly sampled from the
        # memory
		idx = np.random.randint(
			0, self.capacity - L if self.full else self.idx - L, size=n
		)
		pos_in_path = idx - idx // self._path_len * self._path_len
		idx[pos_in_path > self._path_len - L] = idx[
			pos_in_path > self._path_len - L
		] // self._path_len * self._path_len + L
		idxs = np.zeros((n, L), dtype=np.int)
		for i in range(n):
			idxs[i] = np.arange(idx[i], idx[i] + L)
		return idxs.transpose().reshape(-1)

	def drg_sample(self, n, L):
		
    	# random_overlay, random_conv, gray_scale, cut_out, cutout_color, color_jitter
		
		idxs 		= self._sample_sequential_idx(n, L)
		
		obses, _	= self._encode_obses(idxs)

		obses 		= torch.as_tensor(obses).cuda().float()
		actions 	= torch.as_tensor(self.actions[idxs]).cuda().float()
		rewards 	= torch.as_tensor(self.rewards[idxs]).cuda().float()
		not_dones 	= torch.as_tensor(self.not_dones[idxs]).cuda().float()
		
		aug_obses = augmentations.random_shift(obses.clone())
		
		if self.args.cc:
			aug_obses = augmentations.cutout_color(aug_obses)
		if self.args.cj:
			aug_obses = augmentations.color_jitter(aug_obses)
		if self.args.ol:
			aug_obses = augmentations.random_overlay(aug_obses)
		if self.args.con:
			aug_obses = augmentations.random_conv(aug_obses)
		if self.args.gs:
			aug_obses = augmentations.gray_scale(aug_obses)
		if self.args.ran:
			if random.randint(0, 1):
				aug_obses = augmentations.random_conv(aug_obses)
			else:
				aug_obses = augmentations.random_overlay(aug_obses)
		if self.args.ran_2:
			if random.randint(0, 1):
				aug_obses = augmentations.cutout_color(aug_obses)
			else:
				aug_obses = augmentations.random_overlay(aug_obses)
		if self.args.ran_3:
			if random.randint(0, 1):
				aug_obses = augmentations.cutout_color(aug_obses)
			else:
				aug_obses = augmentations.random_conv(aug_obses)
		if self.args.ran_4:
			idx = random.randint(0, 2)
			if idx == 0:
				aug_obses = augmentations.cutout_color(aug_obses)
			elif idx == 1:
				aug_obses = augmentations.random_conv(aug_obses)
			elif idx == 2:
				aug_obses = augmentations.random_conv(aug_obses)
			
		if self.args.hard_2:
			obses = augmentations.random_shift(obses)
			obses = augmentations.random_overlay(obses)
		elif self.args.none_2:
			pass
		else:
			obses = augmentations.random_shift(obses)

		obses 		= obses.reshape(L, n, *obses.shape[1:])
		aug_obses 	= aug_obses.reshape(L, n, *aug_obses.shape[1:])
		actions 	= actions.reshape(L, n, -1)
		rewards 	= rewards.reshape(L, n)
		not_dones 	= not_dones.reshape(L, n, 1)

		preprocess_observation_(obses, self.bit_depth)
		preprocess_observation_(aug_obses, self.bit_depth)
	
		return obses, aug_obses, actions, rewards, not_dones


def random_crop(imgs, out=64):
    """
        args:
        imgs: np.array shape (B,C,H,W)
        out: output size (e.g. 84)
        returns np.array
    """
    n, c, h, w = imgs.shape
    crop_max = h - out + 1
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    cropped = np.empty((n, c, out, out), dtype=imgs.dtype)
    for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
        cropped[i] = img[:, h11:h11 + out, w11:w11 + out]
    return cropped

# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

def write_video(frames, title, path=''):
	frames = np.multiply(np.stack(frames, axis=0).transpose(0, 2, 3, 1), 255).clip(0, 255).astype(np.uint8)[ ::-1]  # VideoWrite expects H x W x C in BGR
	_, H, W, _ = frames.shape
	writer = cv2.VideoWriter(os.path.join(path, '%s.mp4' % title), cv2.VideoWriter_fourcc(*'mp4v'), 30., (W, H), True)
	for frame in frames:
		writer.write(frame)
	writer.release()

class FreezeParameters:
	def __init__(self, modules: Iterable[Module]):
		"""
		Context manager to locally freeze gradients.
		In some cases with can speed up computation because gradients aren't calculated for these listed modules.
		example:
		```
		with FreezeParameters([module]):
		output_tensor = module(input_tensor)
		```
		:param modules: iterable of modules. used to call .parameters() to freeze gradients.
		"""
		self.modules = modules
		self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

	def __enter__(self):
		for param in get_parameters(self.modules):
			param.requires_grad = False

	def __exit__(self, exc_type, exc_val, exc_tb):
		for i, param in enumerate(get_parameters(self.modules)):
			param.requires_grad = self.param_states[i]


class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns total number of params in a network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'

class EnvBatcher():
    # (self.args, eval_mode, self.args.test_episodes)
	def __init__(self, args, eval_mode, n):
		self.args = args
		self.n = n
		if eval_mode == 'de':
			mode = 'train'
		elif eval_mode == 'ce':
			mode = 'color_easy'
		elif eval_mode == 'ch':
			mode = 'color_hard'
		elif eval_mode == 've':
			mode = 'video_easy'
		elif eval_mode == 'vh':
			mode = 'video_hard'
		self.envs = [
			make_env(
				domain_name=args.domain_name,
				task_name=args.task_name,
				seed=args.seed+42+i,
				episode_length=args.episode_length,
				action_repeat=args.action_repeat,
				image_size=args.image_size,
				frame_stack=args.frame_stack,
				mode=mode,
				intensity=args.distracting_cs_intensity,
				multi=True
			) for i in range(n)]
		self.dones = [True] * n

	# Resets every environment and returns observation
	def reset(self):
		observations = [env.reset()for env in self.envs]
		observations = np.array(observations).astype(np.float32)
		self.dones = [False] * self.n
		return observations

	# Steps/resets every environment and returns (observation, reward, done)
	def step(self, actions):
		done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]  # Done mask to blank out observations and zero rewards for previously terminated environments
		observations, rewards, dones, _ = zip(*[env.step(action) for env, action in zip(self.envs, actions)])
		dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]  # Env should remain terminated if previously terminated
		self.dones = dones
		observations, rewards, dones = \
			torch.as_tensor(np.array(observations).astype(np.float32)).cuda().float(), \
			torch.tensor(rewards, dtype=torch.float32), \
			torch.tensor(dones, dtype=torch.uint8)
		observations[done_mask] = 0
		rewards[done_mask] = 0
		return observations, rewards, dones, None

	def close(self):
		[env.close() for env in self.envs]