import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
import torchvision.datasets as datasets
import kornia
import utils
import os
from kornia.color.hsv import hsv_to_rgb, rgb_to_hsv



places_dataloader = None
places_iter = None


def _load_places(batch_size=256, image_size=84, num_workers=16, use_val=False):
	global places_dataloader, places_iter
	partition = 'val' if use_val else 'train'
	print(f'Loading {partition} partition of places365_standard...')
	for data_dir in utils.load_config('datasets'):
		if os.path.exists(data_dir):
			fp = os.path.join(data_dir, 'places365_standard', partition)
			if not os.path.exists(fp):
				print(f'Warning: path {fp} does not exist, falling back to {data_dir}')
				fp = data_dir
			places_dataloader = torch.utils.data.DataLoader(
				datasets.ImageFolder(fp, TF.Compose([
					TF.RandomResizedCrop(image_size),
					TF.RandomHorizontalFlip(),
					TF.ToTensor()
				])),
				batch_size=batch_size, shuffle=True,
				num_workers=num_workers, pin_memory=True)
			places_iter = iter(places_dataloader)
			break
	if places_iter is None:
		raise FileNotFoundError('failed to find places365 data at any of the specified paths')
	print('Loaded dataset from', data_dir)


def _get_places_batch(batch_size):
	global places_iter
	try:
		imgs, _ = next(places_iter)
		if imgs.size(0) < batch_size:
			places_iter = iter(places_dataloader)
			imgs, _ = next(places_iter)
	except StopIteration:
		places_iter = iter(places_dataloader)
		imgs, _ = next(places_iter)
	return imgs.cuda()


def random_overlay(x, dataset='places365_standard'):
	"""Randomly overlay an image from Places"""
	global places_iter
	alpha = 0.5

	if dataset == 'places365_standard':
		if places_dataloader is None:
			_load_places(batch_size=x.size(0), image_size=x.size(-1))
		imgs = _get_places_batch(batch_size=x.size(0)).repeat(1, x.size(1)//3, 1, 1)
	else:
		raise NotImplementedError(f'overlay has not been implemented for dataset "{dataset}"')

	return ((1-alpha)*(x/255.) + (alpha)*imgs)*255.


def random_conv_(x):
	"""Applies a random conv2d, deviates slightly from https://arxiv.org/abs/1910.05396"""
	# print("TEST")
	n, c, h, w = x.shape
	for i in range(n):
		weights = torch.randn(3, 3, 3, 3).to(x.device)
		temp_x = x[i:i+1].reshape(-1, 3, h, w)/255.
		temp_x = F.pad(temp_x, pad=[1]*4, mode='replicate')
		out = torch.sigmoid(F.conv2d(temp_x, weights))*255.
		total_out = out if i == 0 else torch.cat([total_out, out], axis=0)
	return total_out.reshape(n, c, h, w)


def batch_from_obs(obs, batch_size=32):
	"""Copy a single observation along the batch dimension"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def prepare_pad_batch(obs, next_obs, action, batch_size=32):
	"""Prepare batch for self-supervised policy adaptation at test-time"""
	batch_obs = batch_from_obs(torch.from_numpy(obs).cuda(), batch_size)
	batch_next_obs = batch_from_obs(torch.from_numpy(next_obs).cuda(), batch_size)
	batch_action = torch.from_numpy(action).cuda().unsqueeze(0).repeat(batch_size, 1)

	return random_crop_cuda(batch_obs), random_crop_cuda(batch_next_obs), batch_action


def identity(x):
	return x


def random_shift(imgs, pad=4):
	"""Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
	_,_,h,w = imgs.shape
	imgs = F.pad(imgs, (pad, pad, pad, pad), mode='replicate')
	return kornia.augmentation.RandomCrop((h, w))(imgs)

def random_crop(imgs, size=84):
	"""Vectorized random shift, imgs: (B,C,H,W), pad: #pixels"""
	return kornia.augmentation.RandomCrop((size, size))(imgs)

# def random_crop(x, size=84, w1=None, h1=None, return_w1_h1=False):
# 	"""Vectorized CUDA implementation of random crop, imgs: (B,C,H,W), size: output size"""
# 	assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
# 		'must either specify both w1 and h1 or neither of them'
# 	assert isinstance(x, torch.Tensor) and x.is_cuda, \
# 		'input must be CUDA tensor'
	
# 	n = x.shape[0]
# 	img_size = x.shape[-1]
# 	crop_max = img_size - size

# 	if crop_max <= 0:
# 		if return_w1_h1:
# 			return x, None, None
# 		return x

# 	x = x.permute(0, 2, 3, 1)

# 	if w1 is None:
# 		w1 = torch.LongTensor(n).random_(0, crop_max)
# 		h1 = torch.LongTensor(n).random_(0, crop_max)

# 	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
# 	cropped = windows[torch.arange(n), w1, h1]

# 	if return_w1_h1:
# 		return cropped, w1, h1

# 	return cropped

def random_conv(x):
	# print("TEST2")
	_device = x.device
	x = x / 255.
	img_h, img_w = x.shape[2], x.shape[3]
	num_stack_channel = x.shape[1]
	num_batch = x.shape[0]
	num_trans = num_batch
	batch_size = int(num_batch / num_trans)
	
	# initialize random covolution
	with torch.no_grad():
		rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
		
		for trans_index in range(num_trans):
			torch.nn.init.xavier_normal_(rand_conv.weight.data)
			temp_x = x[trans_index*batch_size:(trans_index+1)*batch_size]
			temp_x = temp_x.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
			rand_out = rand_conv(temp_x)
			if trans_index == 0:
				total_out = rand_out
			else:
				total_out = torch.cat((total_out, rand_out), 0)
		total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
	total_out = total_out * 255.
	return total_out

def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)

def grayscale(imgs):
    # imgs: b x c x h x w
    device = imgs.device
    b, c, h, w = imgs.shape
    frames = c // 3
    
    imgs = imgs.view([b,frames,3,h,w])
    imgs = imgs[:, :, 0, ...] * 0.2989 + imgs[:, :, 1, ...] * 0.587 + imgs[:, :, 2, ...] * 0.114 
    
    imgs = imgs.type(torch.uint8).float()
    # assert len(imgs.shape) == 3, imgs.shape
    imgs = imgs[:, :, None, :, :]
    imgs = imgs * torch.ones([1, 1, 3, 1, 1], dtype=imgs.dtype).float().to(device) # broadcast tiling
    return imgs
	

def gray_scale(images, p=.3):
	device = images.device
	in_type = images.type()
	images = images
	images = images.type(torch.uint8)
	# images: [B, C, H, W]
	bs, channels, h, w = images.shape
	images = images.to(device)
	gray_images = grayscale(images)
	rnd = np.random.uniform(0., 1., size=(images.shape[0],))
	mask = rnd <= p
	mask = torch.from_numpy(mask)
	frames = images.shape[1] // 3
	images = images.view(*gray_images.shape)
	mask = mask[:, None] * torch.ones([1, frames]).type(mask.dtype)
	mask = mask.type(images.dtype).to(device)
	mask = mask[:, :, None, None, None]
	out = mask * gray_images + (1 - mask) * images
	out = out.view([bs, -1, h, w]).type(in_type)
	return out


def cut_out(imgs, min_cut=8, max_cut=24):
	device = imgs.device
	n, c, h, w = imgs.shape
	imgs = imgs.cpu().numpy() / 255.
	w1 = np.random.randint(min_cut, max_cut, n)
	h1 = np.random.randint(min_cut, max_cut, n)
	
	cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
	for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
		cut_img = img.copy()
		cut_img[:, h11:h11 + h11, w11:w11 + w11] = 0
		cutouts[i] = cut_img
	cutouts = torch.as_tensor(cutouts, device=device) * 255.
	return cutouts

def cutout_color( imgs, box_min=8, box_max=24):
	device = imgs.device
	imgs = imgs.cpu().numpy() / 255.
	n, c, h, w = imgs.shape
	w1 = np.random.randint(box_min, box_max, n)
	h1 = np.random.randint(box_min, box_max, n)
	
	cutouts = np.empty((n, c, h, w), dtype=imgs.dtype)
	rand_box = np.random.randint(0, 255, size=(n, c)) / 255.
	for i, (img, w11, h11) in enumerate(zip(imgs, w1, h1)):
		cut_img = img.copy()
		# add random box
		cut_img[:, h11:h11 + h11, w11:w11 + w11] = np.tile(
			rand_box[i].reshape(-1,1,1),                                                
			(1,) + cut_img[:, h11:h11 + h11, w11:w11 + w11].shape[1:])     
		cutouts[i] = cut_img
	cutouts = cutouts * 255.
	cutouts = torch.as_tensor(cutouts, device=device)
	return cutouts

def color_jitter( x, 
			brightness=(0.6, 1.4),
			contrast=(0.6, 1.4),
			saturation=(0.6, 1.4),
			hue=(-0.5, 0.5)):
	""" Returns jittered images.

	Args:
		x (torch.Tensor): observation tensor.

	Returns:
		torch.Tensor: processed observation tensor.

	"""
	# check if channel can be devided by three
	if x.shape[1] % 3 > 0:
		raise ValueError('color jitter is used with stacked RGB images')

	# flag for transformation order
	is_transforming_rgb_first = np.random.randint(2)
	x = x / 255.
	# (batch, C, W, H) -> (batch, stack, 3, W, H)
	flat_rgb = x.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

	if is_transforming_rgb_first:
		# transform contrast
		flag_rgb = _transform_contrast(flat_rgb, contrast)

	# (batch, stack, 3, W, H) -> (batch * stack, 3, W, H)
	rgb_images = flat_rgb.view(-1, 3, x.shape[2], x.shape[3])

	# RGB -> HSV
	hsv_images = rgb_to_hsv(rgb_images)

	# apply same transformation within the stacked images
	# (batch * stack, 3, W, H) -> (batch, stack, 3, W, H)
	flat_hsv = hsv_images.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

	# transform hue
	flat_hsv = _transform_hue(flat_hsv, hue)
	# transform saturate
	flat_hsv = _transform_saturate(flat_hsv, saturation)
	# transform brightness
	flat_hsv = _transform_brightness(flat_hsv, brightness)

	# (batch, stack, 3, W, H) -> (batch * stack, 3, W, H)
	hsv_images = flat_hsv.view(-1, 3, x.shape[2], x.shape[3])

	# HSV -> RGB
	rgb_images = hsv_to_rgb(hsv_images)

	# (batch * stack, 3, W, H) -> (batch, stack, 3, W, H)
	flat_rgb = rgb_images.view(x.shape[0], -1, 3, x.shape[2], x.shape[3])

	if not is_transforming_rgb_first:
		# transform contrast
		flat_rgb = _transform_contrast(flat_rgb, contrast)
	flat_rgb = flat_rgb * 255.
	return flat_rgb.view(*x.shape)

def _transform_hue( hsv, hue):
	scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
	scale = scale.uniform_(*hue) * 255.0 / 360.0
	hsv[:, :, 0, :, :] = (hsv[:, :, 0, :, :] + scale) % 1
	return hsv

def _transform_saturate( hsv, saturation):
	scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
	scale.uniform_(*saturation)
	hsv[:, :, 1, :, :] *= scale
	return hsv.clamp(0, 1)

def _transform_brightness( hsv, brightness):
	scale = torch.empty(hsv.shape[0], 1, 1, 1, device=hsv.device)
	scale.uniform_(*brightness)
	hsv[:, :, 2, :, :] *= scale
	return hsv.clamp(0, 1)

def _transform_contrast( rgb, contrast):
	scale = torch.empty(rgb.shape[0], 1, 1, 1, 1, device=rgb.device)
	scale.uniform_(*contrast)
	means = rgb.mean(dim=(3, 4), keepdims=True)
	return ((rgb - means) * (scale + means)).clamp(0, 1)
