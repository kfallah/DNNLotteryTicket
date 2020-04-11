import torch
import torch.nn.functional as F

from PIL import Image
import numpy as np

from collections import deque
import math
import datetime
import random

import os, shutil

def load_image(path:str, device:str='cpu') -> torch.Tensor:
	""" Loads an image and returns it in float [0, 1] format with shape [1, c, h, w]. """
	# You may not like it, but this is what peak pytorch looks like.
	# Well, peak PIL -> numpy -> pytorch, anyway.
	return torch.from_numpy(np.array(Image.open(path))).to(device).permute(2, 0, 1)[None, :3, ...].contiguous().float() / 255


def interpolate(img:torch.Tensor, size:tuple) -> torch.Tensor:
	""" Size should be (h, w). """
	return F.interpolate(img, size=size, mode='bilinear', align_corners=False)


def normalize_img(img:torch.Tensor) -> torch.Tensor:
	""" Remaps a [0, 1] image to [-1, 1]. """
	return img * 2 - 1

def unnormalize_img(img:torch.Tensor) -> torch.Tensor:
	""" Remaps a [-1, 1] image to [0, 1]. """
	return torch.clamp((img + 1) * 0.5, min=0, max=1)


def seconds_to_string(sec:float):
	return str(datetime.timedelta(seconds=sec)).split('.')[0]


def prod(l:list, base=1):
	for x in l:
		base *= x
	return base

def make_perm(in_order:str, out_order:str) -> list:
	"""
	Returns the list necessary to permute one channel order to another.
	
	For instance:
		make_perm('rgb', 'bgr') -> [2, 1, 0]
		make_perm('chw', 'hwc') -> [2, 0, 1]
	"""
	char_map = {c: idx for idx, c in enumerate(in_order)}
	return [char_map[c] for c in out_order]


# That's not a function! >:(
class MovingAverage():
	""" Keeps an average window of the specified number of items. """

	def __init__(self, max_window_size=1000):
		self.max_window_size = max_window_size
		self.reset()

	def add(self, elem):
		""" Adds an element to the window, removing the earliest element if necessary. """
		if not math.isfinite(elem):
			print('Warning: Moving average ignored a value of %f' % elem)
			return
		
		self.window.append(elem)
		self.sum += elem

		if len(self.window) > self.max_window_size:
			self.sum -= self.window.popleft()
	
	def append(self, elem):
		""" Same as add just more pythonic. """
		self.add(elem)

	def reset(self):
		""" Resets the MovingAverage to its initial state. """
		self.window = deque()
		self.sum = 0

	def get_avg(self):
		""" Returns the average of the elements in the window. """
		return self.sum / max(len(self.window), 1)

	def __str__(self):
		return str(self.get_avg())
	
	def __repr__(self):
		return repr(self.get_avg())
	
	def __len__(self):
		return len(self.window)


def itemize_and_combine(total:dict, losses:dict):
	""" Assuming total is a dictionary of moving averages, combines the losses into the total. """
	for k, v in losses.items():
		v = v.item()

		if k not in total:
			total[k] = MovingAverage()
		total[k].add(v)


def module_name(parameter_name:str) -> str:
	""" Returns the name of a module given the name of one of its parameters. """
	last_period_idx = parameter_name.rfind('.')
	return parameter_name if last_period_idx == -1 else parameter_name[:last_period_idx]



def make_dirs(path:str):
	""" Why is this not how the standard library works? """
	path = os.path.split(path)[0]
	if path != '':
		os.makedirs(path, exist_ok=True)

def remove(path:str):
	os.remove(path)


def seed_all(seed:int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
