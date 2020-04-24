import torch
import torch.nn as nn
import torch.nn.functional as F

import functions as f
import math

BLOCK_SIZE = 4
EPS = 1e-7



def block_expand(x:torch.Tensor) -> torch.Tensor:
	""" Expands a tensor to be divisible by block size, filling the gap with 0's. """
	shape = list(x.shape)
	new_shape = [int(math.ceil(a / BLOCK_SIZE)) * BLOCK_SIZE for a in shape]
	
	if shape == new_shape:
		return x

	# In case the weight isn't a multiple of BLOCK_SIZE, we need to do some padding
	new_x = torch.zeros(new_shape, dtype=x.dtype, device=x.device)

	# Initialize the reshaped area as 0
	new_x[[slice(0, a) for a in shape]] = x

	return new_x




class BlockLinear():

	def __init__(self, weight:torch.Tensor, bias:torch.Tensor):
		weight    = block_expand(weight.clone())
		self.bias = block_expand(bias.clone())[None, ...]

		block_mask = F.avg_pool2d(weight.abs()[None, None], BLOCK_SIZE, ceil_mode=True)[0, 0] > EPS
		
		num_blocks = block_mask.long().sum().item()
		self.block_weight = torch.zeros((num_blocks, BLOCK_SIZE, BLOCK_SIZE), device=weight.device, dtype=weight.dtype)
		self.gather_idx   = torch.zeros((num_blocks, BLOCK_SIZE), device=weight.device).long()
		self.add_idx      = torch.zeros((num_blocks,), device=weight.device).long()

		# Eww loops? I couldn't find a better way to do this and this is init anyway so it's ok to be slow.
		block_idx = 0
		for block_y in range(block_mask.shape[0]):
			start_y = BLOCK_SIZE * block_y

			for block_x in range(block_mask.shape[1]):
				if not block_mask[block_y, block_x]:
					continue

				start_x = BLOCK_SIZE * block_x

				self.block_weight[block_idx] = weight[start_y:start_y + BLOCK_SIZE, start_x:start_x + BLOCK_SIZE]
				self.gather_idx[block_idx, :] = block_x # Used to map the input blocks to weight blocks
				self.add_idx[block_idx] = block_y

				block_idx += 1
					

		# Use lazy initialization for the batch size
		self.batch_size = -1
		self.out_blocks = block_mask.shape[0]

	def __call__(self, x:torch.Tensor) -> torch.Tensor:
		if self.batch_size != x.shape[0]:
			self._reset_batch_size(x.shape[0])
		
		# Gather the blocks in x to match the prepared weight blocks
		x = x.view(x.shape[0], -1, BLOCK_SIZE)
		x = x.gather(dim=1, index=self.batch_gather_idx)
		x = x.view(-1, BLOCK_SIZE, 1)

		# Do the batched block product
		block_products = torch.bmm(self.batch_block_weight, x)
		block_products = block_products.view(self.batch_size, -1, BLOCK_SIZE)

		# Accumulate the products to get the output vector
		x = torch.zeros((self.batch_size, self.out_blocks, BLOCK_SIZE), dtype=x.dtype, device=x.device)
		x.index_add_(1, self.add_idx, block_products)
		x = x.view(self.batch_size, -1)

		return x + self.bias
	
	def _reset_batch_size(self, batch_size:int):
		self.batch_size = batch_size

		self.batch_block_weight = self.block_weight.repeat(batch_size, 1, 1).contiguous()
		self.batch_gather_idx   = self.gather_idx[None, ...].repeat(batch_size, 1, 1).contiguous()



class BlockReLU():

	def __call__(self, x):
		return F.relu(x, inplace=True)


class BlockNetwork():
	def __init__(self, layers, out_channels:int, device:str):
		self.out_channels = out_channels
		self.device = device

		self.layers = []

		for layer in layers:
			if isinstance(layer, nn.Linear):
				self.layers.append(BlockLinear(layer.weight, layer.bias))
			elif isinstance(layer, nn.ReLU):
				self.layers.append(BlockReLU())
			else:
				print(f'Warning: No block layer implemented for {type(layer)}.')

	def __call__(self, x:torch.Tensor) -> torch.Tensor:
		batch_size = x.shape[0]
		x = x.view(batch_size, -1)
		
		x = block_expand(x)

		for layer in self.layers:
			x = layer(x)

		return x[:batch_size, :self.out_channels].argmax(dim=-1)

	# Dummy function so this can be treated as a Network
	def eval(self):
		pass
