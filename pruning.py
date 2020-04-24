from network import Network, MaskedNetwork, Baseline
from train import train
from datasets import Dataset, MNIST
import functions as f

from typing import Callable
import torch


TMP_FILE_NAME = 'tmp.pth'
PRUNE_WHITELIST = { torch.nn.Linear, torch.nn.Conv2d }

mask_func_t = Callable[[dict, float], dict]

def lowest_magnitude_mask(info:dict, rate:float) -> dict:
	with torch.no_grad():
		mask = {}

		# This relies on undefined behavior (order of dict entries). Only works in python 3.6 and 3.7
		last_module = list(info['classes'].keys())[-1]

		for name, weight in info['weights'].items():
			module_name = f.module_name(name)

			# Don't prune the bias vectors
			if name.endswith('bias'):
				continue

			# Only prune parameters from the specified modules
			if info['classes'][module_name] not in PRUNE_WHITELIST:
				continue

			# Lottery ticket prunes last layer weights at half the rate
			p = rate / 2 if module_name == last_module else rate

			# We will remove this many elements from this weight
			num_to_prune = int(p * weight.numel())
			
			# Prune the num_prune lowest magnitude weights by masking them out
			prune_indices = weight.view(-1).abs().argsort()[:num_to_prune]

			mask[name] = torch.ones((weight.numel(),), device=weight.device, dtype=torch.float32)
			mask[name][prune_indices] = 0
			mask[name] = mask[name].view(weight.shape)
	
	return mask


def block_mask(info:dict, rate:float, block_size:int=4) -> dict:
	with torch.no_grad():
		mask = {}

		# This relies on undefined behavior (order of dict entries). Only works in python 3.6 and 3.7
		last_module = list(info['classes'].keys())[-1]

		for name, weight in info['weights'].items():
			module_name = f.module_name(name)

			# Don't prune the bias vectors
			if name.endswith('bias'):
				continue

			# Only prune parameters from the specified modules
			if info['classes'][module_name] not in PRUNE_WHITELIST:
				continue

			# Lottery ticket prunes last layer weights at half the rate
			p = rate / 2 if module_name == last_module else rate

			block_mag = torch.nn.functional.avg_pool2d(
				weight.abs()[None, None, ...], block_size, ceil_mode=True)

			# We will remove this many blocks from this weight
			num_to_prune = int(p * block_mag.numel())
			
			# Prune the num_prune lowest magnitude block by masking them out
			prune_indices = block_mag.view(-1).abs().argsort()[:num_to_prune]

			# Create a mask on the blocks
			bmask = torch.ones((block_mag.numel(),), device=block_mag.device, dtype=torch.float32)
			bmask[prune_indices] = 0
			bmask = bmask.view(block_mag.shape)

			# Interpolate the block mask up to the size of the full weight matrix
			wmask = torch.nn.functional.interpolate(bmask, scale_factor=block_size, mode='nearest')[0, 0]

			# In the case that the weight matrix wasn't evenly divisible by the block size,
			# crop the mask down to the correct size
			crop = [slice(0, x, 1) for x in weight.shape]
			wmask = wmask[crop]

			mask[name] = wmask
	
	return mask





def prune_oneshot(data:Dataset, net:Network, prune_rate:float=0.9,
			      mask_func:mask_func_t=lowest_magnitude_mask) -> MaskedNetwork:
	""" Prune a prune_rate fraction of the weights all at once and then retrain. """

	# Temporarily save the initial weights
	initial_weights = net.save_weights(TMP_FILE_NAME)

	# One round of training to decide which weights to prune
	pre_prune_acc = train(net, data)

	# Create pruning mask
	mask = mask_func(net.get_weights(), prune_rate)

	# Reset to the initial weights and delete the tmp file
	net.load_weights(TMP_FILE_NAME)
	f.remove(TMP_FILE_NAME)

	# Create and traine the new masked version
	masked_net = MaskedNetwork(net, mask)
	post_prune_acc = train(masked_net, data)

	print('-- Oneshot Pruning Results --\n')
	print(f'Pruning Rate: {prune_rate}')
	print(f'Pre-prune Accuracy:  {pre_prune_acc: 5.1f}%')
	print(f'Post-prune Accuracy: {post_prune_acc: 5.1f}%')

	return masked_net



def prune_iterative(data:Dataset, net:Network, prune_rate:float=0.9,
	                iterations:int=5, mask_func:mask_func_t=lowest_magnitude_mask) -> MaskedNetwork:
	""" Prune a prune_rate fraction of the weights but do it in iterations steps. """
	iterative_prune_rate = prune_rate / iterations
	accs = []

	# Temporarily save the initial weights
	initial_weights = net.save_weights(TMP_FILE_NAME)

	# One round of training to decide which weights to prune
	accs.append(train(net, data))

	for i in range(iterations):
		mask = mask_func(net.get_weights(), iterative_prune_rate * (i + 1))
		net.load_weights(TMP_FILE_NAME)

		masked_net = MaskedNetwork(net, mask)
		accs.append(train(masked_net, data))

		# Just to make sure that the weights get cleared before creating the mask
		masked_net.eval()
	f.remove(TMP_FILE_NAME)
	
	print('-- Interative Pruning Results --\n')
	print(f'Overall Pruning Rate: {prune_rate}')
	print(f'Per-step Pruning Rate: {iterative_prune_rate:.3f}')
	
	for i in range(len(accs)):
		print(f'Accuracy at {iterative_prune_rate * i * 100: 5.1f}% Pruned: {accs[i]:.2f}%')
	
	return masked_net
		






if __name__ == '__main__':
	dataset = MNIST()

	# net = Baseline(dataset.shape, dataset.num_classes)
	# net.cuda()

	# Vanilla Lottery Ticket Pruning
	# pruned = prune_oneshot(dataset, net)
	# pruned.save_weights('weights/pruned_oneshot.pth')

	# pruned = prune_iterative(dataset, net)
	# pruned.save_weights('weights/pruned_iterative.pth')


	# net = Baseline(dataset.shape, dataset.num_classes)
	# net.cuda()
	
	# Block Pruning
	# pruned = prune_oneshot(dataset, net, mask_func=block_mask)
	# pruned.save_weights('weights/pruned_block_oneshot.pth')

	# pruned = prune_iterative(dataset, net, mask_func=block_mask)
	# pruned.save_weights('weights/pruned_block_iterative.pth')

	for block_size in (2, 8, 16, 32):

		print()
		print(f'PRUNING WITH BLOCK SIZE {block_size}.')
		print()
		net = Baseline(dataset.shape, dataset.num_classes)
		net.cuda()

		pruned = prune_iterative(dataset, net, mask_func=lambda a, b: block_mask(a, b, block_size=block_size))
		pruned.save_weights(f'weights/pruned_block{block_size}_iterative.pth')
