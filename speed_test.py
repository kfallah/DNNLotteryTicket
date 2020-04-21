import argparse
import torch
import torch.optim as optim

import time

import datasets
from network import Network, Baseline, SparseNetwork

from typing import Union


class Timer():
	"""
	Times the enviroment you give it and adds to the total time.

	Sample Usage:
		timer = Timer()

		with timer.time():
			# Write things to time here
		
		timer.total # Contains the seconds elapsed above.
	"""

	def __init__(self):
		self.reset()
	
	def reset(self):
		self.total = 0

	def time(self):
		return Timer._TimerEnv(self)

	class _TimerEnv:
		def __init__(self, timer):
			self.timer = timer

		def __enter__(self):
			self.start_time = time.time()
		
		def __exit__(self, type, value, traceback):
			self.timer.total += time.time() - self.start_time
			

	


def benchmark_sparse(net:SparseNetwork, data:datasets.Dataset):
	""" Note: this will only benchmark on the CPU. """
	timer = Timer()

	num_correct = 0
	num_total   = 0

	for x, y in data.test:
		# Time the actual runtime
		with timer.time():
			y_hat = net(x)
		
		# Untimed, verify that the prediction is accurate
		num_correct += torch.sum(y == y_hat).item()
		num_total   += int(x.size(0))
	
	time_taken = timer.total / len(data.test.dataset)
	acc = num_correct / num_total * 100

	return time_taken, acc


def benchmark_dense(net:Network, data:datasets.Dataset):
	""" Note: this will benchmark using whatever device you pass in net as. """
	net.eval()
	timer = Timer()

	num_correct = 0
	num_total   = 0

	for x, y in data.test:
		y = y.to(net.device)

		# Time the actual runtime
		with timer.time():
			# Note that transfering x to the device is necessary, so it's timed
			x = x.to(net.device)
			y_hat = net(x)

			# Make sure we're timing the asynchronous CUDA calls here
			try: torch.cuda.synchronize()
			except: pass # If CUDA is not supported, we don't care.
		
		# Untimed, verify that the prediction is accurate
		num_correct += torch.sum(y == y_hat).item()
		num_total   += int(x.size(0))
	
	time_taken = timer.total / len(data.test.dataset)
	acc = num_correct / num_total * 100

	return time_taken, acc


def benchmark(net:Union[Network, SparseNetwork], dataset:datasets.Dataset, batch_size:int):
	is_dense = isinstance(net, Network)
	num_samples = len(dataset.test.dataset)
	
	with torch.no_grad():
		# Warm up the network (yes pytorch needs this to happen)
		if is_dense:
			for i in range(5):
				net(dataset.train.dataset[0][0][None, ...].to(net.device))

		runtime, accuracy = benchmark_dense(net, dataset) if is_dense else benchmark_sparse(net, dataset)
		runtime *= 1000*1000 # s -> us
		print()
		print(f'Average fps over {num_samples} samples with batch size {batch_size}: {runtime:.1f} us / sample.')
		print(f'Test accuracy: {accuracy:6.2f}%')



if __name__ == '__main__':

	batch_size = 64
	dataset = datasets.MNIST(batch_size=batch_size)

	net = Baseline(dataset.shape, dataset.num_classes)
	net.load_weights('weights/pruned_block_iterative.pth')
	net.cpu()

	# Test the pytorch network, change params in the block above ^^^
	benchmark(net, dataset, batch_size)

	# Do the sparse stuff
	sparsenet = net.sparsify()
	benchmark(sparsenet, dataset, batch_size)

