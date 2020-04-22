import argparse
import torch
import torch.optim as optim

import time

import datasets
from network import Network, Baseline, SparseNetwork, BlockNetwork

from typing import Union
from functions import Timer, HiddenPrints
			

	


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


def benchmark(net:Union[Network, BlockNetwork, SparseNetwork], dataset:datasets.Dataset, batch_size:int):
	is_dense = isinstance(net, Network) or isinstance(net, BlockNetwork)
	
	with torch.no_grad():
		# Warm up the network (yes pytorch needs this to happen)
		if is_dense:
			for i in range(5):
				net(dataset.train.dataset[0][0][None, ...].to(net.device).repeat(batch_size, 1, 1, 1))

			# Don't leak into the benchmark time
			try: torch.cuda.synchronize()
			except: pass # If CUDA is not supported, we don't care.
		
		runtime, accuracy = benchmark_dense(net, dataset) if is_dense else benchmark_sparse(net, dataset)
		runtime *= 1000*1000 # s -> us
		print(f'{runtime:6.1f} us / sample, {accuracy:6.2f}% Test Accuracy.')



if __name__ == '__main__':

	batch_size = 64
	dataset = datasets.MNIST(batch_size=batch_size)

	tests = [
		('CUDA Random Weights',       'weights/random_init.pth',            'cuda'),
		('CUDA Pruned Weights',       'weights/pruned_iterative.pth',       'cuda'),
		('CUDA Block Pruned Weights', 'weights/pruned_block_iterative.pth', 'cuda'),

		(' CPU Random Weights',       'weights/random_init.pth',            'cpu'),
		(' CPU Pruned Weights',       'weights/pruned_iterative.pth',       'cpu'),
		(' CPU Block Pruned Weights', 'weights/pruned_block_iterative.pth', 'cpu'),
	]
	
	print()
	print(f'Benchmarking with batch_size={batch_size} and {len(dataset.test.dataset)} samples.')
	print()

	blocknets = []

	for name, weights_path, device in tests:
		with HiddenPrints(stderr=True):
			net = Baseline(dataset.shape, dataset.num_classes)
			net.load_weights(weights_path)
			net.to(device)

		print(f' ---- {name} ----')
		
		print('Vanilla Pytorch: ', end='')
		benchmark(net, dataset, batch_size)
		
		print('  Block Network: ', end='')
		blocknet = net.blockify()
		benchmark(blocknet, dataset, batch_size)

		print('   Scipy Sparse: ', end='')
		if device == 'cpu':
			sparsenet = net.sparsify()
			benchmark(sparsenet, dataset, batch_size)
		else:
			print('                   N/A')

		print()

		# For whatever reason, when python garbage collects a blocknet it crashes.
		# Because I don't want to debug a double free / segfault (in fucking python like wtf),
		# which is likely internal to pytorch, I'm just going to make sure that blocknets never
		# go out of scope and put them in this list. Fight me.
		blocknets.append(blocknet)

