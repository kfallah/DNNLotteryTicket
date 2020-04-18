import argparse
import torch
import torch.optim as optim

import time

import datasets
from network import Network, Baseline


def test_speed(net:Network, data:datasets.Dataset):
    pre_time = time.time()
	for x, _ in data.train:
		x = x.to(net.device)
        y_hat = net.sparse_forward(x)

	return (time.time()) - pre_time) / data.train.__length__()


if __name__ == '__main__':
    batch_size  = 64
	dataset = datasets.MNIST(batch_size=batch_size)

	net = Baseline(dataset.shape, dataset.num_classes)
	net.to('cuda')

	try:
		run_time = test_speed(net, dataset)
        print("Average runtime over {} samples with batch size {}: {:.3f} sec".format(
        dataset.train.__length__(), batch_size, run_time))

	except KeyboardInterrupt:
		print('Stopping...')
