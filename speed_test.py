import argparse
import torch
import torch.optim as optim

import datasets
from network import Network, Baseline


def test_speed(net:Network, data:datasets.Dataset):
	for (x, y) in data.train:
		x, y = x.to(net.device), y.to(net.device)
		theta = net(x)

		optimizer.zero_grad()
		loss = net.loss(theta, y)

		loss.backward()
		optimizer.step()

		if iteration % 100 == 0:
			print(f'[{epoch:3}] {iteration:6} || Loss: {loss.item():.3f}')

		iteration += 1

	print()
	acc = test(net, data.test)
	print(f'Test  Accuracy: {acc:.2f}%')
	print()

	net.train()

	return acc


if __name__ == '__main__':
	dataset = datasets.MNIST()

	net = Baseline(dataset.shape, dataset.num_classes)
	net.to('cuda')

	try:
		train(net, dataset)
	except KeyboardInterrupt:
		print('Stopping...')

	net.save_weights('weights/final_weights.pth')
