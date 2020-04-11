import torch
import torch.optim as optim

import datasets
from network import Network, Baseline




def train(net:Network, data:datasets.Dataset, num_epochs:int=5):
	net.train()

	optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)

	iteration = 0

	for epoch in range(num_epochs):
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
		acc = test(net, data.train)
		print(f'Test  Accuracy: {acc:.2f}%')
		print()

		net.train()
	
	return acc

def test(net, dataloader):
	net.eval()
	
	num_correct = 0
	num_total   = 0

	for (x, y) in dataloader:
		x, y = x.to(net.device), y.to(net.device)
		theta = net(x)

		num_correct += torch.sum(y == theta).item()
		num_total   += int(x.size(0))

	return num_correct / num_total * 100


if __name__ == '__main__':
	dataset = datasets.MNIST()

	net = Baseline(dataset.shape, dataset.num_classes)
	net.to('cuda')

	try:
		train(net, dataset)
	except KeyboardInterrupt:
		print('Stopping...')
	
	net.save_weights('weights/final_weights.pth')
