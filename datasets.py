from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Dataset:

	def __init__(self, name:str, train, test, input_shape:tuple, num_classes:int):
		self.train = train
		self.test  = test

		self.shape = input_shape
		self.num_classes = num_classes

		self.name = name


def MNIST(batch_size=64):
	dataset_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,))
	])

	dataset    = datasets.MNIST('./data', train=True, download=True, transform=dataset_transform)
	dataloader = DataLoader(dataset, shuffle=True, pin_memory=True, batch_size=batch_size)

	test_dataset    = datasets.MNIST('./data', train=False, download=True, transform=dataset_transform)
	test_dataloader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=batch_size)

	return Dataset('MNIST', dataloader, test_dataloader, (1, 28, 28), 10)
