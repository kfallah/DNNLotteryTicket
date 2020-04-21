import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sparse

import functions as f


class Network(nn.Module):

	def __init__(self):
		super().__init__()
		self.device = 'cpu'

	def to(self, device):
		super().to(device)
		self.device = device
		return self

	def cuda(self):
		self.to('cuda')
		return self

	def cpu(self):
		self.to('cpu')
		return self

	def get_weights(self) -> dict:
		return {
			'classes': {name: type(module) for name, module in self.named_modules()},
			'weights': self.state_dict(),
		}

	def save_weights(self, path:str):
		f.make_dirs(path)
		torch.save(self.get_weights(), path)

	def load_weights(self, path:str, strict:bool=True):
		if type(path) == str:
			path = torch.load(path)
		self.load_state_dict(path['weights'], strict=strict)

	def apply_mask(self, mask:dict):
		if mask is None:
			return

		with torch.no_grad():
			for k, v in self.state_dict().items():
				if k in mask:
					v *= mask[k]



class MaskedNetwork(Network):

	def __init__(self, net:Network, mask:dict=None):
		super().__init__()

		self.mask = mask
		self.net = net

		self.to(self.net.device)

		self.net.apply_mask(self.mask)

	def forward(self, x):
		if self.training:
			self.net.apply_mask(self.mask)

		return self.net.forward(x)

	def sparse_forward(self, x):
		return self.net.sparse_forward(x)

	def loss(self, preds, y):
		return self.net.loss(preds, y)

	def train(self, training:bool=True):
		self.net.apply_mask(self.mask)

		super().train(training)
		return self

	def eval(self):
		self.net.apply_mask(self.mask)

		super().eval()
		return self

	def save_weights(self, path:str):
		weights = self.net.get_weights()
		weights['mask'] = self.mask

		f.make_dirs(path)
		torch.save(weights, path)



def make_sparse(tensor:torch.Tensor) -> sparse.csr_matrix:
	return sparse.csc_matrix(tensor.detach().cpu().numpy())


class SparseLinear():
	
	def __init__(self, weight:torch.Tensor, bias:torch.Tensor):
		self.weight = make_sparse(weight.t())
		
		# Because scipy sparse addition doesn't broadcast we need to store a batch_sized version
		# So do lazy initialization because we don't know the batch size yet.
		self.dense_bias = bias.cpu()
		self.bias = None
		self.batch_size = -1
	
	def __call__(self, x:sparse.csr_matrix) -> sparse.csr_matrix:
		x @= self.weight
		
		if self.batch_size != x.shape[0]:
			self.batch_size = x.shape[0]
			self.bias = make_sparse(self.dense_bias.expand(self.batch_size, -1).contiguous())
		
		x += self.bias
		x._add_sparse
		return x



class SparseReLU():

	def __call__(self, x:sparse.csr_matrix) -> sparse.csr_matrix:
		x.data *= (x.data > 0)
		x.prune()
		return x


class SparseNetwork():

	def __init__(self, layers, batch_size=64):
		self.layers = []

		for layer in layers:
			if isinstance(layer, nn.Linear):
				self.layers.append(SparseLinear(layer.weight, layer.bias))
			elif isinstance(layer, nn.ReLU):
				self.layers.append(SparseReLU())
			else:
				print(f'Warning: No sparse layer implemented for {type(layer)}.')

	def __call__(self, x:torch.Tensor) -> torch.Tensor:
		x = make_sparse(x.view(x.shape[0], -1))

		for layer in self.layers:
			x = layer(x)

		return torch.from_numpy(x.todense()).argmax(dim=-1)



class Baseline(Network):

	def __init__(self, input_shape:tuple, num_classes:int,
		layer_type=nn.Linear, nonlinearity=nn.ReLU):
		super().__init__()

		self.in_size  = f.prod(input_shape)
		self.out_size = num_classes

		self.layers = nn.Sequential(
			layer_type(self.in_size, 256),
			nonlinearity(),
			layer_type(256, 64),
			nonlinearity(),
			layer_type(64, self.out_size)
		)

	def forward(self, x:torch.Tensor) -> torch.Tensor:
		x = x.view(-1, self.in_size)
		x = self.layers(x)

		if self.training:
			return x
		else:
			return torch.argmax(x, dim=-1)

	def sparsify(self) -> SparseNetwork:
		return SparseNetwork(self.layers)

	def loss(self, preds, y):
		return F.cross_entropy(preds, y)
