import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sparse

import functions as f

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
		x = x @ self.weight
		
		if self.batch_size != x.shape[0]:
			self.batch_size = x.shape[0]
			self.bias = make_sparse(self.dense_bias.expand(self.batch_size, -1).contiguous())
		
		x += self.bias
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
