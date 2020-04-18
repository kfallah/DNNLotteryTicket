# DNNLoteryTicket

Investigation on speeding up the forward pass computation for DNN "lottery tickets". Class project for Georgia Tech ECE 6524 Sp'20.

Contributors: Daniel Bolya, Kion Fallah, Gregory Hessler, Shreyas Patil, Cameron Taylor

## Experiment

This experiment is concerned with speeding the forward pass of sparse networks found by the lottery ticket hypothesis [1]. If a neural network with sparse matrices is found through the lottery ticket hypothesis, then the forward pass can be sped up through the use of structured pruning and sparse matrix multiplication.

This script proceeds by training on a DNN with fully-connected layers trained on MNIST digit classification. A pruned network is found using the original library ticket hypothesis and a new structured block pruning method. These methods are tested for forward pass speed.

## Results

TBD

## Running

#### Dependencies

Python 3.0+, PyTorch

#### Scripts

Testing forward pass speed:

`python speed_test.py`

Training baseline network:

`python network.py`

Pruning network:

`python pruning.py`

## References

[1] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neuralnetworks. In ICLR, 2018.
