[![Build Status](https://travis-ci.com/bsakai2000/SimpleNN.svg?branch=master)](https://travis-ci.com/bsakai2000/SimpleNN)
# SimpleNN

A simple Neural Network implementation in C++

This repository contains a /src folder for files pertaining to the Network class, which can be used to build and train neural networks, and a /test folder for testing both the full Network and its component functions. To run tests, call `make runtest`.

Network objects are initialized with the number of input nodes, the number of hidden layers, the number of nodes per hidden layer, and the number of output nodes. To get a result from a trained Network, call `Network.forward_propagate` with an input array and it will return an output array. **TODO** To train a Network object, call `Network.train` with an array of input arrays, an array of expected output arrays (one per input, in the same order as the inputs), and the number of input arrays.
