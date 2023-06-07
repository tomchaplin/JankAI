# JankAI

A large number of neural network layers are built into your favourite ML framework (e.g. PyTorch).
But how do they actually work and how would you implement them if they weren't already builtin?

This repository provides custom implementations of some of these layers.
The purpose of these implementations is to gain understanding of how the layers work and how you can implement your own layers.
These layers are __not__ designed to be fast or production-ready.

Where possible, we provide implementation of both the forward pass and backpropogation (without relying on autograd).
Of course, implementing the back propogation is usually not necessary and potentially worse for performance.

Layers included so far:
* Convolutional layer
* Max pooling
