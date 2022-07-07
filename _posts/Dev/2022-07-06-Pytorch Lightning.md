---
layout: single
title:  "Pytorch Lightning"
categories: [Dev]
---

Pytorch Lightning is the open source framework that provides. It enables the efficient cross over among GPU, CPU or TPU calculation, and organizes training, validation, test etc modules more concisely which is called efficient abstraction. It helps to implement neural network code less lines of codes than Pytorch.

------

### Lightning Module

A LightningModule organizes into 6 sections

- init: computation
- training_step: Train Loop
- validation_step: Validation Loop
- test_step: Test Loo
- predict_step: Prediction Loop
- configure_optimizers: Optimizers and LR Schedulers

------

#### training_step

training_step may not be directly used, but it is used inside Trainer class provided by Pytorch Lightning, and it trained the model.

------

#### validation_step

validation_step may not be directly used, but it is used inside Trainer class provided by Pytorch Lightning, and it gives loss results from the model.

------

#### configure_optimizers

It defines the optimizers and learning rate schedulers.

/* what is learning rate scheduler? Learning rate schedulers finds the optimal learning rate by reducing the learning rate as the training progressed .



/* what is warm-up steps ? Using very low learning rate for the initial steps. Then the regular learning rate or learning rate scheduler used to increase the learning rate. It prevents from using drastic or big learning rate in initial steps to make easy to tune the neural network.

------

### SSLOnlineEvaluator

Appends a MLP for fine-tuning to the given model. Callback has its own mini-inner loop. 

------

### Multilayer Perceptron

A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique. Since there are multiple layers of neurons, MLP is a deep learning technique.

MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.

https://pytorch-lightning-bolts.readthedocs.io/en/latest/self_supervised_callbacks.html

------

### callback

The callback function in deep learning is the function that is performed during training is processed. The optimal parameters are stored or the training is terminated when the validation performance is not increased any more.

https://towardsdatascience.com/callbacks-in-neural-networks-b0b006df7626

