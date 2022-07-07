---
layout: single
title:  "Pytorch Lightning"
categories: [Dev]
---

Pytorch Lightning is the open source framework that provides. It enables the efficient cross over among GPU, CPU or TPU calculation, and organizes training, validation, test etc modules more concisely which is called efficient abstraction. It helps to implement neural network code less lines of codes than Pytorch.



### <br>Lightning Module

A LightningModule organizes into 6 sections

- init: computation
- training_step: Train Loop
- validation_step: Validation Loop
- test_step: Test Loo
- predict_step: Prediction Loop
- configure_optimizers: Optimizers and LR Schedulers



#### <br>training_step

training_step may not be directly used, but it is used inside Trainer class provided by Pytorch Lightning, and it trained the model.

#### <br>validation_step

validation_step may not be directly used, but it is used inside Trainer class provided by Pytorch Lightning, and it gives loss results from the model.

#### <br>configure_optimizers

It defines the optimizers and learning rate schedulers.

/* what is learning rate scheduler? Learning rate schedulers finds the optimal learning rate by reducing the learning rate as the training progressed .



/* what is warm-up steps ? Using very low learning rate for the initial steps. Then the regular learning rate or learning rate scheduler used to increase the learning rate. It prevents from using drastic or big learning rate in initial steps to make easy to tune the neural network.

