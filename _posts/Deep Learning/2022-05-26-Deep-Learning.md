---
layout: single
title:  "Deep Learning"
categories: [Deep Learning]
---

##### Loss Function

Loss function is the function that specifies the difference between probability distribution of prediction and label. 

------

##### Cross Entropy

Entropy means that that it is hard to predict what prediction value would be gotten. For example, dice has more entropy than coin, since dice has six cases to get in each trial.

It uses true probability distribution(labeling) and predicted probability distribution, and it gets the total loss of the difference between prediction and labeling. Thus, if the cross entropy loss is big, then it means that the model got wrong answer for the label.

------

##### KL divergence



------

##### What is Message Passing Interface(MPI)

MPI is a communication protocol for parallel programming. MPI enables applications to operate in parallel in distributed systems.

------

##### Margin-based Losses

It is particularly useful for binary classification. In contrast to the distance-based losses. It doesn't care about difference between prediction and target. It penalizes prediction based on how well they agree with sign on the target.

------

##### what is node in deep learning?

A node in deep learning is a computational unit that has one or multiple inputs and outputs.

### <br>Off-line vs On-line Learning

#### <br>on-line training

<br>Once the modes is trained, then model is not updated as long as it is not fully retrained with datasets.

#### <br>off-line training

<br>model is being trained depending on the training datasets.



### <br>Contrastive Learning

In order to mitigate the lack of datasets issues, unlabeled different images are used for model to have capability to distinguish the images.



### <br>The correct number of epoch

There is no certain specific number of epochs. The more important is validation and training error. If the two metrics keep dropping as training goes on, the training should be continued. So the number of epochs should be decided depending on the training and validation error tendency.



### Cross Entropy

From deep learning, cross entropy loss is used to have difference between probability distribution between predicted outcomes and real labels.

- Real label is expressed as one-hot-encoding such as

Class A : 0 / Class B : 1 / Class C : 0



- Machine learning outcome is 

Class A : 0.228 / Class B : 0.619 / Class C : 0.153



The Cross Entropy to get the difference between label and predicted value is:

![Screen Shot 2022-06-01 at 3.08.27 PM](../images/2022-05-26-Deep Learning/Screen Shot 2022-06-01 at 3.08.27 PM.png)



H = - ( 0.0 * ln(0.228) + 1.0 * ln(0.619) + 0.0 * ln(0.153) )

The model is being trained to lower the loss between predicted value and real label.

Each two tiles which are green boxes has split parts of single nuclei



From pytorch

```python
outputs = outputs.reshape(batch_size * width_out * height_out, 2)
labels = labels.reshape(batch_size * width_out * height_out)
loss = criterion(outputs, labels)
```





### <br>Entropy

Minimum amount of information to express information.



 
