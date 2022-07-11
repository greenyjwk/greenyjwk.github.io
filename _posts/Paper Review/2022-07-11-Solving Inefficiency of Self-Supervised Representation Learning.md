---
layout: single
title: "Solving Inefficiency of Self-Supervised Representation Learning"
categories: [Research Paper Review]
---

##### Introduction

SimCLR which is one of the self-supervised learning mouthed has issue in inefficiency to train. The paper states that the two main issues in training the self supervised learning which one is under-clustering and the other is over-clustering. Under clustering means that the model can not be trained to discriminate the objects when the negative sample pairs are insufficient, in contrast over-clustering implies that forces model to classify the objects which is the same class in different clusters.

<Figure for under-clustering and over-clustering>



![Screen Shot 2022-07-11 at 4.44.33 PM](../../images/2022-07-11-Truncated Triplet Loss/Screen Shot 2022-07-11 at 4.44.33 PM-7572593.png)



------

##### Contributions

- Analyzing the existing attribute that results in under-clustering and over-clustering.
- Truncated-Triplet Loss Function is proposed to address under-clustering and over-clustering issues. The loss function is guaranteed by Bernoulli distribution model.
- Novel SSL framework with truncated-Triplet Loss function improves learning efficiency and state-of-the-art performance in several large-scale benchmarks.



