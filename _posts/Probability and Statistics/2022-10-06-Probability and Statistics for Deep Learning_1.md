---
layout: single
title: "Probability and Statistics for Deep Learning[1]"
categories: [Probability and Statistics]
---

###### Keywords: Probability Distribution, Marginal Probability, Conditional Probability, Bayes' Theorem



##### Relationship between Statistics and Machine Learning

Statistics use sample to predict and estimate the population dataset. This concept can be applied to explain the concept of machine learning. Machine learning also finds the mainstream pattern from the limited given data or information. Using the pattern that has been taken by given datasets, it predicts the outcome using unseen data.



##### Random Variables and Probability Distribution

Probability is the terms that describes the degree of the certain event would happen from the all event cases. In case of discrete probability distribution, coin flipping could be an example. 1 is head and 0 is back. Likewise, we assign the real number to the event case in the sample space. This is called **Discrete Random Variable**. 


$$
(1) f(x_i)>0,   i =0,1,2, ... k \\
(2) \sum_{i}^{k} w_i = 1
$$




In other case, the distribution of the heigh in the classroom, the height is continuous value rather than discrete value so It is called **Continuous Random Variable**
$$
(1) f(x) \geq 0 \\
(2) \begin{equation}
\int_{-\infty}^{\infty}f(x)dx = 1
\label{eq:1}
\end{equation}
$$






Probability Distribution

| Random Variable _ x | Probability |
| :-----------------: | :---------: |
|          1          |     0.3     |
|          2          |     0.1     |
|          3          |     0.1     |
|          4          |     0.2     |
|          5          |     0.1     |
|          6          |     0.2     |



Sum of the probability should be 1, and each probability should be greater than 0. P(x=1) = 0.3

In case of two variables happen at the same time, we can denote it as p(X=x, Y=y). ex) p(X=3, Y=5). It is called **joint probability**.



When Dice A is 3 p(X=3) means that the dice X is 3 and the dice Y is the sum of all the probability of the cases that Y =1, Y=2, Y=3, ... Y=6.
$$
p(X=3) = \sum_{y} p(X=3, Y=y) \\
p(X=x) = \sum_{y} p(X=x, Y=y)
$$



Marginalization is the sum of joint probability with all the cases of non-focused probability. 

|              | Y = Head | Y = Back |
| :----------: | :------: | :------: |
| **X = Head** |   1/5    |   2/5    |
| **X = Back** |   1/5    |   1/5    |


$$
\sum_{y} p(X=Head, Y=y)
$$
P(X=Head, Y=Head) + P(X=Head, Y=Back) is the marginalization probability in this case, marginalization probability of X = Head.

|              | Y = Head | Y = Back | P(X) |
| :----------: | :------: | :------: | :--: |
| **X = Head** |   1/5    |   2/5    | 3/5  |
| **X = Back** |   1/5    |   1/5    | 2/5  |
|   **P(Y**)   |   2/5    |   3/5    |      |

Marginal distribution can be specified like above. 

The Joint Probability and Marginal Probability are related to the conditional probability.



Reference:

https://doooob.tistory.com/249
