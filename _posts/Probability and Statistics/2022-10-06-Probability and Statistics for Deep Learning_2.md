---
layout: single
title: "Probability and Statistics for Deep Learning[2]"
categories: [Probability and Statistics]
---

###### Keywords: Conditional Probability, Bayes' Theorem



Conditional Probability is the probability that is conditioned in certain circumstance. The table below is the table that it is the distribution of raining when the person brought an umbrella.

| Umbrella (X) | Rain (Y) |
| :----------: | :------: |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      0       |    1     |
|      0       |    1     |
|      0       |    1     |
|      1       |    0     |
|      0       |    0     |
|      0       |    0     |
|      0       |    0     |
|      0       |    0     |
|      0       |    0     |
|      0       |    0     |



We can think of the distribution of the case of raining when(conditioned) a person brought an umbrella.

| Umbrella (X) | Rain (Y) |
| :----------: | :------: |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    1     |
|      1       |    0     |



In that case the conditional probability that the person brought an umbrella when it didn't rain.
$$
p(Y = 0|X = 1) = \frac{1}{7}
$$



Likewise, the conditional probability that the person brought an umbrella when it rained is 
$$
p(Y = 1|X = 1) = \frac{6}{7}
$$



And through the table above, we can check that (2) is different from the joint probability which is below (3) 
$$
p(Y=0, X=1) = \frac{1}{16}
$$


The joint probability distribution table can be specified like below.

|           | Y = 0 | Y = 1 |
| :-------: | :---: | :---: |
| **X = 0** | 6/16  | 3/16  |
| **X = 1** | 1/16  | 6/16  |

​	



The person brought an umbrella and it rained                   : 6/16

It rained on condition of the person brought an umbrella  : 6/7

They are different. Joint probability considers all the cases of

- The person didn't bring umbrella, it rained.
-  The person didn't bring umbrella, it didn't rain.
-  The person brought umbrella, it didn't rain.



Otherwise, conditional probability only considers the case of raining. Thus, two probability have different denominator in the probability since they have different case pools.





From the conditional probability, the sum of the probability of certain condition is 1.
$$
p(Y=0|X=1)+p(Y=1|X=1)=1
$$


However, the below doesn't guarantee 1 as sum of probability.
$$
p(Y=0|X=1)+p(Y=0|X=0)
$$




According to the conditional probability formula, it can be transformed like below, and joint probability and marginalization can be applied to calculate the conditional probability.
$$
p(Y=0|X=1) = \frac{p(X=1,Y=0)}{p(X=1)} \\
=(\frac{joint \ probability}{marginalization})\\
=\frac{1/16}{7/16} \\
=\frac{1}{7}
$$


From the formula above, we can conclude it. Interpretation of the formula is that we can calculate joint probability by calculating the marginalization from the conditional probability.
$$
p(x,y) = p(y|x)p(x)
$$


###### Reference:

https://doooob.tistory.com/249
