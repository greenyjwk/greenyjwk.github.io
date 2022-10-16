---
Signif. codes:layout: single
title: "Logistic Regression Evaluation using Summary() in R""
categories: [Probability and Statistics]
---



##### Residual Deviance

```python
# we are the
```



anova() Function gives the information about residual deviance. From the output of anova() function, there is a p-value for each predictors. 

|   P-value   |        Deviance        |                         Description                          |
| :---------: | :--------------------: | :----------------------------------------------------------: |
| Low P-value | Low Residual Deviance  | Low P-value : It is a meaningful predictors <br />Low Residual Deviance : The predictor helps reduce deviance |
| Low P-value | High Residual Deviance | High P-value : It is a meaningful predictors <br />High Residual Deviance : The predictor doesn't help reduce deviance |

