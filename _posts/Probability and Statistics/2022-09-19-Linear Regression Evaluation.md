---
layout: single
title: "Linear Regression Evaluation using Summary() in R"
categories: [Probability and Statistics]
---



###### Linear Regression is very strong regression model that can predict the estimator value based on given datasets. From R, there is function  ln() that can create linear regression model and summary() function provides the performance about the linear regression model. From this article, we are going to look through the linear regression model evaluation methodology combing with R summary() function.



```R
library(MASS)
library(ISLR)
names(Boston)

# multiple linear regression
lm.fit = lm(medv~lstat+age, data=Boston)
summary(lm.fit)
```

Output:

```R
# Call:
# lm(formula = medv ~ lstat + age, data = Boston)

# Residuals:
#    Min      1Q  Median      3Q     Max 
# -15.981  -3.978  -1.283   1.968  23.158 

# Coefficients:
#            Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 33.22276    0.73085  45.458  < 2e-16 ***
# lstat       -1.03207    0.04819 -21.416  < 2e-16 ***
# age          0.03454    0.01223   2.826  0.00491 ** 
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 6.173 on 503 degrees of freedom
# Multiple R-squared:  0.5513,	Adjusted R-squared:  0.5495 
# F-statistic:   309 on 2 and 503 DF,  p-value: < 2.2e-16
```



1. ##### Residuals

Residual means that the difference between the actual responses and predicted values. If the residuals are skewed then it means that predictor more falls off one side than the other side.



2. ##### Coefficients



3. ##### P-Value

   temp

   

4. ##### Residual Standart Error

   It means how well the linear regression model fits to the actual datasets. The lower RSE score means that it is better accuracy than the regression model with higher RSE. Also, the high RSE stands for the datasets are spread out from the fitted regression line. 
   $$
   Residual Standard Error = √Σ(y – ŷ)2/df
   $$
   y: Datasets

   ŷ: predicted values by the predictor

   df: Degree of Freedom: Total number of observations - Total number of model parameters -1

​		From the Residual Standard Error output, we can see that the RSE score is 6.173. It says that regression model estimates 'medv' with an average error of 6.173



###### References:

https://feliperego.github.io/blog/2015/10/23/Interpreting-Model-Output-In-R

