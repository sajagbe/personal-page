+++
title = 'Z Score Normalization'
date = 2025-02-05T10:06:06-05:00
draft = false
tags = ["machine learning","andrew ng", "normalization"]
+++


### z-score normalization:

This is a way to make all features in the dataset have the same unit (the $\sigma$).

$$
Z = \frac{X - \mu}{\sigma}
$$

steps are:

1. get mean of feature from all examples ($\mu$)
2. get deviation from mean for each item
3. square values from (2) and sum them all
4. Take average of (3), i.e. (3) / m where m is no. of examples 
5. Get $\sqrt{(4)}$ ($\sigma$)

When you do this for each sample feature in a feature set, the range will hence be from +b$\sigma$ (gotten from the z norm for number highest from mean) to -b$\sigma$ (gotten from the z norm for number lowest from mean) with 0 in the middle (i.e. the mean), where b is a scalar. so cool to understand why they are called "scalar", because they literally scale what they multiply. 

e.g. if a sample feature 2 has a value 2$\sigma$, it means it is 2 $\sigma$ away from mean for that feature in that dataset. 

because the software does not truly care about the actual value of $\sigma$, but just about the distance from the sigma that represents the value of the sample feature, all features will have the same unit and thus be comparable. 

this will make the gradient descent much faster because the algo. will be able to make much even changes to all weights at the same time since they are now on the same scale.