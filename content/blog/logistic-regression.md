+++
title = 'Logistic Regression'
date = 2025-02-12T12:25:52-05:00
draft = false
+++


### logistic regression

here the predicted outputs are binary, either a 0 or a 1, therefore the challenge is finding an equation that can translate the input features to either of the choices.

To do this, they use the sigmoid function,viz:

$$
f(x)= \frac{1}{1+e^{-z}}
$$

where z is $\vec{w}.\vec{x} + b$. 

This means if z is high or simply +ve,  $e^{-z}$ is very small and f(x) approaches 1 - thus, it is approximated as 1. And for the erverse, it is approximately 0. This way all the data is transformed. 

when applied to a dataset, this will give a plot like [this](https://drive.google.com/file/d/12yoHcTFsAtLODU9HnIxwkpn_gNhQy1qv/view).


### cost function

to measure the losses for the predictions, the log functions for f(x) is used. however, the predictions do not determine the losses with a single overall equation because for a target of 0, a prediction near 0 has minimal loss and a prediction near 1, has pretty much infinite loss, while it is the reverse for cases with target of 1. therefore there are different loss functions describing this reverse logic, viz:

when the target is 1, loss is $-log(f(x_i)$

and 

when target is 0, loss is $-log (1-(f(x_i)))$

these give, [this](https://drive.google.com/file/d/1vldoP2kRewWZq3pHvBLpyej67cIGE55m/view).



however, the reality is we care about minimizing the losses for both predictions with targets 0 and 1 well.

Thus we need to focus on a loss function combining both versions of the loss function equation. 

To do this we use the prvious idea of a cost function:

$$
J(w,b)= \frac{1}{m} \sum_{i=1}^{m}[\mathrm{losses-for-targets-0-and-losses-for-targets-1}
]
$$

$$
J(w,b)= \frac{1}{m} \sum_{i=1}^{m}[\mathrm{-log(f(x_i)+ -log (1-(f(x_i)))} 
]
$$

but very quickly we see that this doesn’t include the conditional effects of if $y_i$ is 0 or 1, thus, we rewrite  each part as:

$-y_i(log(f(x_i))$ and $((1-yi)-log (1-(f(x_i))))$ respectively, thus if y_i = 0, the first part goes to 0 and the other part is active and when y_i =1 the first part works and the second goes to 0. this is quite simple and elegant now that I think of it. a logical tit for tat. 

thus the cost function $(J)$, becomes:

$$
J(w,b)= \frac{1}{m} \sum_{i=1}^{m}
[-y_i(log(f(x_i))+((1−y_i)-log(1−f(x_i)))]
$$

which we can observe graphically as [such](https://drive.google.com/file/d/1VN0aR00WpdDfihWHqOl5Z0w2XbQHP40f/view).

with this, the idea of minimizing the cost function is more intuitive since it is clearly a sum of all loss functions for that w,b state. 

**notes:**
to converge this with gradient descent we use the same eqautions and principle for gradient descent as univariate and multivariate vectorized linear regression.