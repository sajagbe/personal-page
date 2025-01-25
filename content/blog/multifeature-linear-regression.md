+++
title = 'Multifeature Linear Regression'
date = 2025-01-25T17:32:53-05:00
draft = false
tags = ["machine learning","andrew ng", "linear regression","multifeature linear regression","katex"]
+++
## **the math:**

Logically same as univariate [here](https://soajagbe.xyz/blog/week2-2025/),  just instead of single $w$ and $x$ there are multiple. It isalso more efficient to express all weights and features as vectors  $\vec{w}$ and $\vec{x}$. This way their dot product ($\vec{w}$ $\cdot$ $\vec{x}$)is efficiently handled. 

thus the equation becomes: 

$$
f_{\vec{w},b}(\vec{x}^{(i)}) = \vec w\cdot \vec{x}^{(i)} + b
$$

And the cost function $J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$  would be changed to:

$$
J(\vec{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)})^2
$$

For gradient descent and the derivatives:

1. $\frac{\partial J(w,b)}{\partial w}$, which was $\frac{1}{m}\left[f_x-y\right]x$, will now be done for each $w_n$ to change each dimension of $\vec{w}$, thus for the $n$th feature and weight it’ll be written:

$$
\frac{\partial J(\vec{w},b)}{\partial w_n} = \frac{1}{m}\sum\limits_{i = 0}^{m-1} \left[f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right]x_n^{(i)}
$$

and:

$$
w_n = w_n - \alpha(\frac{\partial J(\vec{w},b)}{\partial w_n})
$$

1. for $b$, it is:

$$
b = b - \alpha  = \frac{1}{m}\sum\limits_{i = 0}^{m-1} \left[f_{\vec{w},b}(\vec{x}^{(i)}) - y^{(i)}\right]
$$

Alternatively you can minimize the cost function with the **Normal equation,** a non-iterative method. Apparently this only works for linear regression and isnt as efficient when features are >10k. 

## **the syntax:**

$\vec𝐱^{(𝑖)}$ is a vector containing example $i$ where $x_0^{(𝑖)}, x_n^{(𝑖)}$ are the first and nth features of example $i$. When combine into a full set of vectors, you have the training matrix $\mathbf{X}$, containing vectors  $\vec𝐱^{(𝑖)}$ to $\vec𝐱^{(m)}$.

$m$ and $n$ are intentionally used to show the number of matrix rows (aka examples represented as vectors) and number of matrix columns (aka features represented as example vector length).

so for example, this matrix will represent a training dataset.

$$
\mathbf{X} =
\begin{pmatrix}
x_0^{(0)} & x_1^{(0)} & \cdots & x_{n-1}^{(0)} \\
x_0^{(1)} & x_1^{(1)} & \cdots & x_{n-1}^{(1)} \\
\vdots & \vdots & \ddots & \vdots \\
x_0^{(m-1)} & x_1^{(m-1)} & \cdots & x_{n-1}^{(m-1)}
\end{pmatrix}
$$

example outputs / targets and parameters will also berepresented as vectors of length $m$ and $n$  respectively because outputs are for each vector / row $\vec𝐱^{(𝑖)}$ and parameters are for each feature $x_n^{(𝑖)}$. 

<aside>
💡

i expected each row vector to have their own parameters but  I thought it through and saw that if the same model is to approximately describe the whole dataset, one set of parameters need to be used. 

</aside>

$b$ the bias will be a scalar. so basic. 😂

## **the code:**

say you have optimal weights and biases, cost function at this state can be calculated by:

```python
def compute_cost(X, y, w, b): #initialize cost function 
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data set, matrix with m examples / rows and n features / columns
      y (ndarray (m,)) : vector containing outputs / target values for each row
      w (ndarray (n,)) : vector containing weights per parameter
      b (scalar)       : model bias, scalar
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0] #get the number of rows / examples
    cost = 0.0 #initialize cost function variable
    
    #To calculate the cost per row / example, for each row, calculate the prediction and sum them all.
    for i in range(m):                #to do this loop through the matrix rows:                       
       f_wb_i = np.dot(X[i], w) + b   #get a prediction per row: take the row vector X[i] and 
                                      #get the dot product with equally long weight vector w, then, 
                                      #add this to bias b.  
     cost = cost + (f_wb_i - y[i])**2 #take the prediction(f_wb_i),
                                      #subtract the target y[i]from it,
                                      #square the difference.  
                                      #add to cost variable.                                       
    
    #After looping diving cost by 2x m
    cost = cost / (2 * m)                       
    return cost
```

For multiple variables, you need to first compute a gradient for $w$ and $b$ before actually applying it in the gradient descent algorith, viz:

```python
def compute_gradient(X, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
      dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
    """
    m,n = X.shape           #(number of examples, number of features)
    dj_dw = np.zeros((n,))  # vector of length n,containing 0 for the gradient of each feature weight
    dj_db = 0.

#To get the overall derivatives for weights and biases for this dataset:
    for i in range(m):                    #for each row do the following:         
        err = (np.dot(X[i], w) + b) - y[i]   #get prediction cost for row / example X[i]
        for j in range(n):                   #to get weight derivative, for each feature 
                                             # get the value X[i, j] and 
            dj_dw[j] = dj_dw[j] + err * X[i, j] #multiply by the prediction cost for the row / example
                                             #then sum them up
        dj_db = dj_db + err        #to get bias derivative, add up all cost errors for each row.                
    dj_dw = dj_dw / m           #normalize with no. of rows                     
    dj_db = dj_db / m           #normalize with no. of rows                     
        
    return dj_db, dj_dw
```

### Aside:

They implemented the algorithm as such:

```python
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      """
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db,dj_dw = gradient_function(X, y, w, b)   ##None

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw               ##None
        b = b - alpha * dj_db               ##None
      
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            J_history.append( cost_function(X, y, w, b))

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w, b, J_history #return final w,b and J history for graphing
    
    # initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 100000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
```

<aside>
💡

`np.zeros_like(x)` , sick function. creates an array of zeros like that of `x`  - which could be a vector, scalar, etc.

</aside>