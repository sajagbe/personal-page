+++
title = 'Numpy and Tensors'
date = 2025-01-25T17:30:00-05:00
draft = false
tags = ["machine learning","andrew ng", "numpy","tensors"]
+++

this is a summary of my understanding of the optional lab on covered numpy, arrays, vectors and matrices. 

because we will be working with huge sets of data and we’ll be manipulating weights, properties and biases repeatedly, we need to do these operations efficiently. Thus, the `numpy` package can help as it has vectors and matrix properties, which are faster and less memory intensive compared to hardcoded calculations. 

to create both matrices and vectors the `np.array` syntax is typically used as such: 

 `np.array([1, 2, 4, 8])`, which yields `[1 2 4 8]`, a vector of numbers.

to convert this to a matrix use the `reshape(x,y)` function where `x` is the number of rows and y is the no of columns. 

when you use `-1` as `x`, `numpy` figures out the optimal number of even rows each containing y number of items as columns. e.g for the above data set, of if I did `reshape(-1,4)` or `reshape(-1, 2)`, I’ll get `[[1], [2], [3], [4]]` or `[[ 1,2], [3,4]]` respectively.

You can also just specify the number of rows a columns in the reshape function, e.g. for the second sample you could also do `reshape(2,2)`.

scalars (i.e regular numbers), vectors and matrices belong to a class called tensors  - apparently named so because they stretch the properties, possibilities and relationships of numbers and relationships between them. 

### **numpy vector operations:**

1. scaling 
2. slicing
3. dot product

### **numpy matrix operations:**

1. indexing aka referencing 
2. slicing
3. multiplication 

> for this the number of rows of one must match the other. so for matrices d and c to multiply, d has to have has many rows as c has columns and the output will have as many columns as d and rows as c.
> 

4. scaling

> if a = `[[ 1,2, 3,4]]` and b = `[ 1,2, 3,4` . I found this confusing because they look identical but apparently they are different. I currently believe they are different not because of their physical uniqueness - which is negligible - but by the computers interpretation of what they are and thus the operations it can perform on them. in numpy a is a matrix with 1 row and 4 columns and b is a vector with 4 columns.
> 

### numpy tensor shape and size:

for a above `a.shape` will give `(1,4)` while b will give `(,4)`. This is because matrices are 2 dimensional and vectors can only be 1 dimensional. This one dimensional can be vertical (giving `(4,)` or horizontal as b above. For a scalar, `.shape` will give an empty tuple because it is 0 dimensional. 

`tensor.size` gives the total number of items in the tensor. therefore `a and b` above will have the same size.