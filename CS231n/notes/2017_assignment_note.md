# Assignment 1

- knn, no loops compute distances functions
    - to fully vectorize, it's necessary to anaylse the computing process and find out how to break into several parts
- matrix operations
    - 常数在矩阵乘法中可以随意换换位置，kAB = ABk, k是实数，A,B是矩阵
- backpropagation trick
    - 在运算是矩阵相乘时，有WX = h, dL/dW = dh/dW * dL/dh，则dL/dW = X.T*dh
    - 之所以能这样做的原因是，对于不同的Xi对Wj求出来的梯度是直接相加的
- computational graph v.s. neural networks architeture
    - 两者并不相同，computational graph是数学意义上的，用于正反向传播计算过程，而neural networks architeture只是一个示意结构，是深度学习意义上的

# Assignment 2
- spatial batch norm
    - the means & variances correspond to the channels, for each channel, there is one mean & variance
    - so means & variances are computed by converge N examples, H heights & W width, just put those at the same chanel  together to compute
