import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_class_score = scores[y]
  a = 0.0
  b = 0.0
  for i in xrange(num_train):
    logC = -np.max(scores[i])
    loss += -(scores[i,y[i]] + logC) + np.log(np.sum(np.exp(scores[i] + logC)))
    a += np.log(np.sum(np.exp(scores[i] + logC)))
    b -= scores[i,y[i]] + logC
    for j in xrange(num_classes):
      dW[:,j] += np.exp(scores[i,j] + logC) / np.sum(np.exp(scores[i] + logC)) * X[i,:]
      if j == y[i]:
        dW[:,j] -= X[i,:]

  loss /= num_train
  dW /= num_train

  loss += reg * np.sum(W * W)
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  logC = -np.max(scores, axis=1)
  logC.shape = (num_train, 1)

  multiplier = np.exp(scores + logC)

  sum_mul = np.sum(multiplier, axis=1)
  sum_mul.shape = (num_train, 1)
  multiplier = multiplier / sum_mul

  loss -=np.sum(np.log(multiplier[np.arange(num_train), y]))

  multiplier[np.arange(num_train), y] -= 1
  dW = X.T.dot(multiplier)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

