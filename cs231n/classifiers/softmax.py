from builtins import range
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
    num_train=X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)***** 
    
    
    for i in range(X.shape[0]):
        
        scores=X[i].dot(W)         # interpreted as unnormalized log probabilites
       
        
        scores-=np.amax(scores)  # for numerical stability
        
        exp_scores=np.exp(scores)
        sum_exp_scores=np.sum(exp_scores)
        
        normalize_scores=exp_scores/(sum_exp_scores) 
        correct_scoreloss=-1*(np.log(normalize_scores[y[i]]))
        loss=loss+(correct_scoreloss)
        
        for j in range(W.shape[1]):
            if y[i]==j:
                dW[:,j]=dW[:,j]+(X[i]*(normalize_scores[j]-1))
            
            else:
                dW[:,j]=dW[:,j]+(X[i]*(normalize_scores[j]))
            
            
    loss=loss/num_train     #averaging the loss from all examples
    loss=loss+reg*(np.sum(W*W))
        
    dW/=num_train  # hence averging the delta for W also 
    dW+=reg*2*(W)
    
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train=X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    scores=X.dot(W)
    
    exp_scores=np.exp(scores)
    
    sum_exp_scores=np.sum(exp_scores,axis=1)
    
    normalize_scores=np.divide(exp_scores.T,sum_exp_scores).T
    
    correct_class_scores=normalize_scores[range(X.shape[0]),y]
    
    loss=(np.sum(-1*np.log(correct_class_scores)))
    
    loss=loss/num_train + (reg*(np.sum(W*W)))
    
    # now code for derivative 
    normalize_scores[range(X.shape[0]),y]=normalize_scores[range(X.shape[0]),y]-1
    
    dW=((X.T).dot(normalize_scores))/(num_train)
    dW+=reg*2*(W)
    
    
   
    
    
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
