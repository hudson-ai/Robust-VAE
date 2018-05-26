import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import xavier_initializer

class MLP(object):
    def __init__(self, n_in, n_out, depth, widths, activation = tf.tanh, 
                 random_seed = None):
        
        self.n_in = n_in
        self.n_out = n_out
        self.depth = depth
        self.activation = activation
        self.random_seed = random_seed
        
        xavier_init = xavier_initializer(seed = random_seed)
        layers = {}
        params = []
        assert depth == len(widths), 'Depth %d != len(widths) %d' %(depth, len(widths))
        for i in range(depth+1):
            layer = {
                'W' :  tf.Variable(xavier_init(
                    [
                        n_in if i == 0 else widths[i-1], 
                        n_out if i == depth else widths[i]
                    ]), 
                                       name = 'W%d'%i),
                'b' : tf.Variable(tf.zeros(
                    [
                        n_out if i == depth else widths[i]
                    ]), 
                                      name = 'b%d'%i)
            }
            params = params + list(layer.values())
            layers.update({i : layer})
            
        self.layers = layers 
        self.params = params
         
    def __call__(self, x, linear_out = True):
        assert x.shape[1] == self.n_in, 'x.shape[1] %d != self.n_in %d' %(x.shape[1], self.n_in)
        value = x
        for i in range(self.depth + 1):
            W = self.layers[i]['W']
            b = self.layers[i]['b']
            if i == self.depth and linear_out:
                value = value @ W + b
            else:
                value = self.activation(value @ W + b)
        return(value)
            
            
            