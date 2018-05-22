import networks
import lib

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

class RobustVAE(object):
    def __init__(self, encoder, decoder):
        assert encoder.n_in == decoder.n_out, 'encoder.n_in %d != decoder.n_out %d' %(encoder.n_in, decoder.n_out)
        assert encoder.n_out == decoder.n_in, 'encoder.n_out %d != decoder.n_in %d' %(encoder.n_out, decoder.n_in)
        
        self.encoder = encoder
        self.decoder = decoder
        self.params = encoder.params + decoder.params
        self.data_dim = encoder.n_in
        self.latent_dim = decoder.n_in
        
        self.X = tf.placeholder(tf.float32, shape=[None, self.data_dim], name='X')
        self.Z = encoder(self.X)
        self.X_hat = decoder(self.Z)
    
    def train(self, x, n_dense = None, n_sparse = 0, iters=100, batch_size=500, lr=1e-3):
        if n_dense is None:
            if n_sparse <= self.latent_dim:
                n_dense = self.latent_dim - n_sparse
            else: n_dense = 0
        assert n_dense + n_sparse == self.latent_dim, 'n_dense %d + n_sparse %d != self.latent_dim %d' %(n_dense,n_sparse, self.latent_dim)
        self.n_dense = n_dense
        self.n_sparse = n_sparse
        #Define log pdf over latent space
        logp = lambda z: lib.log_joint_pdf(z, n_dense, n_sparse)
        
        #Losses 
        self.L2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.X - self.X_hat), 1))
        self.Stein = -tf.reduce_mean(tf.einsum('ij,ij->i', self.Z, tf.stop_gradient(lib.phi_star(self.Z, logp))))
        self.Loss = self.L2 + self.Stein
        
        #Optimizer 
        solver = tf.train.AdagradOptimizer(learning_rate = lr).minimize(self.Loss, var_list=self.params)
        
        #Training
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.sess = sess
        with sess.as_default():
            sess.run(init)

            losses = []
            l2s = []
            steins = []
            for i in range(iters):
                mb_idx = np.random.randint(x.shape[0], size = (batch_size))
                x_mb = x[mb_idx]
                _, loss_curr, l2_curr, stein_curr = sess.run([solver, self.Loss, self.L2, self.Stein], feed_dict = {self.X:x_mb})
                losses.append(loss_curr)
                l2s.append(l2_curr)
                steins.append(stein_curr)
            plt.figure()
            plt.plot(losses)
            plt.figure()
            plt.plot(l2s)
            plt.figure()
            plt.plot(steins)
            
    def sample_z(self, n):
        return(lib.sample_dense_sparse(n, self.n_dense, self. n_sparse))
    
    def sample(self, n):
        return(self.decoder(self.sample_z(n)))