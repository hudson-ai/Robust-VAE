import networks
import lib

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

EPS = 1e-8

class LaplaceVAE(object):
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
    
class L0VAE(object):
    def __init__(self, encoder, decoder, cts_dim, sparse_dim):
        #Encoder n_out must equal cts_dim*2 + sparse_dim for reparametrization trick
        self.encoder = encoder
        #Decoder n_in must equal cts_dim + sparse_dim, the latent space shape after reparametrization
        self.decoder = decoder
        
        self.params = encoder.params + decoder.params
        
        self.data_dim = encoder.n_in
        self.cts_dim = cts_dim
        self.sparse_dim = sparse_dim
        
        #Data
        self.X = tf.placeholder(tf.float32, shape=[None, self.data_dim], name='X')
        
        #Encode for reparametrization trick
        self.Z_params = encoder(self.X, linear_out=True)
        
        #Reparametrization trick for continuous portion
        self.stds = self.Z_params[:, :self.cts_dim]
        self.means = self.Z_params[:, self.cts_dim:self.cts_dim*2]
        self.eps = tf.placeholder(tf.float32, shape=[None, self.cts_dim], name='eps')
        self.cts_z = self.stds*self.eps + self.means
        
        #Gumbel reparametrization
        self.probs = tf.sigmoid(self.Z_params[:, cts_dim*2:])
        self.pis = tf.stack((self.probs, 1-self.probs), axis=2)
        self.g = tf.placeholder(tf.float32, shape=self.pis.shape, name='g')
        self.sparse_z = lib.gumbel_soft(self.pis, self.g, tau = .1)[:,:,0]
        
        #Concatenate reparametrized cts and sparse dims 
        self.Z = tf.concat((self.cts_z, self.sparse_z), axis = 1)
        
        #Decode
        self.X_hat = decoder(self.Z)
        
    def train(self, x, sparsity = 0, iters=100, batch_size=64, lr=1e-3):
        #Define log pdf over latent space
        logp = lambda z: lib.log_normal_pdf(z)

        #Losses 
        #Autoencoder
        self.L2 = tf.reduce_mean(tf.reduce_sum(tf.square(self.X - self.X_hat), 1))
        #Cts variable
        self.Stein = -tf.reduce_mean(tf.einsum('ij,ij->i', self.cts_z, tf.stop_gradient(lib.phi_star(self.cts_z, logp))))
        #Sparse variable
        Q = tf.reduce_mean(self.sparse_z, 0)
        self.sparse_KL = tf.reduce_sum(
            tf.multiply(Q, tf.log(Q/sparsity + EPS)) + tf.multiply(1-Q, tf.log((1-Q)/(1-sparsity) + EPS))
        )
        self.Loss = self.L2 + self.Stein + self.sparse_KL

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
            sparse_kls = []
            for i in range(iters):
                mb_idx = np.random.randint(x.shape[0], size = (batch_size))
                x_mb = x[mb_idx]
                eps = np.random.normal(size=(len(x_mb), self.cts_dim))
                g = np.random.gumbel(size=(len(x_mb), self.sparse_dim, 2))
                _, loss_curr, l2_curr, stein_curr, sparse_kl_curr = sess.run([solver, self.Loss, self.L2, self.Stein, self.sparse_KL], feed_dict = {self.X:x_mb, self.eps: eps, self.g: g})
                losses.append(loss_curr)
                l2s.append(l2_curr)
                steins.append(stein_curr)
                sparse_kls.append(sparse_kl_curr)

            plt.figure()
            plt.plot(losses)
            plt.title('total loss')
            
            plt.figure()
            plt.plot(l2s)
            plt.title('autoenc')
            
            plt.figure()
            plt.plot(steins)
            plt.title('cts steins')
            
            plt.figure()
            plt.plot(sparse_kls)
            plt.title('sparse_kl')
            

    def encode(self, x):
        eps = np.random.normal(size=(len(x), self.cts_dim))
        g = np.random.gumbel(size=(len(x), self.sparse_dim, 2))
        return(self.sess.run(self.Z, feed_dict = {self.X:x, self.eps: eps, self.g: g}))