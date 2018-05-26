import tensorflow as tf
import numpy as np
EPS = 1e-8

####################Stein##########################################
def sq_dists(Z):
    A = tf.reshape(Z, (tf.shape(Z)[0], -1))
    r = tf.reduce_sum(A*A, 1)
    r = tf.reshape(r, [-1, 1])
    sqdists = r - 2*tf.matmul(A, tf.transpose(A)) + tf.transpose(r)
    return sqdists

def med(D):
    m = tf.contrib.distributions.percentile(D, 50)
    return m
    
def kernel(Z, h = -1):
    sqdists = sq_dists(Z)
    if h <= 0:
        medsq = med(sqdists)
        h = medsq / tf.log(tf.to_float(tf.shape(Z)[0]))
    h = tf.stop_gradient(h)
    ker = tf.exp(-sqdists/h)
    A = tf.tile(tf.expand_dims(Z, 0), (tf.shape(Z)[0],1,1))
    T = tf.transpose(A, (1,0,2)) - A
    dker_dz = -2*tf.multiply(tf.tile(tf.expand_dims(ker, 2), (1,1,tf.shape(Z)[1])), T)/h
    return(ker, dker_dz)

def phi_star(Z, logp, h = -1):
    ker, dker = kernel(Z, h=h)
    dlogp = tf.gradients(logp(Z), Z)[0]
    phi_mat = tf.einsum('ij,ik->ijk', ker, dlogp) + dker
    phi_mean = tf.reduce_mean(phi_mat, 0)
    return(phi_mean)


##########################Density Functions#######################
def log_normal_pdf(Z):
    matlike = tf.log((2*np.pi)**(-1/2) * tf.exp(-tf.square(Z)/2) + EPS)
    return(tf.reduce_sum(matlike, 1))
            
def log_laplace_pdf(Z):
    b = .01
    matlike = tf.log((2*b)**(-1) * tf.exp(-tf.abs(Z)/b) + EPS)
    return(tf.reduce_sum(matlike, 1))

def log_joint_pdf(Z, n_dense, n_sparse):
    assert Z.shape[1] == n_dense + n_sparse, 'Z.shape[1] %d != n_dense %d + n_sparse %d' %(Z.shape[1], n_dense, n_sparse)
    return log_normal_pdf(Z[:, :n_dense]) + log_normal_pdf(Z[:, n_dense:])

def sample_dense_sparse(n, n_dense, n_sparse):
    out = np.zeros((n, n_dense+n_sparse))
    out[:, :n_dense] = np.random.normal(0,1, (n,n_dense))
    out[:, n_dense:(n_dense+n_sparse)] = np.random.laplace(0,1, (n,n_sparse))
    return(out)

###########################Gumbel########################################
def gumbel_soft(pi, g, tau=.1):
    logit = tf.log(pi)
    y = tf.nn.softmax((g+logit)/tau, axis = 2)
    return(y)