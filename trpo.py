import numpy as np
import math
import random
from scipy import signal
import tensorflow as tf

improved = 0
save_loss = True
save_models = True
resume_training = False
use_ppo = False
LOGSTD = tf.math.log(0.005)
SPEED = 0.02
BUFFER_RATIO = 0.67

def discount(x, gamma):
    assert x.ndim >= 1
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2 * var) - .5*tf.math.log(tf.constant(2*np.pi, dtype=tf.float32)) - logstd
    return tf.reduce_sum(gp, [1])

def gauss_ent(logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*tf.math.log(2*math.pi*math.e), tf.float32))
    return h

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
            "shape function assumes that shape is fully known"
    return out

def numel(x):
    return np.prod(x.shape)

def get_flat(model):
    var_list = model.trainable_weights
    return tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

def set_from_flat(model, theta):
    var_list = model.trainable_weights
    shapes = [v.shape for v in var_list]
    start = 0

    for (shape, v) in zip(shapes, var_list):
        size = np.prod(shape)
        v.assign(tf.reshape(theta[start:start + size], shape))
        start += size

def flatgrad(model, loss, tape, type='n'):
    var_list = model.trainable_weights
    if type == 'n': grads = tape.gradient(loss, var_list)
    else: grads = tape.jacobian(loss, var_list, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)

def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return gauss_KL(mu1, logstd1, mu2, logstd2)

def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)
    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl

def normalize_mu(action_mu):
    norm = tf.math.sqrt(tf.math.add(tf.math.pow(tf.gather(action_mu, 0, axis=1),2), tf.math.pow(tf.gather(action_mu, 1, axis=1),2)))
    norm = tf.expand_dims(norm, 1)

    if tf.where(tf.math.greater(norm, 1e-8)).shape[0] != 0:
        action_mu /= norm
    else:
        action_mu = tf.concat([tf.ones((action_mu.shape[0],1)), tf.zeros((action_mu.shape[0],1))], 1)
    
    action_mu *= SPEED

    return action_mu

def linesearch(f, x, feed, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x, feed)[0]
    for (_, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew, feed)[0]
        actual_improve = fval - newfval
        # actual_improve = newfval - fval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            global improved
            improved += 1
            return xnew
    return x

def conjugate_gradient(f_Ax, feed, b, cg_iters=10, residual_tol=1e-10):
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p, feed)
        z = z.numpy()
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        assert z.shape == p.shape and p.shape == x.shape \
            and x.shape == r.shape, "Conjugate shape difference"
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x

class ReplayBuffer():
    def __init__(self, buffer_size=1024, sample_size=512):
        self.buffer_size = buffer_size
        self.sample_size = sample_size
        self.buffer = []
    
    def len(self):
        return len(self.buffer)
    
    def add(self, trajectory):
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(trajectory)
        else:
            self.buffer.pop(0)
            self.buffer.append(trajectory)
    
    def sample(self):
        if len(self.buffer) < self.sample_size:
            return random.sample(self.buffer, len(self.buffer))
        else:
            return random.sample(self.buffer, self.sample_size)