# $File: core_net.py
# $Author: Harvey Chang
import tensorflow as tf
import numpy as np


def actor_net(obs_ph, act_dim, suppress_ratio=1.0):
    with tf.variable_scope('actor'):
        obs_dim = obs_ph.shape.as_list()[-1]  # the last dim of shape
        hid1_size = obs_dim * 10
        hid3_size = act_dim * 10
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        # hidden net:
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio*np.sqrt(1/obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio*np.sqrt(1/hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio*np.sqrt(1/hid2_size)), name="h3")
        means = tf.layers.dense(out, act_dim, tf.tanh, kernel_initializer=tf.random_normal_initializer(
                                    stddev=suppress_ratio*np.sqrt(1 / hid3_size)), name='means')
        # variance:
        log_vars = tf.get_variable('logvars', [act_dim], tf.float32, 
            tf.random_normal_initializer(mean=-2, stddev=1.0/act_dim)) 

        sigma_init = tf.variables_initializer([log_vars], 'sigma_initializer')
        sigma = tf.exp(log_vars) 
        return means, sigma, sigma_init
      
        
def critic_net(obs_ph, suppress_ratio=1.0):
    with tf.variable_scope('critic'):
        obs_dim = obs_ph.shape.as_list()[-1]  
        hid1_size = obs_dim * 10  
        hid3_size = 10  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / hid2_size)), name="h3")
        out = tf.layers.dense(out, 1,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=suppress_ratio * np.sqrt(1 / hid3_size)), name='output')
        out = tf.squeeze(out)
        return out


def activate_net(obs_ph):
    with tf.variable_scope('activate'):
        obs_dim = obs_ph.shape.as_list()[-1]  
        hid1_size = obs_dim * 10  
        hid3_size = 10  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        out = tf.layers.dense(out, 1,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid3_size)), name='output')
        out = tf.squeeze(out)
        return out

