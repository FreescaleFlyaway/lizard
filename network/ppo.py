#  Author: Harvey Chang
#  Email: chnme40cs@gmail.com
#  ppo defines the a_loss, c_loss, t_loss
import tensorflow as tf
import network.calnet as cl


class PolicyRep(cl.Calnet):
    def __init__(self, policy_config):
        cl.Calnet.__init__(self, policy_config)
        #  info:
        self.epsilon = float(self.policy_config['epsilon'])
        self.norms = []
        self.As = []
        #  get As| should be in list
        for _mean, _sigma in zip(self.means, self.sigmas):
            self.norms.append(tf.contrib.distributions.Normal(_mean, _sigma))
            self.As.append(tf.squeeze(self.norms[-1].sample()))
        
        #  Loss structures:
        #  c_loss
        if self.policy_config['c_activate'] and \
                (eval(self.policy_config['c_trainable'].split(',')[self.index])):
            self.Adam_c = tf.train.AdamOptimizer(float(self.policy_config['lr_c']))

            with tf.name_scope('c_loss'):
                self.val_ph = tf.placeholder(tf.float32, (None, 1), 'val_valfunc')
                self.map_dict['q_vals'] = self.val_ph  #  we are using discounted sum reward to be the value placeholder

                #  we only update the update name
                self.c_loss = tf.reduce_mean(tf.square(self.value - self.val_ph), name='c_loss')
                self.c_grads = tf.gradients(self.c_loss,
                                            self.param_dict['{}_value'.format(self.update_name)], name='c_gradient')
                self.c_update_op = self.Adam_c.apply_gradients(
                    zip(self.c_grads, self.param_dict['{}_value'.format(self.update_name)]))
                tf.summary.scalar('c_loss', self.c_loss)

        #  t_loss
        if self.policy_config['t_activate'] and eval(self.policy_config['t_trainable'].split(',')[max(self.index-1, 0)]):

            self.Adam_t = tf.train.AdamOptimizer(float(self.policy_config['lr_t']))
            with tf.name_scope('t_loss'):
                self.nc_val_ph = tf.placeholder(tf.float32, (None, 1), 'none_compensate_value')
                # activation is supposed to = value - nc_value > 0
                self.map_dict['nc_values'] = self.nc_val_ph
                self.activation_ph = self.value - self.nc_val_ph  #  compensated value - uncompensated value
                self.t_loss = tf.reduce_mean(tf.square(self.activations[self.update_name] - self.activation_ph))
                self.t_grads = tf.gradients(self.t_loss, self.param_dict['{}_activation'.format(self.update_name)])
                self.t_update_op = self.Adam_t.apply_gradients(
                    zip(self.t_grads, self.param_dict['{}_activation'.format(self.update_name)]))
                tf.summary.scalar('t_loss', self.t_loss)

        #  a_loss
        if eval(self.policy_config['a_trainable'].split(',')[self.index]):
            self.norm = self.norms[self.index]
            self.Adam_a = tf.train.AdamOptimizer(float(self.policy_config['lr_a']))
            with tf.name_scope('a_loss'):
                with tf.name_scope('old_normal'):
                    self.old_norm = tf.contrib.distributions.Normal(self.old_mean, self.old_sigma)

                with tf.name_scope('act_advantage'):
                    self.act_ph = tf.placeholder(tf.float32, (None, self.action_dim), 'act') 
                    self.advantages_ph = tf.placeholder(tf.float32, (None, 1), 'advantages')  # (None, 1)
                    
                    self.map_dict['advantages'] = self.advantages_ph
                    self.map_dict['old_acts'] = self.act_ph

                with tf.name_scope('r'):
                    self.r = tf.reduce_mean(self.norm.prob(self.act_ph) /
                                            (self.old_norm.prob(self.act_ph) + 1e-6), axis=-1)  # (None, 2)-> (None, 1)
                    self.r_clip = tf.clip_by_value(self.r, 1 - self.epsilon, 1 + self.epsilon)
                with tf.name_scope('J_clip'):
                    self.J = tf.minimum(self.r * self.advantages_ph, self.r_clip * self.advantages_ph)
                    self.a_loss = -tf.reduce_mean(self.J, name='a_loss')
                    
                self.a_grads = tf.gradients(self.a_loss, self.param_dict['{}_action'.format(self.update_name)],
                                            name='a_grads')
                self.a_update_op = self.Adam_a.apply_gradients(
                    zip(self.a_grads, self.param_dict['{}_action'.format(self.update_name)]))
                tf.summary.scalar('a_loss', self.a_loss)
        
        #  summary structure
        self.summary_op = tf.summary.merge_all()
        self.log_dir = self.policy_config['log_dir'] 
        self.summary_writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

    def update(self, data):
        #  general update function
        for name in self.policy_config['update_list']:
            eval('self.update_{}(data)'.format(name))

    def update_c(self, data):
        feed_dict = self.get_feed_dict(data)
        self.sess.run(self.c_update_op, feed_dict) 
    
    def update_a(self, data):
        feed_dict = self.get_feed_dict(data)
        self.sess.run(self.a_update_op, feed_dict) 

    def update_t(self, data):
        feed_dict = self.get_feed_dict(data)
        self.sess.run(self.t_update_op, feed_dict) 
     
    def choose_action(self, data):
        feed_dict = self.get_feed_dict(data)
        return [self.sess.run(_A, feed_dict) for _A in self.As]

    def log(self, data):
        feed_dict = self.get_feed_dict(data)
        summary_str = self.sess.run(self.summary_op, feed_dict)
        self.summary_writer.add_summary(summary_str)
