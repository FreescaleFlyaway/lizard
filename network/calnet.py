# $File: policy_rep: policy representation
# $Usage: This file is used to establish the abstract net for attribute
# In order to decrease the covariation of net, we will save those model in Num:
# Each config will be saved in the dict.json file.

import tensorflow as tf
import network.core_net as br 
import network.write_dict as write_dict


class Calnet(object):
    def __init__(self, policy_config):
        #  index:
        if policy_config['update_name'] is not None:
            self.index = (policy_config['environment'].split(',')).index(policy_config['update_name'])
        else:
            self.index = 0

        #  connect in two world
        self.map_dict = {}
        self.sess = tf.Session()
        self.policy_config = policy_config

        #  reload save dict:
        self.save_path_dict = write_dict.write_dict() 
        for name in self.save_path_dict.keys():
            self.save_path_dict[name] = self.policy_config['save_dir'] + self.save_path_dict[name]

        #  infos:
        self.update_name = self.policy_config['update_name']
        self.base_name = self.policy_config['a_names'].split(',')[0]
        self.attrib_num = self.policy_config['attribute_num']

        #  init states
        self.s = []
        for dim_num in self.policy_config['state_dim'].split(','): 
            _s = tf.placeholder(tf.float32, [None, int(dim_num)])
            self.s.append(_s)

        #  mappings: attrib_num should include the base attribute: statei
        for i in range(self.attrib_num):
            self.map_dict['state{}'.format(i)] = self.s[i]

        #  CALN self.mean, self.sigma are those for output:

        self.means = []
        self.sigmas = []
        self.sigma_inits = {}
        self.saver_dict = {}
        self.param_dict = {}

        self.build_a_network();
        self.build_c_network();
        self.build_t_network();

        # local restart:
        if self.policy_config['partial_restart']:
            self.init_dicts = {} 
            for name, variables in self.param_dict.items():
                self.init_dicts[name] = tf.variables_initializer(variables)

    def build_a_network(self):
        #  a net setup:
        for i, name in enumerate(self.policy_config['a_names'].split(',')):
            #  different attrib  has different dim of action
            _action_dim = int(self.policy_config['action_dim'].split(',')[i])
            with tf.variable_scope(name) as scope:
                #  if reuse attribute
                if name in self.policy_config['a_names'].split(',')[:i]:
                    scope.reuse_variables()
                if i == 0:
                    mean, sigma, sigma_init = br.actor_net(self.s[0], _action_dim)
                    _mean = mean  #  mean of first attrib
                    #  old_mean part:
                    if name == self.update_name:
                        #  setting action_dim
                        self.action_dim = _action_dim
                        with tf.variable_scope('old'):
                            self.old_mean, self.old_sigma, _ = br.actor_net(self.s[0], _action_dim)
                else:
                    cs = tf.concat([self.s[i], _mean], axis=-1)  #  concat with the first attrib
                    mean, sigma, sigma_init = br.actor_net(cs, _action_dim, self.policy_config['suppress_ratio'])
                    #  old_mean part:
                    if name == self.update_name:
                        self.action_dim = _action_dim
                        with tf.variable_scope('old'):
                            self.old_mean, self.old_sigma, _ = br.actor_net(cs, _action_dim, self.policy_config['suppress_ratio'])

                if name == self.update_name:
                    self.param_dict['{}_old_action'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/old/{}'
                                                                              .format(name, 'actor'))
                self.sigma_inits[name] = sigma_init
                self.means.append(mean)
                self.sigmas.append(sigma)

                self.param_dict['{}_action'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/{}'
                                                                              .format(name, 'actor'))
                self.saver_dict['{}_action'.format(name)] = tf.train.Saver(self.param_dict['{}_action'.format(name)])
                #  update_variable:
                with tf.variable_scope('update_oldpi'):
                    self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.param_dict['{}_action'.format(self.update_name)], self.param_dict['{}_old_action'.format(self.update_name)])]

    def build_c_network(self):
        if self.policy_config['c_activate']:
            #  value is the average reward with policy
            self.value = 0
            self.values = {}
            for i, name in enumerate(self.policy_config['c_names'].split(',')):
                with tf.variable_scope(name) as scope:
                    if name in self.policy_config['c_names'].split(',')[:i]:
                        scope.reuse_variables()
                    if i == 0:
                        value = br.critic_net(self.s[0])
                    else:
                        #  we don't need to concern with mean, which is too detail to converge
                        value = br.critic_net(self.s[i], self.policy_config['suppress_ratio'])

                    self.values[name] = value
                    
                    #  value is already added
                    self.value = value + self.value
                    self.param_dict['{}_value'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                 '{}/{}'.format(name, 'critic'))
                    self.saver_dict['{}_value'.format(name)] = tf.train.Saver(self.param_dict['{}_value'.format(name)])
    
    def build_t_network(self):
        if self.policy_config['t_activate']:
            # activation is the average reward without policy
            # use activation and train activation is different
            # activation = pre_value - {no policy reward} # means the importance of target policy
            self.activations = {}
            for i, name in enumerate(self.policy_config['t_names'].split(',')):
                with tf.variable_scope(name) as scope:
                    if name in self.policy_config['t_names'].split(',')[:i]:
                        scope.reuse_variables()
                    # activation only use for additional attrib
                    activation = br.activate_net(self.s[i+1])

                    self.activations[name] = activation
                    self.param_dict['{}_activation'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                                      '{}/{}'.format(name, 'activate'))
                    self.saver_dict['{}_activation'.format(name)] = tf.train.Saver(self.param_dict['{}_value'
                                                                                   .format(name)])
        
    def save(self):
        # get param_save:
        self.policy_config['param_save'] = {}
        # a:
        for i, value in enumerate(self.policy_config['a_param_save'].split(',')):
            self.policy_config['param_save']['{}_action'.format(self.policy_config['a_names'].split(',')[i])] = \
                True if value == 'True' else False
        # c
        if self.policy_config['c_activate']:
            for i, value in enumerate(self.policy_config['c_param_save'].split(',')):
                self.policy_config['param_save']['{}_value'.format(self.policy_config['c_names'].split(',')[i])] \
                    = True if value == 'True' else False

        # t
        if self.policy_config['t_activate']:
            for i, value in enumerate(self.policy_config['t_param_save'].split(',')):
                self.policy_config['param_save']['{}_activation'.format(self.policy_config['t_names'].split(',')[i]
                                                                             )] = True if value == 'True' else False

        for name in self.param_dict.keys():
            if self.policy_config['param_save'][name]:
                self.saver_dict[name].save(self.sess,
                                           self.save_path_dict['{}_{}'.format(self.base_name, name)])

    def restore(self):
        self.policy_config['param_restore'] = {}
        # a:
        for i, value in enumerate(self.policy_config['a_param_restore'].split(',')):
            self.policy_config['param_restore'][
                '{}_action'.format(self.policy_config['a_names'].split(',')[i])] = True if value == 'True' else False
        # c
        if self.policy_config['c_activate']:
            for i, value in enumerate(self.policy_config['c_param_restore'].split(',')):
                self.policy_config['param_restore'][
                    '{}_value'.format(self.policy_config['c_names'].split(',')[i])] = True if value == 'True' else False

        # t
        if self.policy_config['t_activate']:
            for i, value in enumerate(self.policy_config['t_param_restore'].split(',')):
                self.policy_config['param_restore']['{}_activation'.format(
                    self.policy_config['t_names'].split(',')[i])] = True if value == 'True' else False

        for name in self.param_dict.keys():
            if self.policy_config['param_restore'][name]:
                self.saver_dict[name].restore(self.sess,
                                              self.save_path_dict['{}_{}'.format(self.base_name, name)])

    def restart_part(self):
        # name is action/value/activation
        # all of the base part should be update_name
        if self.policy_config['a_trainable']:
            self.sess.run(self.init_dicts['{}_{}'.format(self.policy_config['update_name'], 'action')])
            print('{}_{} restart'.format(self.policy_config['update_name'], 'action'))
        if self.policy_config['c_trainable']:
            self.sess.run(self.init_dicts['{}_{}'.format(self.policy_config['update_name'], 'value')])
            print('{}_{} restart'.format(self.policy_config['update_name'], 'action'))
        if self.policy_config['t_activate']:
            if self.policy_config['t_trainable']:
                self.sess.run(self.init_dicts['{}_{}'.format(self.policy_config['update_name'], 'activation')])
                print('{}_{} restart'.format(self.policy_config['update_name'], 'action'))

    def refresh_sigma(self):
        # we refresh update_name
        self.sess.run(self.sigma_inits[self.policy_config['update_name']])

    def update_old_pi(self):
        self.sess.run(self.update_oldpi_op)
     
    def get_feed_dict(self, data):
        feed_dict = {}
        for name, d in data.items():
            if name in self.map_dict.keys():
                feed_dict[self.map_dict[name]] = d
        return feed_dict

    def predict_mean(self, data):
        feed_dict = self.get_feed_dict(data)
        return [self.sess.run(_mean, feed_dict) for _mean in self.means]

    def predict_sigma(self, data):
        feed_dict = self.get_feed_dict(data)
        return [self.sess.run(_sigma, feed_dict) for _sigma in self.sigmas]

    def predict_value(self, data, name=None):
        feed_dict = self.get_feed_dict(data)
        if name is not None:
            return self.sess.run(self.values[name], feed_dict)
        else:
            return self.sess.run(self.value, feed_dict)
