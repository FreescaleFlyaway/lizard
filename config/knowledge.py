# -*- coding:utf-8 -*-
#  configuration mapping


knowledge_dict = dict()
knowledge_dict['state_dim'] = {'fish':'41'}
knowledge_dict['action_dim'] = {'fish':'5'}


def network_config(config):
    if config['run_type'] == 'train':
        config['a_names'] = config['env_type']
        config['c_names'] = config['env_type']
        config['c_activate'] = True
        config['t_activate'] = False
        config['a_trainable'] = ''
        config['c_trainable'] = ''
        for i in range(config['attribute_num']):
            if i != config['index']:
                config['a_trainable'] += 'False,'
                config['c_trainable'] += 'False,'
            else:
                config['a_trainable'] += 'True,'
                config['c_trainable'] += 'True,'

        config['a_trainable'] = config['a_trainable'][:-1] 
        config['c_trainable'] = config['c_trainable'][:-1] 

    elif config['run_type'] == 'test':
        config['a_names'] = config['env_type']
        config['c_names'] = config['env_type']
        config['c_activate'] = False
        config['t_activate'] = False
        config['a_trainable'] = ''
        config['c_trainable'] = ''
        for i in range(config['attribute_num']):
            config['a_trainable'] += 'False,'
            config['c_trainable'] += 'False,'

        config['a_trainable'] = config['a_trainable'][:-1] 
        config['c_trainable'] = config['c_trainable'][:-1] 


