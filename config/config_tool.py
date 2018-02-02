# -*- coding:utf-8 -*-
import json
import config.knowledge as K

if __name__ == "__main__":
    #  setting configuration:
    config_name = 'fish_train'

    config = dict()
    config['env_type'] = 'fish'
    config['update_name'] = 'fish'  #  can be none

    config['environment'] = 'fish'

    config['run_type'] = 'train'
    config['e_update_type'] = 'curriculum'
    #  save directory:
    config['save_dir'] =  'train_log/model'
    config['log_dir'] = 'train_log/log'

    #  construction configuration:
    #  action
    config['a_names'] = 'fish'
    config['a_trainable'] = 'True'

    #  critic
    config['c_activate'] = True
    config['c_names'] = 'fish'
    config['c_trainable'] = 'True'

    #  activate
    config['t_activate'] = False
    config['t_names'] = ''
    config['t_trainable'] = 'False'


    #  tricky:
    config['partial_restart'] = True
    config['suppress_ratio'] = 1.0

    #  trainiing parameters:
    config['lr_a'] = 1e-4
    config['lr_c'] = 1e-4
    config['lr_t'] = 1e-4
    config['epsilon'] = 0.2
    
    
    #  training configuration:
    config['n_iter'] =  5  #  five is for test
    config['max_steps_per_epoch'] = 50
    config['min_steps_per_iter'] = 100
   
    config['gamma'] = 0.9

    config['a_update_steps'] = 20
    config['c_update_steps'] = 20
    config['t_update_steps'] = 20

    #  auto_generating
    config['index'] = 0 if config['update_name'] == 'none' else config['env_type'].split(',').index(config['update_name'])
    #  dimension:
    config['state_dim'] = ''
    config['action_dim'] = ''
    config['attribute_num'] = len(config['env_type'].split(','))
    for atrb in config['env_type'].split(','):
        config['state_dim'] = config['state_dim'] + '{},'.format(K.knowledge_dict['state_dim'][atrb])
        config['action_dim'] = config['action_dim'] + '{},'.format(K.knowledge_dict['action_dim'][atrb])
    config['state_dim'] = config['state_dim'][:-1]
    config['action_dim'] = config['action_dim'][:-1]

    #  network:
    K.network_config(config)
    with open('./config/files/{}.json'.format(config_name), 'w') as f:
        json.dump(config, f)
