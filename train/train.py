# -*- coding:utf-8 -*-
#  training structure can be devided into 3 parts:
import numpy as np
import tensorflow as tf
import network.ppo as ppo
import env.mujoco.env_wrapper as ev


class Train:
    
    def __init__(self, config):
        self.config = config
        self.env = ev.env_wrapper(self.config['env_type'])

        self.network = ppo.PolicyRep(self.config) 
        self.network.sess.run(tf.global_variables_initializer())
        self.index = self.network.index

        self.run = self.run_wrapper(self.config['run_type'])
        #  policy update: determined by run_type
        self.p_update = self.p_update_wrapper(self.config['run_type'])
        #  environment update
        self.e_update = self.e_update_wrapper(self.config['e_update_type'])

    def train(self):
        for itr in range(self.config['n_iter']):
            self.e_update()
            results = self.run()    # results in this iter
            self.p_update(results)  # update the policy
            print('iter:{} | reward: {}'.format(itr, results['rewards'].mean())) 

    def test(self):
        results = self.run()
        print('reward: {}'.format(results['rewards'].mean()))
        return results

    #  all kinds of wrappers are listed down:
    #  run_wrapper:
    def run_wrapper(self, type_string):
        if type_string == 'train':
            return self.train_run

    def train_run(self):
        '''
        results has
        {
            'state1':  #  list of list
            [(epoch1_total_steps+step_num2+step_num3),state1_dim
            ]
            'state2':
            [
            *****
            ]
            'q_vals':  #  discounted sum rewards| Q_value
            [(step_num1+step_num2+step_num3)*1,
             ...
            ]
            'old_acts':  #  old_acts of the update attrib
            [(step_num1+step_num2+step+num3)*act_update_dim,
            ...
            ]
            'advantages':
            [(step_num1+step_num2+step_num3)*1,
             ...
            ]
        }
        '''
        results = dict()
        states = self.env.reset()  #  states is in shape of [[step_num1,state1_dim], ...]
        _iter_step = 0
        while(True):
            #  whole iter
            _step = 0
            _reward = []
            _value = []
            while(True):
                for i in range(self.config['attribute_num']):
                    _name = 'state{}'.format(i)
                    if _name not in results.keys():
                        results[_name] = states[_name]
                    else:
                        results[_name] = np.concatenate((results[_name], states[_name]), axis=0)

                As = self.network.choose_action(states)
                value = self.network.predict_value(states)

                #  add old acts
                if 'old_acts' not in results.keys():
                    results['old_acts'] = As[self.index].reshape([1, -1])   #  index is the index for update_name
                else:
                    results['old_acts'] = np.concatenate((results['old_acts'], As[self.index].reshape([1, -1])), axis=0)
            
                #  add value:
                _value.append(value)
                #  step:
                states = self.env.step(As)
                _step += 1

                reward = self.env.reward()
                _reward.append(reward)
                if (_step >= self.config['max_steps_per_epoch']) or self.env.done():
                    break
            #  log rewards:
            #  Q_value the reward and add advantage:
            q_val = self.get_q_val(_reward)
            advantage = self.get_advantage(q_val, _value)
            if 'q_vals' not in results.keys():
                results['rewards'] = np.array(_reward)
                results['q_vals'] = q_val
                results['advantages'] = advantage
            else:
                results['rewards'] = np.concatenate((results['rewards'], _reward), axis=0)
                results['q_vals'] = np.concatenate((results['q_vals'], q_val), axis=0)
                results['advantages'] = np.concatenate((results['advantages'], advantage), axis=0)
            _iter_step += _step
            #  let the step the same length
            if _iter_step >= self.config['min_steps_per_iter']:
                break

        return results

    #  get q_val and advantage:
    def get_q_val(self, rewards):
        q_val = []
        data_len = len(rewards)
        q_val = np.zeros([data_len, 1])
        for i in range(data_len)[::-1]:
            if i == data_len-1:
                q_val[i] = rewards[i]
            else:
                q_val[i] = rewards[i] + self.config['gamma']*q_val[i+1]
        #  normalization:
        q_val = (q_val - q_val.mean())/q_val.std()
        return q_val

    def get_advantage(self, q_values, values):
        values = np.array(values).reshape([-1, 1])
        advantage = q_values - values
        advantage = (advantage - advantage.mean())/advantage.std()
        return advantage

    #  environment update wrapper:
    def e_update_wrapper(self, type_string):
        if type_string == 'curriculum':
            return self.curriculum

    def curriculum(self):
        pass

    #  p_update_wrappers:
    def p_update_wrapper(self, type_string):
        if type_string == 'train':
            return self.ac_update

    def ac_update(self, results):
        self.network.update_old_pi()
        for _ in range(self.config['a_update_steps']):
            self.network.update_a(results)

        for _ in range(self.config['c_update_steps']):
            self.network.update_c(results)
    

if __name__ == '__main__':
    pass 