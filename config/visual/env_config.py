#  Author: Harvey Chang
#  Email: chnme40cs@gmail.com
#  this file is the specific setting for env config
import sys
import config.configure

class EnvConfig(configure.SubConfig):
    def __init__(self, name):
        configure.SubConfig.__init__(self, name)
        #  core config for mujoco
        self.get_knowledge()
        self.get_data()

    def get_data(self):
        self.data['environment'] = ''
        self.data['attribute_num'] = 0
        self.data['base'] = ''

    def get_knowledge(self):
        #  knowledge is used to give choice
        #  the list for base task choice
        self.knowledge['base'] = ['fish']
        self.knowledge['attribute_0'] = ['ob']

        #  state dimension
        self.knowledge['action_dims'] = {'fish':'5'}
        self.knowledge['state_dims'] = {'fish':'38', 'ob':',40'}

    def refresh(self, name=None):
        #  refresh is dynamically generating configuration:
        if name == 'attribute_num':
            try:
                self.data['attribute_num'] = int(self.data['attribute_num'])
                for i in range(self.data['attribute_num']):
                    self.data['attribute_{}'.format(i)] = ''
                    #  add dynamic knowledge
                    self.knowledge['attribute_{}'.format(i)] = self.knowledge['attribute_0']

                num = self.data['attribute_num']

                while ('attribute_{}'.format(num) in self.data):
                    del self.data['attribute_{}'.format(num)]
                    num += 1
            except:
                print('attribute_num is not int')

        elif name == 'base':
            #  when chosen base:
            self.data['action_dim'] = self.knowledge['action_ims'][self.data['base']]
            self.data['state_dim'] = self.knowledge['state_dims'][self.data['base']]

        elif name[:9] == 'attribute':
            #  then dim
            i = 0
            try:
                self.data['dim_list'] = self.knowledge['dim_lists'][self.data['base']]
                while('attribute_{}'.format(i) in self.data.keys()):
                    self.data['dim_list'] += self.knowledge['dim_lists'][self.data['attribute_{}'.format(i)]]
                    i += 1
            except:
                print('Choose base first')
        else:
            pass

        name_list = self.data['base']
        i = 0
        while ('attribute_{}'.format(i) in self.data.keys()):
            name_list += ',{}'.format(self.data['attribute_{}'.format(i)])
            i += 1
        self.data['environment'] = name_list

    def push(self, updata):
        self.data['reset_from_pool'] = False
        #  push
        for name, value in self.data.items():
            updata[name] = value


if __name__ == '__main__':
    EnvConfig('environment')
