# -*- coding:utf-8 -*-
# detailed model:
'''
fish.xml
qpos: 14 d
7:(rootx, rooty, rootz, a, x, y, z)
3:(tail1, tailtwist, tail2)
4:(finright roll, finright pitch, finleft roll, finleft pitch)
ctrl: 5d
(tail, tail_twist, fins_flap, finleft pitch, finright pitch)
sensor: 6d
(torso vel 3d, gygro 3d) 
'''
import os
import math
import numpy as np
from env.mujoco.core import m_core
from env.mujoco.core import attrib

class fish(m_core):

    def __init__(self, attrib_list=[], n_step=1):
        # dynamic file path
        file_path = os.getcwd()
        if attrib_list == []:
            file_name = "{}/env/mujoco/frames/fish.xml".format(file_path)
            m_core.__init__(self, file_name, n_step)
            self.swim_attrib()
            self.reward = self.swim_reward
            self.done = self.swim_done
        elif 'ob' in attrib_list:
            file_name = "{}/env/mujoco/frames/fish_ob.xml".format(file_path)
            m_core.__init__(self, file_name, n_step)
            self.swim_attrib()
            self.ob_attrib()
            self.reward = lambda: self.swim_reward() + self.ob_reward()
            self.done = self.swim_done

    #  attribs configuration
    #  a swim attribute is construct from a attrib and reward structure.
    #  swim towards target
    def swim_attrib(self):
        #  swim structure
        name = "swim"
        #  input part
        state_map = np.array(range(38))
        geom_map = ['target']

        #  output part
        ctrl_map = np.array(range(5))
        ctrl_dim = 5
        a = attrib(name, state_map, geom_map, ctrl_map, ctrl_dim)
        self.attribs.append(a)
    
    def swim_reset(self):
        pass

    def swim_reward(self):
        #  return reward and done
        #  distance from mouth to target
        mouth_to_target = np.linalg.norm(self.geoms['mouth'] - self.geoms['target'])
        threshold = self.geoms_size['target'][0] * 1.25
        return math.exp(-mouth_to_target)

    def swim_done(self):
        mouth_to_target = np.linalg.norm(self.geoms['mouth'] - self.geoms['target'])
        threshold = self.geoms_size['target'][0] * 1.25
        if mouth_to_target <= threshold:
            _done = 1
        else:
            _done = 0
        return  _done

    #  ob attrib:
    def ob_attrib(self):
        name = "ob"
        state_map = np.array(range(38))
        geom_map = ['obstacle']

        #  output part
        ctrl_map = np.array(range(5))
        ctrl_dim = 5
        a = attrib(name, state_map, geom_map, ctrl_map, ctrl_dim)
        self.attribs.append(a)

    def ob_reset(self):
        pass

    def ob_reward(self):
        torso_to_ob = np.linalg.norm(self.geoms['torso'] - self.geoms['obstacle'])
        return 0.1*torso_to_ob


if __name__ == '__main__':
    F = fish(['ob'])
    ctrl_list = np.array([0., 0., 0., 0., 0.])
    print(F.step(ctrl_list))
