# -*- coding:utf-8 -*-
#  detailed model:
'''
ball.xml

'''
import os
import math
import numpy as np
from env.mujoco.core import m_core
from env.mujoco.core import attrib

class ball(m_core):
    
    def __init__(self, attrib_list=[], n_step=1):
        #  dynamic file path:
        file_path = os.getcwd()
        if attrib_list == []:
            file_name = "{}/env/mujoco/frames/ball.xml".format(file_path)
            m_core.__init__(self, file_name, n_step)
            #  ball is going to roll:
            self.roll_attrib()
            self.reward = self.roll_reward
            self.done = self.roll_done

    #  attribs configuration:
    def roll_attrib(self):
        #  roll attrib:
        name = "roll"
        state_map = np.array(range(6))
        geom_map = ['target']
        #  output part:
        ctrl_map = np.array(range(2))
        ctrl_dim = 2
        a = attrib(name, state_map, geom_map, ctrl_map, ctrl_dim)
        self.attribs.append(a)
        
    def roll_reset(self):
        pass

    def roll_reward(self):
        #  return reward and done
        #  distance from ball to target
        ball_to_target = np.linalg.norm(self.geoms['ball'] - self.geoms['target'])
        threshold = self.geoms_size['target'][0] * 1.25
        return math.exp(-ball_to_target)

    def roll_done(self):
        ball_to_target = np.linalg.norm(self.geoms['ball'] - self.geoms['target'])
        threshold = self.geoms_size['target'][0] * 1.25
        if ball_to_target <= threshold:
            _done = 1
        else:
            _done = 0
        return  _done


if __name__ == '__main__':
    B = ball([])
    ctrl_list = np.array([0., 0.])
    print(B.step(ctrl_list))
