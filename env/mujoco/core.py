# -*- coding:utf-8 -*-
# this file defines the core function of mojoco model:

import cv2
import numpy as np
import mujoco_py as mj
from collections import OrderedDict as OD


class m_core:
    
    def __init__(self, filename, n_step=1):
        self.attribs = []
        self.geoms = dict()
        self.model = mj.load_model_from_path(filename)
        self.sim = mj.MjSim(self.model, nsubsteps=n_step)
        # get geom_dict
        self.init_state = self.state()
        #  normalize
        self.geoms = dynamic_dict(self.model.geom_names, self.sim.data.geom_xpos)
        self.geoms_size = dynamic_dict(self.model.geom_names, self.model.geom_size)

        self.frange()
        self.xrange()

    # normalization
    def frange(self):
        ranges = self.model.actuator_ctrlrange
        self.fscaler = []
        for r in ranges:
            scale = (r[1] - r[0])/2.0
            offset = (r[1] + r[0])/2.0
            self.fscaler.append([scale, offset])

    def xrange(self):
        ranges = self.model.jnt_range
        self.xscaler = []
        for r in ranges:
            scale = (r[1] - r[0])/2.0
            offset = (r[1] + r[0])/2.0
            self.xscaler.append([scale, offset])

    #  get information
    def state(self):
        return self.sim.get_state()

    def qpos(self):
        return self.sim.get_state().qpos

    def qvel(self):
        return self.sim.get_state().qvel

    def sensor(self):
        return self.sim.data.sensordata

    def time(self):
        return self.sim.get_state().time

    def ctrl(self):
        return self.sim.data.ctrl

    def output(self):
        #  concat the [qpos, qvel, sensor, ctrl]
        output = np.concatenate((self.qpos(), self.qvel(), self.sensor(), self.ctrl()))
        return output
  
    def picture(self, size, camera_name):
        img = self.sim.render(width=size[0], height=size[1], camera_name = camera_name)
        return img

    #  set state
    def run(self):
        self.sim.step()

    def set(self, state):
        self.sim.set_state(state)

    def reset(self):
        self.set(self.init_state)
        return self.output_dict()

    def pure_step(self, forces):

        forces = np.clip(forces, -1.0, 1.0)
        for i, s in enumerate(self.fscaler):
            self.sim.data.ctrl[i] = forces[i]*s[0] + s[1]
        self.run()

    #  vedio: wait
    #  attribs manipulation:
    def run_from_list(self, ctrl_list):
        final_ctrl = np.zeros(self.attribs[0].ctrl_dim)
        for ctrl_val, atrb in zip(ctrl_list, self.attribs):
            final_ctrl += atrb.ctrl_mapping(ctrl_val)
        self.pure_step(final_ctrl)

    def output_dict(self):
        #  output dict
        outputs = dict()
        output = self.output()
        for i, atrb in enumerate(self.attribs):
            state_array = atrb.state_mapping(output)
            for name in atrb.geom_map:
                state_array = np.concatenate((state_array, self.geoms[name]), axis=0)
            outputs['state{}'.format(i)] = state_array.reshape([1, -1])
        return outputs

    def step(self, ctrl_list):
        self.run_from_list(ctrl_list)
        return self.output_dict()
    

class attrib:
    
    def __init__(self, name, state_map, geom_map, ctrl_map, ctrl_dim):
        self.name = name
        self.state_map = state_map
        self.geom_map = geom_map
        self.ctrl_map = ctrl_map
        self.ctrl_dim = ctrl_dim

    def ctrl_mapping(self, ctrl_value):
        ctrl_val = np.zeros(self.ctrl_dim)
        ctrl_val[self.ctrl_map] = ctrl_value
        return ctrl_val

    def state_mapping(self, state_value):
        state_val = state_value[self.state_map]
        return state_val


class dynamic_dict:
    
    def __init__(self, name_list, value_list):
       self.name_list = name_list
       self.value_list = value_list

    def __getitem__(self, name):
        _index = self.name_list.index(name)
        return self.value_list[_index]

    def __setitem__(self, name, value):
        _index = self.name_list.index(name)
        self.value_list[_index] = value

    def names(self):
        return self.name_list

    def values(self):
        return self.value_list


if __name__ == '__main__':
    filename = './frames/fish.xml'
    mycore = m_core(filename)
    print(mycore.state())
    print(mycore.qpos().shape)
    print(mycore.qvel().shape)
    print(mycore.sensor().shape)
    print(mycore.ctrl().shape)
    print(mycore.output().shape)
