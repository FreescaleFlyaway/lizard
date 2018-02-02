# -*- coding:utf-8 -*-
import env.mujoco.fish as fish


def env_wrapper(type_string):
    base_env = type_string.split(',')[0]
    return {
    'fish': fish.fish(type_string.split(',')[1:]),         
    }[base_env]

