# -*- coding:utf-8 -*-
import json
import train.train as train

if __name__ == '__main__':
    with open('./config/files/ball_train.json', 'r') as f:
        config = json.load(f)
    trainer = train.Train(config)
    trainer.train()
