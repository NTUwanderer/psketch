#!/usr/bin/env python3

from misc.util import Struct
import models
import trainers
import worlds

import logging
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import random
import sys
import tensorflow as tf
import traceback
import yaml

sys.path.append('')

def main():
    config = configure()
    world = worlds.load(config)
    model = models.load(config)
    trainer = trainers.load(config, world, model)
    #trainer.train(model, world)
    trainer.test(model, world)
    with model.session.as_default():
        trainer.transfer(model, world)

def configure():
    # load config
    with open("config.yaml") as config_f:
        config = Struct(**yaml.load(config_f))

    # set up experiment
    config.experiment_dir = os.path.join("experiments/%s" % config.name)
    # assert not os.path.exists(config.experiment_dir), \
    #         "Experiment %s already exists!" % config.experiment_dir
    if not os.path.exists(config.experiment_dir):
        os.mkdir(config.experiment_dir)

    # set up logging
    log_name = os.path.join(config.experiment_dir, "run.log")
    logging.basicConfig(filename=log_name, level=logging.DEBUG,
            format='%(asctime)s %(levelname)-8s %(message)s')
    def handler(type, value, tb):
        logging.exception("Uncaught exception: %s", str(value))
        logging.exception("\n".join(traceback.format_exception(type, value, tb)))
    sys.excepthook = handler

    print ('logging')
    logging.info("BEGIN")
    logging.info(str(config))

    return config

if __name__ == "__main__":
    main()
