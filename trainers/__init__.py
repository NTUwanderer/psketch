from .curriculum import CurriculumTrainer

def load(config, world, model):
    cls_name = config.trainer.name
    try:
        cls = globals()[cls_name]
        return cls(config, world, model)
    except KeyError:
        raise Exception("No such trainer: {}".format(cls_name))
