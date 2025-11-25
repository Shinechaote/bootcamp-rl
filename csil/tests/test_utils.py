import pytest
from csil.utils import get_config
from ml_collections import config_dict
import yaml


# def test_get_config():
#     # A few example values to check whether all of the type conversions work when loading from a file
old_config = config_dict.ConfigDict()
old_config.general = config_dict.ConfigDict()
old_config.algorithm = config_dict.ConfigDict()
old_config.pretraining = config_dict.ConfigDict()
old_config.checkpoints = config_dict.ConfigDict()
old_config.environment = config_dict.ConfigDict()
old_config.vla = config_dict.ConfigDict()

old_config.algorithm.anneal_learning_rate = False
old_config.algorithm.buffer_size = 1000000
old_config.algorithm.learning_starts = 10000

old_config.algorithm.critic_warmup_steps = 0
old_config.algorithm.reward_refinement_steps = -1
old_config.algorithm.batch_size = 256
old_config.algorithm.tau = 0.005
old_config.algorithm.gamma = 0.99

old_config.algorithm.target_entropy = "auto"

config_for_wandb = {}
for key in old_config.keys():
    if type(old_config[key]) == config_dict.ConfigDict:
        config_for_wandb[key] = dict(old_config[key])

print(yaml.dump(config_for_wandb))
