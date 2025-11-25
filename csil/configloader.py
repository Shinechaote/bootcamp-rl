from ml_collections import config_dict as ml_config_dict_module
from argparse import ArgumentParser
from csil.loss_configs import BasicSACConfig
from csil.environment_wrappers import (
    mimicgen_env_creation,
    robomimic_env_creation,
    RobomimicStateOnly,
    RobomimicNormalizedImage,
    RobomimicPizero,
)
import os
from csil.networks import periodic_relu_activation, triangle_activation
from csil.evaluation import run_normal_eval, run_vla_eval
from openpi.training import config as pizero_config_module
from openpi.policies import policy_config
import h5py
from robosuite.controllers import load_controller_config
import json
import numpy as np
from csil.utils import linear_normalize, linear_unnormalize, normalize, unnormalize
import wandb
from csil.networks import ResNetTorso
from csil.environment_wrappers import Sample
from datetime import datetime


def verify_config(config, args):
    # assert args.token_selection in ["first", "second", "both"]
    # assert not (config.algorithm.use_vla and config.algorithm.joint_space_obs), "Cant use vla and joint space as observations at the same time"
    # assert int(args.is_simpler_env) + int(args.is_normal_env) + \
    #     int(args.is_robomimic_env) < 2
    pass


def get_config(config_dict, args):
    config = ml_config_dict_module.ConfigDict()
    config.update(config_dict)

    config.general.start_date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    if args.debug:
        config.algorithm.learning_starts = 100
        config.algorithm.evaluation_episodes = 5

    if config.algorithm.stationary_activation_function_str == "per_relu":
        config.algorithm.stationary_activation_function = periodic_relu_activation
    elif config.algorithm.stationary_activation_function_str == "triangle":
        config.algorithm.stationary_activation_function = triangle_activation

    config.algorithm.debug = args.debug
    if not args.debug and config.algorithm.use_vla:
        print("Loading Pizero")
        pizero_config = pizero_config_module.get_config(config.vla.pizero_config_name)

        # Create a trained policy.
        config.vla.policy = policy_config.create_trained_policy(
            pizero_config, config.vla.checkpoint_dir
        )

    config.algorithm.joint_space_obs = (
        not config.algorithm.use_vla and not config.algorithm.image_based_csil
    )

    if config.algorithm.action_normalization_str is None:
        config.algorithm.action_normalization = lambda x, *args: x
        config.algorithm.action_unnormalization = lambda x, *args: x
    elif config.algorithm.action_normalization_str == "linear":
        config.algorithm.action_normalization = linear_normalize
        config.algorithm.action_unnormalization = linear_unnormalize
    elif config.algorithm.action_normalization_str == "mean_and_variance":
        config.algorithm.action_normalization = normalize
        config.algorithm.action_unnormalization = unnormalize

    if config.general.algorithm == "sacfd":
        config.algorithm.residual_action_scaling = np.array([1, 1, 1, 1, 1, 1, 1])
    else:
        config.algorithm.residual_action_scaling = None

    if config.environment.name in ["ThreePieceAssembly_D0", "StackThree_D0"]:
        config.environment.env_type = "mimicgen"
        config.environment.env_creation_fn = mimicgen_env_creation
    elif config.environment.name in ["Lift", "PickPlaceCan", "NutAssemblySquare"]:
        # config.environment.run_eval_fn = run_simpler_env_eval
        config.environment.env_type = "robomimic"
        config.environment.env_creation_fn = robomimic_env_creation
        f = h5py.File(config.pretraining.dataset_filename, "r")
        config.environment.env_meta = json.loads(f["data"].attrs["env_args"])
        del config.environment.env_meta["env_kwargs"]["has_renderer"]
        del config.environment.env_meta["env_kwargs"]["has_offscreen_renderer"]
        del config.environment.env_meta["env_kwargs"]["use_camera_obs"]

        config.environment.env_meta["env_kwargs"]["controller_configs"] = (
            load_controller_config(default_controller="OSC_POSE")
        )

        config.environment.env_meta["camera_names"] = [
            "agentview",
            "robot0_eye_in_hand",
        ]
        f.close()
    else:
        print(f"{config.environment.name} is not yet implemented")
        raise NotImplementedError()

    if config.algorithm.image_based_csil and config.algorithm.freeze_embeddings:
        config.algorithm.image_encoder_network = ResNetTorso()
        assert config.algorithm.embedding_dim is not None

    config.environment.run_eval_fn = run_normal_eval
    config.environment.run_vla_eval_fn = run_vla_eval

    env = config.environment.env_creation_fn(
        config, render=False, offcamera_render=True
    )
    if config.algorithm.is_normal_env:
        # For normal gym envs
        config.environment.sampled_obs = env.observation_space.sample()
        config.environment.sampled_action = env.action_space.sample()
    else:
        # For robomimic gym envs
        if config.algorithm.joint_space_obs:
            env = RobomimicStateOnly(env)
        elif config.algorithm.image_based_csil:
            env = RobomimicNormalizedImage(
                env,
                eef_state=config.algorithm.concat_joint_state,
                object_state=config.algorithm.concat_object_state,
            )
        elif config.algorithm.use_vla:
            env = RobomimicPizero(
                env,
                config.vla.policy,
                config.environment.task_prompt,
                (np.ones((7,)), -np.ones((7,)), np.zeros((7,)), np.ones((7,))),
                config.algorithm.action_normalization,
                object_state=config.algorithm.concat_object_state,
            )
        else:
            raise NotImplementedError(
                "The selected environment wrapper option is not defined"
            )

        obs, _ = env.reset()
        config.environment.sampled_obs = obs
        config.environment.sampled_action = env.action_spec[0]

    config.environment.num_actions = np.prod(config.environment.sampled_action.shape)
    if config.general.algorithm in ["sacfd", "sac"]:
        config.algorithm.target_entropy = -config.environment.num_actions
    else:
        config.algorithm.target_entropy = 0.0

    if config.algorithm.is_simpler_env or config.algorithm.is_normal_env:
        config.environment.as_low = env.action_space.low
        config.environment.as_high = env.action_space.high
    else:
        config.environment.as_low = env.action_spec[0]
        config.environment.as_high = env.action_spec[1]

    config.general.example_item = Sample(
        config.environment.sampled_obs,
        np.zeros((7,)),
        config.environment.sampled_obs,
        np.zeros((7,)),
        np.zeros((1,)),
        np.zeros((1,)),
    )


    config.checkpoints.reload_checkpoint = config.checkpoints.checkpoint_dir is not None

    if config.general.algorithm in ["sac", "sacfd"]:
        config.algorithm.loss_config = BasicSACConfig(config.algorithm.gamma)

    return config


def create_run_dir(config, cur_date):
    if wandb.run is not None:
        wandb_run_id = wandb.run.id
    else:
        wandb_run_id = "None"
    config.general.run_dir = os.path.join(
        config.general.root_dir_path,
        "runs",
        f"{config.general.experiment_name} - {config.environment.name} - {wandb_run_id} -{cur_date.replace('/', '_')}",
    )
    print(f"Creating run dir {config.general.run_dir}")

    os.makedirs(config.general.run_dir, exist_ok=True)
    os.makedirs(os.path.join(config.general.run_dir, "rendered_video"), exist_ok=True)

    if config.checkpoints.checkpoint_dir is None:
        config.checkpoints.checkpoint_dir = os.path.join(
            config.general.run_dir, "checkpoints"
        )


def get_args_parsed():
    parser = ArgumentParser()
    # Below are only used for the rl simulation using the replay buffer
    parser.add_argument("--config_file", default="")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    return args
