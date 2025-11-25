import csil.sac as sac
from csil.networks import (
    StationaryHeteroskedasticNormalTanhDistribution,
    DoubleMLP,
    CriticMLP,
)
import wandb
import orbax.checkpoint as ocp
from typing import Callable
import jax.numpy as jnp
import jax
import numpy as np
import tensorflow_probability.substrates.jax as tfp

# import mani_skill2_real2sim.envs
from csil.regression import pretrain_policy, critic_pretraining
from csil.utils import disable_tf_from_using_gpus, expand_obs_dim
from csil.configloader import get_config, get_args_parsed, create_run_dir
from ml_collections import config_dict as ml_config_dict_module
import yaml
from csil.dataloader import load_data, encode_images, calculate_simulated_vla_actions
from csil.environment_wrappers import RobomimicEncodedImage, Sample

tfd = tfp.distributions

np.set_printoptions(precision=3, suppress=True)
disable_tf_from_using_gpus()


def prior_policy_log_likelihood(sampled_action) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """We assume a uniform hyper prior in a [-1, 1] action space."""
    num_actions = np.prod(sampled_action.shape, dtype=int)

    def prior(x, y):
        return -num_actions * jnp.log(2.0)

    return jax.vmap(jax.jit(prior))


def create_wandb_run(config, args):
    tags = [config.environment.name]
    if config.algorithm.image_based_csil:
        tags.append("image based")

    if config.algorithm.use_vla_action_for_hetstat:
        tags.append("residual")
    else:
        tags.append("action head")

    run_name = f"{config.general.experiment_name} - {config.environment.name} - {config.pretraining.num_demonstrations} Demos - {config.general.start_date}"

    if config.general.wandb_project_type == "final":
        project_suffix = "-final"
    elif config.general.wandb_project_type == "sharing":
        project_suffix = "-sharing"
    elif config.general.wandb_project_type == "debug":
        project_suffix = "-debug"
    else:
        raise NotImplementedError(
            f"There is no such wandb project type as {config.general.wandb_project_type}"
        )

    if config.algorithm.image_based_csil:
        wandb.init(
            entity="robot-learning-rt2",
            project=f"image_based_csil{project_suffix}",
            name=run_name,
            tags=tags,
        )
    elif config.general.algorithm in ["sacfd", "sac"]:
        wandb.init(
            entity="robot-learning-rt2",
            project=f"sacfd-for-csil{project_suffix}",
            name=run_name,
            tags=tags,
        )
    else:
        wandb.init(
            entity="robot-learning-rt2",
            project=f"csil-for-vlas{project_suffix}",
            name=run_name,
            tags=tags,
        )

    wandb.run.log_code(".")
    config_for_wandb = {}
    for key in config.keys():
        if type(config[key]) == ml_config_dict_module.ConfigDict:
            # To prevent wandb from trying to upload image obs
            tmp_config = dict(config[key]).copy()
            if key == "environment":
                del tmp_config["sampled_obs"]
                del tmp_config["sampled_action"]
            elif key == "general":
                del tmp_config["example_item"]
            config_for_wandb[key] = tmp_config

    wandb.config.update(config_for_wandb)


def create_networks(config):
    if not config.algorithm.use_vla and (
        config.algorithm.use_vla_action_for_hetstat
        or config.algorithm.use_linear_residual_combination
    ):
        # TODO need to check this, think it doesn't work for image based rn
        simulated_vla_network = StationaryHeteroskedasticNormalTanhDistribution(
            config.environment.sampled_action.shape[-1],
            layers=[nr_hidden_units, nr_hidden_units, bottleneck_size, nr_hidden_units],
            feature_dimension=nr_hidden_units,
            use_resnet=config.algorithm.image_based_csil,
            stationary_activation=config.algorithm.stationary_activation_function,
            is_residual=False,
            use_embeddings=config.algorithm.use_vla
            or config.algorithm.image_based_csil,
            embedding_dim=config.algorithm.embedding_dim,
        )
    else:
        simulated_vla_network = None

    policy_network = StationaryHeteroskedasticNormalTanhDistribution(
        config.environment.sampled_action.shape[-1],
        layers=[nr_hidden_units, nr_hidden_units, bottleneck_size],
        feature_dimension=nr_hidden_units,
        use_resnet=config.algorithm.image_based_csil,
        stationary_activation=config.algorithm.stationary_activation_function,
        is_residual=config.algorithm.use_vla_action_for_hetstat
        and not config.algorithm.use_linear_residual_combination,
        residual_action_dataset_statistics=config.algorithm.residual_action_dataset_statistics,
        residual_action_scaling=config.algorithm.residual_action_scaling,
        use_embeddings=config.algorithm.use_vla or config.algorithm.image_based_csil,
        embedding_dim=config.algorithm.embedding_dim,
    )
    if config.general.algorithm in ["sac", "sacfd"]:
        print("Using double MLP Critic", nr_hidden_units)
        critic_network = DoubleMLP(
            [nr_hidden_units, nr_hidden_units, 1],
            use_resnet=config.algorithm.image_based_csil
            and not config.algorithm.freeze_embeddings,
            use_embeddings=config.algorithm.use_vla
            or config.algorithm.freeze_embeddings,
            embedding_dim=config.algorithm.embedding_dim,
        )
    else:
        print("Using single MLP Critic", nr_hidden_units)
        critic_network = CriticMLP(
            [nr_hidden_units, nr_hidden_units, 1],
            use_resnet=config.algorithm.image_based_csil
            and not config.algorithm.freeze_embeddings,
            use_embeddings=config.algorithm.use_vla
            or config.algorithm.freeze_embeddings,
            embedding_dim=config.algorithm.embedding_dim,
            # freeze_resnet=True
        )

    return policy_network, critic_network, simulated_vla_network


def create_networks_for_training_embeds_during_rl(config):
    if config.algorithm.freeze_embeddings:
        policy_network = StationaryHeteroskedasticNormalTanhDistribution(
            config.environment.sampled_action.shape[-1],
            layers=[nr_hidden_units, nr_hidden_units, bottleneck_size],
            feature_dimension=nr_hidden_units,
            use_resnet=False,
            stationary_activation=config.algorithm.stationary_activation_function,
            is_residual=config.algorithm.use_vla_action_for_hetstat
            and not config.algorithm.use_linear_residual_combination,
            residual_action_dataset_statistics=config.algorithm.residual_action_dataset_statistics,
            residual_action_scaling=config.algorithm.residual_action_scaling,
            use_embeddings=True,
            embedding_dim=config.algorithm.embedding_dim,
        )

    else:
        policy_network = StationaryHeteroskedasticNormalTanhDistribution(
            config.environment.sampled_action.shape[-1],
            layers=[nr_hidden_units, nr_hidden_units, bottleneck_size],
            feature_dimension=nr_hidden_units,
            stationary_activation=config.algorithm.stationary_activation_function,
            is_residual=config.algorithm.use_vla_action_for_hetstat
            and not config.algorithm.use_linear_residual_combination,
            residual_action_dataset_statistics=config.algorithm.residual_action_dataset_statistics,
            residual_action_scaling=config.algorithm.residual_action_scaling,
            use_resnet=True,
            use_embeddings=True,
            embedding_dim=config.algorithm.embedding_dim,
            freeze_resnet=True,
        )

    return policy_network


if __name__ == "__main__":
    args = get_args_parsed()

    disable_tf_from_using_gpus()

    with open(args.config_file, "r") as config_file:
        config_file_content = yaml.safe_load(config_file)

    config = get_config(config_file_content, args)
    if not args.debug and config.general.start_wandb_run:
        create_wandb_run(config, args)

    # Only create a run dir if we record rollouts or save checkpoints
    if (
        config.algorithm.save_training_recordings
        or config.algorithm.record_before_after
        or config.algorithm.record_eval_episodes
        or config.checkpoints.saving_frequency > 0
    ):
        create_run_dir(config, config.general.start_date)
        # Define the checkpoint manager
        options = ocp.CheckpointManagerOptions()
        config.checkpoints.manager = ocp.CheckpointManager(
            directory=config.checkpoints.checkpoint_dir, options=options
        )

    if config.general.algorithm != "sac":
        demo_buffer_state = load_data(config, args)
    else:
        config.algorithm.action_dataset_statistics = None
        config.algorithm.state_dataset_statistics = None
        config.algorithm.residual_action_dataset_statistics = None
        config.algorithm.residual_action_scaling = None


    nr_hidden_units = config.algorithm.nr_hidden_units
    bottleneck_size = config.algorithm.bottleneck_size

    # For sanity checking whether residual code works with just state-based csil
    simulated_vla_network = None
    simulated_vla_params = None

    policy_network, critic_network, simulated_vla_network = (
        create_networks(config)
    )

    if simulated_vla_network is not None:
        simulated_vla_params = simulated_vla_network.init(
            jax.random.PRNGKey(0), expand_obs_dim(config.environment.sampled_obs)
        )

    policy_params = policy_network.init(
        jax.random.PRNGKey(0), expand_obs_dim(config.environment.sampled_obs)
    )

    if config.checkpoints.reload_checkpoint:
        print(f"Reloading checkpoint from {config.checkpoints.checkpoint_dir}")
        checkpoint_step = config.checkpoints.manager.latest_step()
        assert checkpoint_step is not None
        print(f"Loading step {checkpoint_step}")
        simulated_vla_params = config.checkpoints.manager.restore(step=checkpoint_step)[
            "params"
        ]

    # print(get_statistics(demo_buffer_state.buffer.obs.vla_action))
    # visualize_action_dims(demo_buffer_state.buffer.obs.vla_action)
    # visualize_action_dims(demo_buffer_state.buffer.actions)

    # from csil.evaluation import run_vla_eval
    # run_vla_eval(config, render_video=True)

    if config.general.algorithm not in ["sac", "sacfd"]:
        # If we are not using a VLA but want to use a residual or an ensemble, we will simulate a vla via a second policy network
        if (
            not config.algorithm.use_vla
            and (
                config.algorithm.use_vla_action_for_hetstat
                or config.algorithm.use_linear_residual_combination
            )
            and not config.checkpoints.reload_checkpoint
        ):
            simulated_vla_params, _ = pretrain_policy(
                simulated_vla_network,
                simulated_vla_params,
                demo_buffer_state,
                config,
            )
            demo_buffer_state = calculate_simulated_vla_actions(
                simulated_vla_network, simulated_vla_params, demo_buffer_state, config
            )
            policy_params, _ = pretrain_policy(
                policy_network, policy_params, demo_buffer_state, config
            )

            config.vla_simulated_vla_network = simulated_vla_network
            config.vla_simulated_vla_params = simulated_vla_params
        else:
            last_policy_params, best_policy_params = pretrain_policy(
                policy_network, policy_params, demo_buffer_state, config
            )

            policy_params = last_policy_params
            # policy_params = best_policy_params

        if config.pretraining.save_checkpoint:
            config.checkpoints.manager.save(
                0,
                args=ocp.args.Composite(params=ocp.args.StandardSave(policy_params)),
            )
            config.checkpoints.manager.wait_until_finished()

        encoder_params = None
        if config.algorithm.image_based_csil:
            policy_network = (
                create_networks_for_training_embeds_during_rl(config)
            )

            if config.algorithm.freeze_embeddings:
                encoder_params = {}
                encoder_params["params"] = jax.tree_util.tree_map(
                    lambda x: x.copy(), policy_params["params"]["ResNetTorso_0"]
                )
                env = config.environment.env_creation_fn(
                    config, render=False, offcamera_render=True
                )
                env = RobomimicEncodedImage(
                    env,
                    config.algorithm.image_encoder_network,
                    encoder_params,
                    eef_state=config.algorithm.concat_joint_state,
                    object_state=config.algorithm.concat_object_state,
                )

                demo_buffer_state = encode_images(
                    encoder_params, demo_buffer_state, config
                )

                # Update Sampled Obs and RewardNetwork because we will now use the image embeddings and not the images themselves
                obs, _ = env.reset()
                config.environment.sampled_obs = obs
                config.environment.sampled_action = env.action_spec[0]
                config.general.example_item = Sample(
                    config.environment.sampled_obs,
                    config.environment.sampled_action,
                    config.environment.sampled_obs,
                    config.environment.sampled_action,
                    np.zeros((1,)),
                    np.zeros((1,)),
                )

    bc_policy_params = jax.tree_util.tree_map(lambda x: x.copy(), policy_params)

    critic_params = critic_network.init(
        jax.random.PRNGKey(0),
        expand_obs_dim(config.environment.sampled_obs),
        np.expand_dims(config.environment.sampled_action, 0),
    )

    if config.algorithm.image_based_csil and not config.algorithm.freeze_embeddings:
        critic_params["params"]["ResNetTorso_0"] = jax.tree.map(
            lambda x: x.copy(), policy_params["params"]["ResNetTorso_0"]
        )


    if config.general.algorithm != "sac":
        critic_params = critic_pretraining(
            critic_network,
            critic_params,
            demo_buffer_state,
            config,
        )

    if config.general.algorithm == "sac":
        sac_module = sac.SAC(
            config, policy_network, policy_params, critic_network, critic_params
        )
        (
            policy_params,
            max_eval_params,
            policy_target_params,
            critic_params,
            nr_episodes,
        ) = sac_module.train()
    elif config.general.algorithm == "sacfd":
        sac_module = sac.SAC(
            config, policy_network, policy_params, critic_network, critic_params
        )
        (
            policy_params,
            max_eval_params,
            policy_target_params,
            critic_params,
            nr_episodes,
        ) = sac_module.train(demo_buffer_state)
    else:
        sac_module = sac.SAC(
            config,
            policy_network,
            policy_params,
            critic_network,
            critic_params,
            image_encoder_params=encoder_params,
            simulated_vla_network=simulated_vla_network,
            simulated_vla_params=simulated_vla_params,
        )
        (
            policy_params,
            max_eval_params,
            policy_target_params,
            critic_params,
            nr_episodes,
        ) = sac_module.train(demo_buffer_state)

    if config.pretraining.save_checkpoint:
        config.checkpoints.manager.save(
            config.algorithm.total_timesteps,
            args=ocp.args.Composite(params=ocp.args.StandardSave(policy_params)),
        )
        config.checkpoints.manager.wait_until_finished()

    bc_success_rates = []
    final_success_rates = []
    max_eval_success_rates = []
    for i in range(1):
        if config.general.algorithm not in ["sac", "sacfd"]:
            bc_success_rates.append(
                config.environment.run_eval_fn(
                    config,
                    policy_network,
                    bc_policy_params,
                    render_video=config.algorithm.record_before_after,
                    num_eval_episodes=config.algorithm.final_evaluation_episodes,
                    video_name_prefix="before_rl_",
                    policy_target_params=bc_policy_params,
                    critic_network=critic_network,
                    critic_params=critic_params,
                    learning_steps=-1,
                    simulated_vla_network=simulated_vla_network,
                    simulated_vla_params=simulated_vla_params,
                    final_eval=True,
                )
            )
        max_eval_success_rates.append(
            config.environment.run_eval_fn(
                config,
                policy_network,
                max_eval_params,
                render_video=config.algorithm.record_before_after,
                num_eval_episodes=config.algorithm.final_evaluation_episodes,
                video_name_prefix="max_eval_rl_",
                learning_steps=int(config.algorithm.total_timesteps / 2),
                simulated_vla_network=simulated_vla_network,
                simulated_vla_params=simulated_vla_params,
                final_eval=True,
            )
        )
        final_success_rates.append(
            config.environment.run_eval_fn(
                config,
                policy_network,
                policy_params,
                render_video=config.algorithm.record_before_after,
                num_eval_episodes=config.algorithm.final_evaluation_episodes,
                video_name_prefix="after_rl_",
                policy_target_params=policy_target_params,
                critic_network=critic_network,
                critic_params=critic_params,
                learning_steps=config.algorithm.total_timesteps + 1,
                simulated_vla_network=simulated_vla_network,
                simulated_vla_params=simulated_vla_params,
                final_eval=True,
            )
        )

    print(bc_success_rates)
    print(max_eval_success_rates)
    print(final_success_rates)

    if wandb.run is not None:
        if len(bc_success_rates) > 0:
            wandb.log(
                {
                    "evaluation/average_final_success_rate": np.array(
                        bc_success_rates
                    ).mean()
                }
            )
        wandb.log(
            {
                "evaluation/average_final_success_rate": np.array(
                    max_eval_success_rates
                ).mean()
            }
        )
        wandb.log(
            {
                "evaluation/average_final_success_rate": np.array(
                    final_success_rates
                ).mean()
            }
        )
