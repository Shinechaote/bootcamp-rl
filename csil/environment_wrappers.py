import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
from openpi_client import image_tools
from dataclasses import dataclass
from typing import Optional
import mimicgen


@jax.tree_util.register_dataclass
@dataclass
class Observation:
    state: Optional[jnp.ndarray] = None
    image: Optional[jnp.ndarray] = None
    embeddings: Optional[jnp.ndarray] = None
    vla_action: Optional[jnp.ndarray] = None


@jax.tree_util.register_dataclass
@dataclass
class ObservationStats(Observation):
    pass


@jax.tree_util.register_dataclass
@dataclass
class Sample:
    obs: Observation
    actions: jnp.ndarray
    next_obs: Observation
    next_actions: jnp.ndarray
    rewards: jnp.ndarray
    terminations: jnp.ndarray


class RobomimicStateOnly(gym.Env):
    def __init__(self, env, render_cam="agentview"):
        self.env = env
        self.action_spec = self.env.action_spec
        self.render_cam = render_cam

    def parse_obs(self, obs):
        state_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object-state",
        ]
        processed_obs = np.squeeze(np.concatenate([obs[key] for key in state_keys]))
        obs = Observation(state=processed_obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = reward >= 1.0
        truncated = done
        processed_obs = self.parse_obs(obs)

        return processed_obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        processed_obs = self.parse_obs(obs)

        return processed_obs, None

    def render(self):
        return self.env.sim.render(height=512, width=512, camera_name=self.render_cam)[
            ::-1
        ]


class RobomimicStateOnlySimulatedVLA(gym.Env):
    def __init__(
        self, env, simulated_vla_network, simulated_vla_params, render_cam="agentview"
    ):
        self.env = env
        self.action_spec = self.env.action_spec
        self.render_cam = render_cam
        self.simulated_vla_network = simulated_vla_network
        self.simulated_vla_params = simulated_vla_params

        def get_simulated_vla_action(policy_params: dict, obs):
            dist = self.simulated_vla_network.apply(policy_params, obs)[0]
            action = dist.mode().squeeze()
            action = jnp.expand_dims(action, 0)

            return action

        self.get_simulated_vla_action = jax.jit(get_simulated_vla_action)

    def parse_obs(self, obs):
        state_keys = [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object-state",
        ]
        processed_obs = np.squeeze(np.concatenate([obs[key] for key in state_keys]))
        obs = Observation(state=processed_obs)

        obs.vla_action = self.get_simulated_vla_action(self.simulated_vla_params, obs)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = reward >= 1.0
        truncated = done
        processed_obs = self.parse_obs(obs)

        return processed_obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        processed_obs = self.parse_obs(obs)

        return processed_obs, None

    def render(self):
        return self.env.sim.render(height=512, width=512, camera_name=self.render_cam)[
            ::-1
        ]


class RobomimicNormalizedImage(gym.Env):
    def __init__(
        self,
        env,
        resolution=(84, 84),
        render_cam="agentview",
        eef_state=True,
        object_state=False,
    ):
        self.env = env
        self.include_joint_state = eef_state
        self.include_object_state = object_state
        self.img_height = resolution[0]
        self.img_width = resolution[1]
        self.action_spec = self.env.action_spec
        self.render_cam = render_cam

    def parse_obs(self, obs):
        obs["agentview_image"] = self.env.sim.render(
            height=self.img_height, width=self.img_width, camera_name="agentview"
        )[::-1]
        obs["robot0_eye_in_hand"] = self.env.sim.render(
            height=self.img_height,
            width=self.img_width,
            camera_name="robot0_eye_in_hand",
        )[::-1]

        combined_imgs = np.squeeze(
            np.concatenate([obs["agentview_image"], obs["robot0_eye_in_hand"]], axis=-1)
        )

        combined_imgs = (combined_imgs - 127.5) / 255.0

        state_keys = []
        if self.include_joint_state:
            state_keys.extend(
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            )
        if self.include_object_state:
            state_keys.append("object-state")

        processed_state_obs = np.squeeze(
            np.concatenate([obs[key] for key in state_keys])
        )

        obs = Observation(state=processed_state_obs, image=combined_imgs)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = reward >= 1.0
        truncated = done
        processed_obs = self.parse_obs(obs)

        return processed_obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        processed_obs = self.parse_obs(obs)

        return processed_obs, None

    def render(self):
        return self.env.sim.render(height=512, width=512, camera_name=self.render_cam)[
            ::-1
        ]


class RobomimicEncodedImage(gym.Env):
    def __init__(
        self,
        env,
        encoder_network,
        encoder_params,
        resolution=(84, 84),
        render_cam="agentview",
        eef_state=True,
        object_state=False,
    ):
        self.env = env
        self.encoder_network = encoder_network
        self.encoder_params = encoder_params
        self.include_joint_state = eef_state
        self.include_object_state = object_state
        self.img_height = resolution[0]
        self.img_width = resolution[1]
        self.action_spec = self.env.action_spec
        self.render_cam = render_cam

    def parse_obs(self, obs):
        obs["agentview_image"] = self.env.sim.render(
            height=self.img_height, width=self.img_width, camera_name="agentview"
        )[::-1]
        obs["robot0_eye_in_hand"] = self.env.sim.render(
            height=self.img_height,
            width=self.img_width,
            camera_name="robot0_eye_in_hand",
        )[::-1]

        combined_imgs = np.expand_dims(
            np.squeeze(
                np.concatenate(
                    [obs["agentview_image"], obs["robot0_eye_in_hand"]], axis=-1
                )
            ),
            0,
        )
        combined_imgs = (combined_imgs - 127.5) / 255.0

        encoded_imgs = np.squeeze(
            self.encoder_network.apply(self.encoder_params, combined_imgs)
        )

        state_keys = []
        if self.include_joint_state:
            state_keys.extend(
                ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
            )
        if self.include_object_state:
            state_keys.append("object-state")

        if state_keys != []:
            processed_state_obs = np.squeeze(
                np.concatenate([obs[key] for key in state_keys])
            )
        else:
            processed_state_obs = None

        obs = Observation(state=processed_state_obs, embeddings=encoded_imgs)

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = reward >= 1.0
        truncated = done
        processed_obs = self.parse_obs(obs)

        return processed_obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        processed_obs = self.parse_obs(obs)

        return processed_obs, None

    def render(self):
        return self.env.sim.render(height=512, width=512, camera_name=self.render_cam)[
            ::-1
        ]


class RobomimicPizero(gym.Env):
    def __init__(
        self,
        env,
        pizero_policy,
        prompt,
        normalization_statistics,
        normalization_fn,
        object_state=False,
        debug=False,
        render_cam="agentview",
    ):
        self.env = env
        self.policy = pizero_policy
        self.include_object_state = object_state
        self.task_prompt = prompt
        self.debug = debug
        self.normalization_statistics = normalization_statistics
        self.normalization_fn = normalization_fn
        self.render_cam = render_cam
        self.action_spec = self.env.action_spec

    def parse_obs(self, obs):
        # To speed up debugging
        if self.debug:
            return np.zeros((50, 7)), np.ones((1024,))

        joint_states = get_joint_states(obs)

        obs["agentview"] = self.env.sim.render(
            height=224, width=224, camera_name="agentview"
        )[::-1]
        obs["robot0_eye_in_hand"] = self.env.sim.render(
            height=224, width=224, camera_name="robot0_eye_in_hand"
        )[::-1]

        pizero_observation = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["agentview"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["robot0_eye_in_hand"], 224, 224)
            ),
            "observation/state": joint_states,
            "prompt": self.task_prompt,
        }
        response = self.policy.infer(pizero_observation)
        action_chunk = response["actions"]
        embeddings = response["embeddings"][:1024]

        if self.normalization_statistics is not None:
            normalized_vla_a = self.normalization_fn(
                action_chunk[0], *self.normalization_statistics
            )
        else:
            normalized_vla_a = action_chunk[0]

        if self.include_object_state:
            joint_states = np.squeeze(
                np.concatenate([joint_states, obs["object-state"]])
            )

        obs = Observation(
            state=joint_states, embeddings=embeddings, vla_action=normalized_vla_a
        )

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        terminated = reward >= 1.0
        truncated = done
        processed_obs = self.parse_obs(obs)

        return processed_obs, reward, terminated, truncated, info

    def reset(self):
        obs = self.env.reset()
        processed_obs = self.parse_obs(obs)

        return processed_obs, None

    def render(self):
        return self.env.sim.render(height=512, width=512, camera_name=self.render_cam)[
            ::-1
        ]


def get_joint_states(obs):
    # wrist img needs to be added to obs as well
    state_keys = ["robot0_eef_pos", "robot0_eef_quat"]
    state = np.concatenate([[obs[key]] for key in state_keys], axis=1)[0]

    gripper = int(
        (obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]) < 0.05
    )
    state = np.concatenate([state, [gripper]])
    return state


def mimicgen_env_creation(config, render=False, offcamera_render=False):
    options = {}
    options["env_name"] = config.environment.name
    options["robots"] = "Panda"

    options["controller_configs"] = load_controller_config(
        default_controller="OSC_POSE"
    )

    # initialize the task
    env = suite.make(
        **options,
        has_renderer=render,
        has_offscreen_renderer=offcamera_render,
        ignore_done=False,
        use_camera_obs=False,
        control_freq=20,
    )
    return env


def robomimic_env_creation(config, render, offcamera_render):
    # These are the ones we get
    # kwargs = {'ignore_done': True, 'use_object_obs': True, 'control_freq': 20, 'controller_configs': {'type': 'BASIC', 'body_parts': {'right': {'type': 'OSC_POSE', 'input_max': 1, 'input_min': -1, 'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5], 'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5], 'kp': 150, 'damping': 1, 'impedance_mode': 'fixed', 'kp_limits': [
    #     0, 300], 'damping_limits': [0, 10], 'position_limits': None, 'orientation_limits': None, 'uncouple_pos_ori': True, 'control_delta': True, 'interpolation': None, 'ramp_ratio': 0.2, 'input_ref_frame': 'world', 'gripper': {'type': 'GRIP'}}}}, 'camera_depths': False, 'camera_heights': 84, 'camera_widths': 84, 'lite_physics': False, 'reward_shaping': False}

    env = suite.make(
        env_name=config.environment.name,  # try with other tasks like "Stack" and "Door"
        # robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
        has_renderer=render,
        has_offscreen_renderer=offcamera_render,
        use_camera_obs=False,
        **config.environment.env_meta["env_kwargs"].to_dict(),
    )

    return env


def default_gym_env_creation(config, render=False, offcamera_render=False):
    render_mode = None
    if render:
        render_mode = "human"
    elif offcamera_render:
        render_mode = "rgb_array"

    env = gym.make(config.environment.name, render_mode=render_mode)

    return env
