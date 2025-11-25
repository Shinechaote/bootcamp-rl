from openpi.training import config as pizero_config_module
from openpi.policies import policy_config
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from openpi_client import image_tools

pizero_config = pizero_config_module.get_config("pi0_mimicgen_robomimic_ph")
checkpoint_dir = os.path.join(
    Path.home(),
    "openpi/checkpoints/pi0_mimicgen_robomimic_ph/pi0_probably_wrong_mimicgen_dataset/9300",
)
vla_type = "general_4_tasks"
print("Loading pizero")
policy = policy_config.create_trained_policy(pizero_config, checkpoint_dir)

ds = h5py.File("/home/scherer/openpi/robomimic_datasets/can/ph/high_dim_v15.hdf5")[
    "data"
]


def get_pizero_action_chunk(obs):
    # wrist img needs to be added to obs as well
    state_keys = ["robot0_eef_pos", "robot0_eef_quat"]
    state = np.concatenate([[obs[key]] for key in state_keys], axis=1)[0]

    gripper = int(
        (obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]) < 0.05
    )
    state = np.concatenate([state, [gripper]])

    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["agentview"], 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(obs["robot0_eye_in_hand"], 224, 224)
        ),
        "observation/state": state,
        "prompt": "lift cube",
    }

    response = policy.infer(observation)
    action_chunk = response["actions"]
    if "embeddings" in response:
        # embeddings = np.squeeze(response["embeddings"])[-1024:]
        embeddings = response["embeddings"]
    else:
        embeddings = None
    return action_chunk, embeddings


all_embeddings = []

cur_episode = ds[f"demo_{0}"]
episode_length = cur_episode["actions"].shape[0]

# Pizero does not support batched inference so I haven't implemented it here and instead just call it on each episode step
for step in range(10):
    agentview_img = cur_episode["obs"]["agentview_image"][step]
    wrist_img = cur_episode["obs"]["robot0_eye_in_hand_image"][step]

    obs = {
        "agentview": agentview_img,
        "robot0_eye_in_hand": wrist_img,
        "robot0_eef_pos": cur_episode["obs"]["robot0_eef_pos"][step],
        "robot0_eef_quat": cur_episode["obs"]["robot0_eef_quat"][step],
        "robot0_gripper_qpos": cur_episode["obs"]["robot0_gripper_qpos"][step],
    }
    action_chunk, embeddings = get_pizero_action_chunk(obs)
    all_embeddings.append(embeddings)

for i in range(1, 10):
    print(np.linalg.norm(all_embeddings[i - 1] - all_embeddings[i]))
