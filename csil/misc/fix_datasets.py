import numpy as np
from utils import linear_unnormalize, calculate_dataset_statistics
import h5py
from tqdm import tqdm


ds_path = "../ThreePieceAssembly_D0_lora_scaled_up_threepieceassembly_general_4_tasks_200_dataset_with_pizero_actions_double.npy"
hdf_path = (
    "/home/scherer/openpi/robomimic_datasets/mimicgen/three_piece_assembly_d0.hdf5"
)

loaded_dataset = np.load(ds_path, allow_pickle=True).item()

s = loaded_dataset["s"]
a = loaded_dataset["a"]
next_s = loaded_dataset["next_s"]
next_a = loaded_dataset["next_a"]
rewards = loaded_dataset["rewards"]
terminations = loaded_dataset["terminations"]
discounts = loaded_dataset["discounts"]
vla_a = loaded_dataset["vla_a"]
next_state_vla_a = loaded_dataset["next_state_vla_a"]
joint_states = loaded_dataset["joint_states"]
next_joint_states = loaded_dataset["next_joint_states"]

ds = h5py.File(hdf_path)["data"]
object_states = np.zeros((s.shape[0], ds["demo_0"]["obs"]["object"].shape[1]))
num_samples = 0

print(object_states.shape)

for i in tqdm(range(200)):
    cur_episode = ds[f"demo_{i}"]
    episode_num_samples = cur_episode["obs"]["object"].shape[0]
    object_states[num_samples : num_samples + episode_num_samples] = cur_episode["obs"][
        "object"
    ]
    num_samples += episode_num_samples

next_object_states = np.concatenate(
    [object_states[1:], [np.zeros_like(object_states[0])]], axis=0
)


fields = {
    "s": s,
    "a": a,
    "next_s": next_s,
    "next_a": next_a,
    "rewards": rewards,
    "terminations": terminations,
    "discounts": discounts,
    "vla_a": vla_a,
    "next_state_vla_a": next_state_vla_a,
    "joint_states": joint_states,
    "next_joint_states": next_joint_states,
    "object_states": object_states,
    "next_object_states": next_object_states,
}


np.save(ds_path, fields, allow_pickle=True)
