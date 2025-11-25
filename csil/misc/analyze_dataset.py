import mediapy as media
import h5py
from tqdm import tqdm
from PIL import Image
import numpy as np

ds = h5py.File("/home/scherer/openpi/robomimic_datasets/lift/ph/mid_dim_v141.hdf5")
normal_imgs = []
wrist_imgs = []
# ds = h5py.File("/home/stud_scherer/openpi/robomimic_datasets/mimicgen/three_piece_assembly_d0.hdf5")

print(len(ds["data"].keys()))
for demo in tqdm(range(0, 200, 25)):
    dataset = ds["data"][f"demo_{demo}"]

    # Image.fromarray(dataset["obs"]["agentview_image"][25]).save("normal_img.png")
    # Image.fromarray(dataset["obs"]["robot0_eye_in_hand_image"][25]).save("wrist_img.png")
    # normal_imgs =  np.concatenate([dataset["obs"]["agentview_image"], dataset["obs"]["robot0_eye_in_hand_image"]], axis=-1)[...,3:6]
    image_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    normal_imgs = np.concatenate([dataset["obs"][key] for key in image_keys], axis=-1)
    print(normal_imgs.dtype)

    # wrist_imgs.extend([i for i in dataset["obs"]["robot0_eye_in_hand_image"]])

    media.write_video("normal_imgs.mp4", normal_imgs[..., :-3], fps=20)
    break
# media.write_video("wrist_imgs.mp4", wrist_imgs, fps=20)
