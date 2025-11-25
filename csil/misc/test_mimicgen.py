import mimicgen
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
import numpy as np
import robosuite as suite


options = {}
options["env_name"] = "ThreePieceAssembly_D0"
options["robots"] = "Panda"

options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

# initialize the task
env = suite.make(
    **options,
    has_renderer=True,
    has_offscreen_renderer=True,
    ignore_done=True,
    use_camera_obs=False,
    control_freq=20,
)
env.reset()

# Get action limits
low, high = env.action_spec

print(low)
print(high)

# do visualization
for i in range(0):
    action = np.random.uniform(low, high)
    obs, reward, done, _ = env.step(action)
    state_keys = ["robot0_eef_pos", "robot0_eef_quat"]
    state = np.concatenate([[obs[key]] for key in state_keys], axis=1)[0]

    gripper = int(
        (obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]) < 0.05
    )
    state = np.concatenate([state, [gripper]])
    print(gripper, state)
    env.render()


from matplotlib import pyplot as plt


def show_img(img):
    plt.imshow(img, interpolation="nearest")
    plt.show()


# obs["robot0_eye_in_hand"] = env.sim.render(height=224, width=224, camera_name="robot0_eye_in_hand")[::-1]
# obs["agentview"] = env.sim.render(height=224, width=224, camera_name="agentview")[::-1]

show_img(obs["robot0_eye_in_hand"])
show_img(obs["agentview"])


state_keys = ["robot0_eef_pos", "robot0_eef_quat"]
state = np.concatenate([[obs[key]] for key in state_keys], axis=1)[0]

gripper = int((obs["robot0_gripper_qpos"][0] - obs["robot0_gripper_qpos"][1]) < 0.05)
state = np.concatenate([state, [gripper]])

print(state)
