import matplotlib.pyplot as plt
from plotting_utils import (
    latex_figsize,
    load_vla_base_run,
    load_runs,
    get_unique_legend,
)
import wandb
import numpy as np

wandb_api = wandb.Api()

# --- User settings ---
sacfd = "robot-learning-rt2/sacfd-for-csil"  # e.g. "myteam/myproject"
csil = "robot-learning-rt2/csil-for-vlas"

figsize = latex_figsize()
plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 10})


def plot_wandb(
    runs,
    metrics,
    environment_name,
    vla_eval_values=None,
    axis=None,
    title=None,
    set_limits=True,
):

    # Compute figsize from LaTeX textwidth
    figsize = latex_figsize()
    figsize = (figsize[0], figsize[1])
    if axis is None:
        axis = plt
    for i, metric in enumerate(metrics):
        metric_label, x_axis, metric_plot_name = metric
        metrics_data = load_runs(runs, metric)

        # Add BC metric
        minimum_step = np.inf
        for plot_name in metrics_data.keys():
            minimum_step = min(min(list(metrics_data[plot_name].keys())), minimum_step)

        maximum_step = -np.inf
        for plot_name in metrics_data.keys():
            maximum_step = max(max(list(metrics_data[plot_name].keys())), maximum_step)

        initial_eval_values = []
        for plot_name in metrics_data.keys():
            if minimum_step in metrics_data[plot_name]:
                initial_eval_values.extend(metrics_data[plot_name][minimum_step])

        for plot_name in metrics_data.keys():
            if maximum_step not in metrics_data[plot_name]:
                maximum_step_for_plot_name = max(list(metrics_data[plot_name].keys()))
                metrics_data[plot_name][maximum_step] = metrics_data[plot_name][
                    maximum_step_for_plot_name
                ].copy()

        metrics_mean = {}
        metrics_std = {}
        for plot_name in metrics_data.keys():
            metrics_mean[plot_name] = {}
            metrics_std[plot_name] = {}
            for step in metrics_data[plot_name].keys():
                metrics_mean[plot_name][step] = metrics_data[plot_name][step].mean()
                metrics_std[plot_name][step] = metrics_data[plot_name][step].std()

        # --- Plot ---
        colors = (
            np.array([[161, 106, 188], [138, 161, 67], [202, 99, 82], [77, 171, 155]])
            / 255.0
        )
        for plot_name in metrics_mean.keys():
            linestyle = "solid"
            x_values = np.array(list(metrics_data[plot_name].keys()))
            # Manually convert to scientific notation
            x_values = x_values / 10000
            mean = np.array(list(metrics_mean[plot_name].values()))
            axis.plot(
                x_values,
                mean,
                label=metric_plot_name,
                color=colors[i],
                linestyle=linestyle,
            )

            axis.set_xlim(left=0, right=maximum_step // 10000)
            axis.set_ylim(-20, 5)

        if title is None:
            title = environment_name
        axis.set_title(title)


# metrics = [
#     ("pretrainer_policy/expert_entropy", "_step", "Expert Entropy"),
# ]
metrics = [
    ("pretrainer_policy/expert_entropy", "_step", "Expert Entropy"),
    ("evaluation/eval_entropy", "_step", "Online Entropy"),
]

# for model_type in ["new_model_lora", "new_model_full", "new_model", "old_model"]:
for model_type in ["new_model_lora"]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(figsize[0], figsize[1] * 1.2),
        sharex="col",
        sharey="row",
    )
    plt.subplots_adjust(
        left=0.15, right=0.92, top=0.95, bottom=0.2, wspace=0.1, hspace=0.15
    )

    env_names = ["Can", "Square"]
    envs = ["PickPlaceCan", "NutAssemblySquare"]
    for i, environment in enumerate(envs):
        if i == 0:
            environment_name = "Object Pose as Inputs"
        else:
            environment_name = None
        runs = []

        csil_filter = {
            "$and": [
                {"tags": "thesis experiment"},
                {"tags": "entropy investigation"},
                {"tags": "states only"},
                {"tags": environment},
            ]
        }
        csil_runs = wandb_api.runs(
            path="robot-learning-rt2/csil-for-vlas", filters=csil_filter
        )
        # csil_runs = []
        for run in csil_runs:
            runs.append((csil, run.id, "csil", 1))

        axes[i][0].tick_params(axis="y", left=True, labelleft=True)
        axes[i][1].tick_params(axis="y", left=False, labelleft=False)

        axes[i][0].set_ylabel(f"Entropy\n({env_names[i]})")
        if i == len(envs) - 1:
            axes[i][0].set_xlabel(r"Steps $(1 \times 10^4)$")
            axes[i][1].set_xlabel(r"Steps $(1 \times 10^4)$")
            axes[i][0].set_xticks(np.arange(0, 6, 1.0))
            axes[i][1].set_xticks(np.arange(0, 6, 1.0))

        if len(runs) > 0:
            plot_wandb(
                runs, metrics, environment_name, axis=axes[i][0], set_limits=False
            )

        if i == 0:
            environment_name = "VLA Embeddings as Inputs"
        else:
            environment_name = None
        runs = []

        csil_filter = {
            "$and": [
                {"tags": "thesis experiment"},
                {"tags": "entropy investigation"},
                {"tags": model_type},
                {"tags": environment},
            ]
        }
        csil_runs = wandb_api.runs(
            path="robot-learning-rt2/csil-for-vlas", filters=csil_filter
        )
        # csil_runs = []
        for run in csil_runs:
            runs.append((csil, run.id, "csil", 1))

        if len(runs) > 0:
            plot_wandb(
                runs, metrics, environment_name, axis=axes[i][1], set_limits=False
            )

    handles, labels = get_unique_legend(axes)
    fig.legend(
        handles,
        labels,
        loc="lower center",  # Position relative to the whole figure
        bbox_to_anchor=(0.5, -0.18),  # Move it outside the plot area
        ncol=len(labels),
        frameon=False,
        borderpad=5,
        fontsize=12,
    )
    fig.savefig(f"pretraining_entropy_{model_type}.png", dpi=600)

fig, axes = plt.subplots(
    2, 2, figsize=(figsize[0], figsize[1] * 1.2), sharex="col", sharey="row"
)
plt.subplots_adjust(
    left=0.15, right=0.92, top=0.95, bottom=0.2, wspace=0.1, hspace=0.15
)

envs = ["PickPlaceCan", "NutAssemblySquare"]
env_names = ["Can", "Square"]
for i, environment in enumerate(envs):
    if i == 0:
        environment_name = "Frozen Vision Encoder"
    else:
        environment_name = None
    runs = []

    csil_filter = {
        "$and": [
            {"tags": "thesis experiment"},
            {"tags": "entropy investigation"},
            {"tags": "new_model_lora"},
            {"tags": environment},
        ]
    }
    csil_runs = wandb_api.runs(
        path="robot-learning-rt2/csil-for-vlas", filters=csil_filter
    )
    # csil_runs = []
    for run in csil_runs:
        runs.append((csil, run.id, "csil", 1))

    axes[i][0].tick_params(axis="y", left=True, labelleft=True)
    axes[i][1].tick_params(axis="y", left=False, labelleft=False)

    axes[i][0].set_ylabel(f"Entropy\n({env_names[i]})")
    if i == len(envs) - 1:
        axes[i][0].set_xlabel(r"Steps $(1 \times 10^4)$")
        axes[i][1].set_xlabel(r"Steps $(1 \times 10^4)$")
        axes[i][0].set_xticks(np.arange(0, 6, 1.0))
        axes[i][1].set_xticks(np.arange(0, 6, 1.0))

    if len(runs) > 0:
        plot_wandb(runs, metrics, environment_name, axis=axes[i][0], set_limits=False)

    if i == 0:
        environment_name = "Updated Vision Encoder"
    else:
        environment_name = None
    runs = []

    csil_filter = {
        "$and": [
            {"tags": "thesis experiment"},
            {"tags": "entropy investigation"},
            {"tags": "new_model_full"},
            {"tags": environment},
        ]
    }
    csil_runs = wandb_api.runs(
        path="robot-learning-rt2/csil-for-vlas", filters=csil_filter
    )
    # csil_runs = []
    for run in csil_runs:
        runs.append((csil, run.id, "csil", 1))

    if len(runs) > 0:
        plot_wandb(runs, metrics, environment_name, axis=axes[i][1], set_limits=False)

handles, labels = get_unique_legend(axes)
fig.legend(
    handles,
    labels,
    loc="lower center",  # Position relative to the whole figure
    bbox_to_anchor=(0.5, -0.18),  # Move it outside the plot area
    ncol=len(labels),
    frameon=False,
    borderpad=5,
    fontsize=12,
)
fig.savefig(f"pretraining_entropy_compare_full_to_lora.pdf", dpi=600)
