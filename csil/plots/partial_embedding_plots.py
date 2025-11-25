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

            axis.set_xlim(left=0, right=maximum_step / 10000)
            axis.set_ylim(-15, 5)

        axis.set_xlabel(r"Steps $(1 \times 10^4)$")
        if title is None:
            title = environment_name
        axis.set_title(title)


metrics = [
    ("pretrainer_policy/expert_entropy", "_step", "Expert Entropy"),
    ("evaluation/eval_entropy", "_step", "Online Entropy"),
]

# for model_type in ["old_model", "new_model", "new_model_lora", "new_model_full"]:
for model_type in ["new_model_lora"]:
    for i, environment in enumerate(["NutAssemblySquare"]):
        fig, axes = plt.subplots(
            1, 3, figsize=(figsize[0], figsize[1] / 2), sharey="row"
        )
        for j, num_embeds in enumerate([1024, 50, 1]):
            if num_embeds == 1:
                environment_name = f"{num_embeds} embedding value"
            else:
                environment_name = f"{num_embeds} embedding values"
            runs = []

            csil_filter = {
                "$and": [
                    {"tags": "thesis experiment"},
                    {"tags": "partial embeddings"},
                    {"tags": f"{num_embeds} embeds"},
                    {"tags": environment},
                    {"tags": model_type},
                ]
            }
            csil_runs = wandb_api.runs(
                path="robot-learning-rt2/csil-for-vlas", filters=csil_filter
            )
            for run in csil_runs:
                runs.append((csil, run.id, "csil", 1))

            if j == 0:
                axes[j].set_ylabel("Entropy")
            axes[j].set_yticks(np.arange(-15, 10, 5.0))
            axes[j].set_xticks(np.arange(0.0, 6.0, 1.0))
            if len(runs) > 0:
                plot_wandb(
                    runs, metrics, environment_name, axis=axes[j], set_limits=False
                )

    handles, labels = get_unique_legend(axes)
    fig.legend(
        handles,
        labels,
        loc="lower center",  # Position relative to the whole figure
        bbox_to_anchor=(0.5, -0.5),  # Move it outside the plot area
        ncol=len(labels),
        frameon=False,
        borderpad=5,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.9])
    fig.suptitle("VLA Embeddings + Object Pose as Inputs")
    fig.savefig(f"Partial_Embeds_entropy_{model_type}.png", dpi=600)
