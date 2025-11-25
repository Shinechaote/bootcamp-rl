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
plt.rcParams.update({"text.usetex": True, "font.family": "roboto", "font.size": 10})


def plot_wandb(runs, metrics, title, vla_eval_values=None, axis=None, set_limits=True):

    # Compute figsize from LaTeX textwidth
    figsize = latex_figsize()
    figsize = (figsize[0], figsize[1])
    if axis is None:
        axis = plt
    for metric in metrics:
        metric_label, x_axis, metric_plot_name = metric
        metrics_data = load_runs(runs, metric)

        # Add BC metric
        minimum_step = np.inf
        for plot_name in metrics_data.keys():
            minimum_step = min(min(list(metrics_data[plot_name].keys())), minimum_step)

        maximum_step = -np.inf
        for plot_name in metrics_data.keys():
            maximum_step = max(max(list(metrics_data[plot_name].keys())), maximum_step)

        if maximum_step == -np.inf:
            print(metrics_data)
            print(metrics_data.keys())
            return

        initial_eval_values = []
        for plot_name in metrics_data.keys():
            if minimum_step in metrics_data[plot_name]:
                initial_eval_values.extend(metrics_data[plot_name][minimum_step])

        if vla_eval_values is not None:
            metrics_data["vla"] = {}
            metrics_data["vla"][minimum_step] = np.expand_dims(
                np.array(vla_eval_values), 1
            )
        else:
            metrics_data["bc"] = {}
            metrics_data["bc"][minimum_step] = np.expand_dims(
                np.array(initial_eval_values), 1
            )

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
        for i, plot_name in enumerate(metrics_mean.keys()):
            print(plot_name)
            if "csil" in plot_name:
                color_index = 0
            elif "sacfd" in plot_name:
                color_index = 1
            else:
                color_index = 2

            linestyle = "solid"
            x_values = np.array(list(metrics_data[plot_name].keys()))
            # Manually convert to scientific notation
            x_values = x_values / 1000000
            mean = np.array(list(metrics_mean[plot_name].values()))
            std = np.array(list(metrics_std[plot_name].values()))
            axis.plot(
                x_values,
                mean,
                label=plot_name,
                color=colors[color_index],
                linestyle=linestyle,
            )
            lower = np.clip(mean - std, 0.0, 1.0)
            upper = np.clip(mean + std, 0.0, 1.0)
            axis.fill_between(
                x_values,
                lower,
                upper,
                alpha=0.3,
                color=colors[color_index],
                linestyle=linestyle,
            )

        if set_limits:
            axis.axhline(0.0, color="grey", linestyle="--", linewidth=1.5, zorder=1)
            axis.axhline(1.0, color="grey", linestyle="--", linewidth=1.5, zorder=1)
            axis.set_xlim(left=0, right=maximum_step / 1000000)
            axis.set_ylim(-0.05, 1.05)

        if metric_plot_name is not None:
            axis.set_ylabel(metric_plot_name)
        if title is not None:
            axis.set_title(title)


metrics = [("evaluation/success_rate", "csil/learning_steps", "Success Rate")]


# for model_type in ["new_model_lora"]:
#     fig, axes = plt.subplots(
#         1,
#         1,
#         figsize=(figsize[0], figsize[1]),
#         sharey="row",
#     )
#     for i, environment in enumerate(
#         ["PickPlaceCan"]
#     ):
#         if model_type == "old_model" and environment == "ThreePieceAssembly_D0":
#             continue

#         filter = {
#             "$and": [
#                 {"tags": "thesis experiment"},
#                 {"tags": "residual"},
#                 {"tags": environment},
#                 {"tags": model_type},
#             ]
#         }
#         # New Model
#         envs = ["Can", "Square", "Assembly"]
#         environment_name = f"{envs[i]}"
#         csil_runs = wandb_api.runs(
#             path="robot-learning-rt2/csil-for-vlas", filters=filter
#         )
#         runs = []
#         for run in csil_runs:
#             runs.append((csil, run.id, "csil", 25000))

#         sacfd_runs = wandb_api.runs(
#             path="robot-learning-rt2/sacfd-for-csil", filters=filter
#         )
#         for run in sacfd_runs:
#             runs.append((sacfd, run.id, "sacfd", 50000 if model_type =="old_model" else 25000))

#         vla_filter = {
#             "$and": [
#                 {"tags": "thesis experiment"},
#                 {"tags": "vla_base"},
#                 {"tags": environment},
#                 {"tags": model_type},
#                 {"$not": {"tags": "4 step action chunk"}},
#             ]
#         }
#         vla_runs = wandb_api.runs(
#             path="robot-learning-rt2/csil-for-vlas", filters=vla_filter
#         )
#         vla_eval_values = []
#         for run in vla_runs:
#             vla_eval = load_vla_base_run((csil, run.id, "vla", 1))
#             if vla_eval is not None:
#                 vla_eval_values.extend(vla_eval)

#         if i == 0:
#             tmp_metrics = [
#                 ("evaluation/success_rate", "csil/learning_steps", "Success Rate")
#             ]
#         else:
#             tmp_metrics = [("evaluation/success_rate", "csil/learning_steps", None)]

#         axes.set_xlabel(r"Steps $(1 \times 10^5)$")
#         axes.set_xticks(np.arange(0, 6, 1.0))
#         axes.set_yticks(np.arange(0, 1.5, 0.5))
#         plot_wandb(
#             runs,
#             tmp_metrics,
#             environment_name,
#             vla_eval_values=vla_eval_values,
#             axis=axes,
#         )


# handles, labels = axes.get_legend_handles_labels()
#     print(labels)
#     fig.legend(
#         handles,
#         labels,
#         loc="lower center",  # Position relative to the whole figure
#         bbox_to_anchor=(0.5, -0.25),  # Move it outside the plot area
#         ncol=len(labels),
#         frameon=False,
#         mode="expand",
#         borderpad=5,
#         fontsize=12,
#     )
#     fig.tight_layout(rect=[0, 0.05, 1, 1])
#     # fig.savefig(f"figs/ppt_success_rate_plot.pdf", dpi=600)
#     fig.savefig(f"figs/ppt_success_rate_plot.png", dpi=600)

fig, axes = plt.subplots(
    1,
    2,
    figsize=(figsize[0], figsize[1] / 2),
    sharey="row",
)
for i, environment in enumerate(["PickPlaceCan", "NutAssemblySquare"]):

    filter = {
        "$and": [
            {"tags": environment},
        ]
    }
    # New Model
    envs = ["Can", "Square", "Assembly"]
    environment_name = f"{envs[i]}"
    csil_runs = wandb_api.runs(
        path="robot-learning-rt2/image_based_csil", filters=filter
    )
    runs = []
    for run in csil_runs:
        runs.append(("robot-learning-rt2/image_based_csil", run.id, "csil", 25000))

    if i == 0:
        tmp_metrics = [
            ("evaluation/success_rate", "csil/learning_steps", "Success Rate")
        ]
    else:
        tmp_metrics = [("evaluation/success_rate", "csil/learning_steps", None)]

    axes[i].set_xlabel(r"Steps $(1 \times 10^6)$")
    axes[i].set_xticks(np.arange(0, 1.1, 0.25))
    axes[i].set_yticks(np.arange(0, 1.5, 0.5))
    plot_wandb(
        runs,
        tmp_metrics,
        environment_name,
        axis=axes[i],
    )


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",  # Position relative to the whole figure
    bbox_to_anchor=(0.5, -0.5),  # Move it outside the plot area
    ncol=len(labels),
    frameon=False,
    mode="expand",
    borderpad=5,
    fontsize=12,
)
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(f"figs/image_based.png", dpi=600)
