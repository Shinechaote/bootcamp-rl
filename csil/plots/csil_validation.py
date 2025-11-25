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
deepmind = "robot-learning-rt2/dm-runs"  # e.g. "myteam/myproject"
csil = "robot-learning-rt2/csil-for-vlas"

figsize = latex_figsize()
plt.rcParams.update({"text.usetex": True, "font.family": "roboto", "font.size": 10})


def plot_wandb(
    runs,
    metrics,
    title,
    vla_eval_values=None,
    axis=None,
    set_limits=True,
    deepmind=False,
):

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
            if len(list(metrics_data[plot_name].keys())) == 0:
                continue
            minimum_step = min(min(list(metrics_data[plot_name].keys())), minimum_step)

        maximum_step = -np.inf
        for plot_name in metrics_data.keys():
            if len(list(metrics_data[plot_name].keys())) == 0:
                continue
            maximum_step = max(max(list(metrics_data[plot_name].keys())), maximum_step)

        if minimum_step == np.inf or maximum_step == -np.inf:
            continue

        if maximum_step == -np.inf:
            print(metrics_data)
            print(metrics_data.keys())
            return

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
        for i, plot_name in enumerate(metrics_mean.keys()):
            linestyle = "solid"
            x_values = np.array(list(metrics_data[plot_name].keys()))
            # Manually convert to scientific notation
            x_values = x_values / 1000000
            mean = np.array(list(metrics_mean[plot_name].values()))
            std = np.array(list(metrics_std[plot_name].values()))

            # Sorting
            sorted_x_value_indeces = np.argsort(x_values)
            x_values = x_values[sorted_x_value_indeces]
            mean = mean[sorted_x_value_indeces]
            std = std[sorted_x_value_indeces]

            axis.plot(
                x_values,
                mean,
                label=plot_name,
                color=colors[1 if deepmind else 0],
                linestyle=linestyle,
                zorder=0 if deepmind else 1,
            )
            lower = np.clip(mean - std, 0.0, 1.0)
            upper = np.clip(mean + std, 0.0, 1.0)
            axis.fill_between(
                x_values,
                lower,
                upper,
                alpha=0.3,
                color=colors[1 if deepmind else 0],
                linestyle=linestyle,
                zorder=0 if deepmind else 1,
            )

        if set_limits:
            axis.axhline(0.0, color="grey", linestyle="--", linewidth=1.5, zorder=-1)
            axis.axhline(1.0, color="grey", linestyle="--", linewidth=1.5, zorder=-1)
            axis.set_xlim(left=0, right=1.0)
            axis.set_ylim(-0.05, 1.05)

        if metric_plot_name is not None:
            axis.set_ylabel(metric_plot_name)
        if title is not None:
            axis.set_title(title)


metrics = [("evaluation/success_rate", "csil/learning_steps", "Success Rate")]
environments = ["Lift", "PickPlaceCan", "NutAssemblySquare"]
fig, axes = plt.subplots(
    4,
    3,
    figsize=(figsize[0], figsize[1] * 1.7),
    sharex="col",
    sharey="row",
)
for i, environment in enumerate(environments):
    demonstrations = [25, 50, 100, 200]
    for j, num_demos in enumerate(demonstrations):
        # New Model
        title = None
        if j == 0:
            envs = ["Lift", "Can", "Square"]
            axes[j, i].set_title(
                f"{envs[i]}",
            )

        if j == len(demonstrations) - 1:
            axes[j][i].set_xlabel(r"Steps $(1 \times 10^6)$")

        if i == 0:
            axes[j][i].tick_params(axis="y", left=True, labelleft=True)
        if i > 0:
            axes[j][i].tick_params(axis="y", left=False, labelleft=False)

        runs = []

        csil_filter = {
            "$and": [
                {"tags": "states only"},
                {"tags": f"{num_demos} demos"},
                {"tags": "codebase validation"},
                {"tags": environment},
            ]
        }
        csil_runs = wandb_api.runs(path=csil, filters=csil_filter)
        for run in csil_runs:
            runs.append((csil, run.id, "ours", 50000))

        if len(runs) > 0:
            # Only first column should have y label
            if i == 0:
                tmp_metrics = [
                    (
                        "evaluation/success_rate",
                        "csil/learning_steps",
                        f"Success Rate\n(n = {num_demos})",
                    )
                ]
            else:
                tmp_metrics = [("evaluation/success_rate", "csil/learning_steps", None)]
            plot_wandb(runs, tmp_metrics, title, axis=axes[j][i], deepmind=False)
            axes[j][i].set_xticks(np.arange(0, 1.25, 0.25))
            axes[j][i].set_yticks(np.arange(0, 1.5, 0.5))

        # runs = []
        # dm_filter = {"$and": [{"tags": f"{num_demos} demos"}, {"tags": environment}]}
        # dm_runs = wandb_api.runs(
        #         path=deepmind,
        #         filters=dm_filter
        #         )
        # for run in dm_runs:
        #     runs.append((deepmind, run.id, "original", 25000))

        # if len(runs) > 0:
        #     # Only first column should have y label
        #     if i == 0:
        #         tmp_metrics = [("n_rollout_evaluator/episode_return", "n_rollout_evaluator/actor_steps", f"Success Rate\n(n = {num_demos})")]
        #     else:
        #         tmp_metrics = [("n_rollout_evaluator/episode_return", "n_rollout_evaluator/actor_steps", None)]
        #     plot_wandb(runs, tmp_metrics, title, axis=axes[j][i], deepmind=True)

        #     axes[j][i].set_xticks(np.arange(0, 1.25, 0.25))
        #     axes[j][i].set_yticks(np.arange(0, 1.5, 0.5))

handles, labels = get_unique_legend(axes)
# if len(labels) > 0:
#     fig.legend(handles, labels,
#                loc='lower center', # Position relative to the whole figure
#                bbox_to_anchor=(0.5, -0.12), # Move it outside the plot area
#                ncol=len(labels),
#                frameon=False,
#                mode='expand',
#                borderpad=5,
#                fontsize=12
#                )
fig.tight_layout(rect=[0, 0.05, 1, 1])
fig.savefig(f"figs/codebase_validation_states_only.pdf", dpi=600)
