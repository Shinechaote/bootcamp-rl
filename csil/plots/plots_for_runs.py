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
            if "csil" in plot_name:
                color_index = 0
            elif "sacfd" in plot_name:
                color_index = 1
            else:
                color_index = 2

            linestyle = "solid"
            x_values = np.array(list(metrics_data[plot_name].keys()))
            # Manually convert to scientific notation
            x_values = x_values / 100000
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
            axis.set_xlim(left=0, right=maximum_step / 100000)
            axis.set_ylim(-0.05, 1.05)

        if metric_plot_name is not None:
            axis.set_ylabel(metric_plot_name)
        if title is not None:
            axis.set_title(title)


metrics = [("evaluation/success_rate", "csil/learning_steps", "Success Rate")]

for model_type in [
    "new_model_full",
    "new_model_lora",
]:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(figsize[0], figsize[1] * 1.7),
        sharex="col",
        sharey="row",
    )

    modalities = ["residual", "action head", "ensemble"]
    modality_names = ["Residual", "Action Head", "Ensemble"]
    for i, environment in enumerate(
        ["PickPlaceCan", "NutAssemblySquare", "ThreePieceAssembly_D0"]
    ):
        for j, modality in enumerate(modalities):
            # New Model
            title = None
            if j == 0:
                envs = ["Can", "Square", "Assembly"]
                axes[j, i].set_title(
                    f"{envs[i]}",
                )

            if j == len(modalities) - 1:
                axes[j][i].set_xlabel(r"Steps $(1 \times 10^5)$")

            if i == 0:
                axes[j][i].tick_params(axis="y", left=True, labelleft=True)
            if i > 0:
                axes[j][i].tick_params(axis="y", left=False, labelleft=False)

            runs = []

            csil_filter = {
                "$and": [
                    {"tags": "thesis experiment"},
                    {"tags": modality},
                    {"tags": model_type},
                    {"tags": environment},
                ]
            }
            csil_runs = wandb_api.runs(
                path="robot-learning-rt2/csil-for-vlas", filters=csil_filter
            )
            for run in csil_runs:
                runs.append((csil, run.id, "csil", 25000))

            vla_filter = {
                "$and": [
                    {"tags": "thesis experiment"},
                    {"tags": "vla_base"},
                    {"tags": model_type},
                    {"tags": environment},
                    {"$not": {"tags": "4 step action chunk"}},
                ]
            }
            vla_runs = wandb_api.runs(
                path="robot-learning-rt2/csil-for-vlas", filters=vla_filter
            )
            vla_eval_values = []
            for run in vla_runs:
                vla_eval = load_vla_base_run((csil, run.id, "vla", 1))
                if vla_eval is not None:
                    vla_eval_values.extend(vla_eval)

            if len(runs) > 0:
                # Only first column should have y label
                if i == 0:
                    tmp_metrics = [
                        (
                            "evaluation/success_rate",
                            "csil/learning_steps",
                            f"Success Rate\n({modality_names[j]})",
                        )
                    ]
                else:
                    tmp_metrics = [
                        ("evaluation/success_rate", "csil/learning_steps", None)
                    ]
                plot_wandb(
                    runs,
                    tmp_metrics,
                    title,
                    vla_eval_values=vla_eval_values,
                    axis=axes[j][i],
                )
                axes[j][i].set_xticks(np.arange(0, 3, 1.0))
                axes[j][i].set_yticks(np.arange(0, 1.5, 0.5))

    handles, labels = get_unique_legend(axes)
    fig.legend(
        handles,
        labels,
        loc="lower center",  # Position relative to the whole figure
        bbox_to_anchor=(0.5, -0.12),  # Move it outside the plot area
        ncol=len(labels),
        frameon=False,
        mode="expand",
        borderpad=5,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(f"figs/modalities_{model_type}.pdf", dpi=600)


for model_type in ["old_model", "new_model_lora", "new_model_full"]:
    if model_type == "old_model":
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(figsize[0], figsize[1] / 2),
            sharey="row",
        )
    else:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(figsize[0], figsize[1] / 2),
            sharey="row",
        )
    for i, environment in enumerate(
        ["PickPlaceCan", "NutAssemblySquare", "ThreePieceAssembly_D0"]
    ):
        if model_type == "old_model" and environment == "ThreePieceAssembly_D0":
            continue

        filter = {
            "$and": [
                {"tags": "thesis experiment"},
                {"tags": "residual"},
                {"tags": environment},
                {"tags": model_type},
            ]
        }
        # New Model
        envs = ["Can", "Square", "Assembly"]
        environment_name = f"{envs[i]}"
        csil_runs = wandb_api.runs(
            path="robot-learning-rt2/csil-for-vlas", filters=filter
        )
        runs = []
        for run in csil_runs:
            runs.append((csil, run.id, "csil", 25000))

        sacfd_runs = wandb_api.runs(
            path="robot-learning-rt2/sacfd-for-csil", filters=filter
        )
        for run in sacfd_runs:
            runs.append(
                (sacfd, run.id, "sacfd", 50000 if model_type == "old_model" else 25000)
            )

        vla_filter = {
            "$and": [
                {"tags": "thesis experiment"},
                {"tags": "vla_base"},
                {"tags": environment},
                {"tags": model_type},
                {"$not": {"tags": "4 step action chunk"}},
            ]
        }
        vla_runs = wandb_api.runs(
            path="robot-learning-rt2/csil-for-vlas", filters=vla_filter
        )
        vla_eval_values = []
        for run in vla_runs:
            vla_eval = load_vla_base_run((csil, run.id, "vla", 1))
            if vla_eval is not None:
                vla_eval_values.extend(vla_eval)

        if i == 0:
            tmp_metrics = [
                ("evaluation/success_rate", "csil/learning_steps", "Success Rate")
            ]
        else:
            tmp_metrics = [("evaluation/success_rate", "csil/learning_steps", None)]

        axes[i].set_xlabel(r"Steps $(1 \times 10^5)$")
        axes[i].set_xticks(np.arange(0, 6, 1.0))
        axes[i].set_yticks(np.arange(0, 1.5, 0.5))
        plot_wandb(
            runs,
            tmp_metrics,
            environment_name,
            vla_eval_values=vla_eval_values,
            axis=axes[i],
        )

    handles, labels = get_unique_legend(axes)
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
    fig.savefig(f"figs/Evaluation_Success_Rate_{model_type}.pdf", dpi=600)
