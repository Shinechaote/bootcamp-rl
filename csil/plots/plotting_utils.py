import wandb
import matplotlib.pyplot as plt
import numpy as np

wandb_api = wandb.Api()


def latex_figsize(
    textwidth_pt=418.25555,
    textwidth_in=None,
    fraction=1.0,
    ratio=0.618,  # height = width * ratio (golden-ish ratio)
    subplots=(1, 1),
):

    if textwidth_in is None:
        # TeX pt: 1 in = 72.27 pt (TeX points)
        textwidth_in = float(textwidth_pt) / 72.27

    fig_width = textwidth_in * float(fraction)
    # If multiple subplots horizontally, you might want to scale width:
    # (we assume subplots are arranged vertically/horizontally as usual)
    nrows, ncols = subplots
    # If many columns, divide width across columns (user can also request wide fraction)
    fig_width = fig_width
    fig_height = fig_width * float(ratio) * (nrows / ncols)
    return fig_width, fig_height


def load_runs(runs, metric):

    metric_label, x_axis, metric_plot_name = metric

    # Structure of metrics data is metrics_data[name of run in plot] -> a dictionary where the keys are steps and the values are arrays with all the values we have for the specific step
    metrics_data = {}
    for run in runs:
        project_url, run_id, plot_name, eval_step_size = run

        print(f"{project_url}/{run_id}")
        wandb_run = wandb_api.run(f"{project_url}/{run_id}")
        history = wandb_run.history(keys=[metric_label], x_axis=x_axis)

        if metric_label not in history:
            continue

        if plot_name not in metrics_data:
            metrics_data[plot_name] = {}

        for step, value in history[metric_label].items():
            x_axis_value = history[x_axis][step]
            # We need to floor the x_axis_value to the nearest step size because the exact step at which eval is done can vary, the way I implemented it in sac
            x_axis_value = np.maximum(
                (x_axis_value // eval_step_size) * eval_step_size, 0
            )

            if x_axis_value not in metrics_data[plot_name]:
                metrics_data[plot_name][x_axis_value] = []

            metrics_data[plot_name][x_axis_value].append(value)

    # Convert to np array for ease of use
    for run in runs:
        project_url, run_id, plot_name, eval_step_size = run
        if plot_name in metrics_data:
            for step in metrics_data[plot_name].keys():
                metrics_data[plot_name][step] = np.array(metrics_data[plot_name][step])
        else:
            print(
                f"WARNING: No data for metric {metric_label} in {plot_name} ({run_id})"
            )

    return metrics_data


def load_vla_base_run(run):
    wandb_api = wandb.Api()

    project_url, run_id, plot_name, eval_step_size = run
    print(f"VLA: {project_url}/{run_id}")
    wandb_run = wandb_api.run(f"{project_url}/{run_id}")

    vla_eval_values_dict = wandb_run.history(keys=["evaluation/vla_eval_success_rate"])
    if "evaluation/vla_eval_success_rate" in vla_eval_values_dict:
        return vla_eval_values_dict["evaluation/vla_eval_success_rate"]
    else:
        return None


def get_unique_legend(axes):
    handles = []
    labels = []

    for ax in axes.flat:
        ax_handles, ax_labels = ax.get_legend_handles_labels()

        # Use a dictionary or 'if l not in labels' to ensure uniqueness
        for h, l in zip(ax_handles, ax_labels):
            if l not in labels:
                labels.append(l)
                handles.append(h)

    reordering_indeces = []
    if "csil" in labels:
        reordering_indeces.append(labels.index("csil"))
    if "sacfd" in labels:
        reordering_indeces.append(labels.index("sacfd"))
    if "vla" in labels:
        reordering_indeces.append(labels.index("vla"))
    if len(reordering_indeces) == 0:
        reordering_indeces = list(range(len(labels)))

    handles = [handles[i] for i in reordering_indeces]
    labels = [labels[i] for i in reordering_indeces]

    return handles, labels
