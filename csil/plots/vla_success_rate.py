import numpy as np
import matplotlib.pyplot as plt

num_eval_episodes = 25
num_eval_seeds = 5

success_rates = {
    "Lift": [0.95],
    "PickPlaceCan": [0.84, 0.88, 0.72, 0.8, 0.72],
    "NutAssemblySquare": [0.18],
    "ThreePieceAssembly_D0": [0.31],
}

# Bar width and positions
bar_width = 0.2

# Colors
colors = {
    "Lift": "red",
    "PickPlaceCan": "seagreen",
    "NutAssemblySquare": "slategray",
    "ThreePieceAssembly_D0": "turquoise",
}

# Figure
fig, ax = plt.subplots(figsize=(18.5, 5.2))  # compact aspect ratio

# Plot bars with error bars
for i, env in enumerate(success_rates.keys()):
    success_rate_mean = np.array(success_rates[env]).mean()
    success_rate_std = np.array(success_rates[env]).std()

    ax.bar(
        (-1.5 + i) * bar_width * 1.5,
        success_rate_mean,
        bar_width,
        yerr=success_rate_std,
        label=env,
        color=colors[env],
        capsize=3,
        edgecolor="black",
        linewidth=0.3,
        align="center",
    )

# Axes labels and title
ax.tick_params(
    axis="x",  # changes apply to the x-axis
    which="both",  # both major and minor ticks are affected
    bottom=False,  # ticks along the bottom edge are off
    top=False,  # ticks along the top edge are off
    labelbottom=False,
)

ax.set_ylabel("Success", fontsize=20)
ax.set_title(r"$\pi_0$ Performance", fontsize=25, weight="medium")
ax.set_ylim(-0.05, 1.02)
ax.set_yticks([0, 0.5, 1.0])

# Dotted horizontal baseline at y=0
ax.axhline(1, color="black", linestyle="-", linewidth=1)
ax.axhline(0, color="black", linestyle=":", linewidth=1)

# Thin black border around plot
for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_color("black")

# Legend outside plot
# ax.legend(bbox_to_anchor=(1.80, 0.5), loc="center right", frameon=False, fontsize=20)
ax.legend(loc="upper right", frameon=False, fontsize=18)

plt.tight_layout()
# plt.show()
plt.savefig("vla_base_performance.png", bbox_inches="tight", dpi=fig.dpi)
