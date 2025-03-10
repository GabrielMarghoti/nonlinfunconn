
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


def plot_level_curves(t_s, G, probe_times_idx, plot_times_idx, save_file):
    plt.figure(figsize=(5, 4), dpi=200)
    
    plt.plot(t_s[-1] - t_s[:], G[-1, :], label="G₀", color='black', linewidth=2.4)
    
    for t_idx in probe_times_idx:
        color = to_rgba((t_idx / plot_times_idx[-1], 0, 1 - (2 * t_idx / plot_times_idx[-1] - 1) ** 2, 1))
        plt.plot(t_s[t_idx] - t_s[1:t_idx], G[t_idx, 1:t_idx],
                 label=f"t = {round(t_s[t_idx], 2)}",
                 color=color, alpha=0.9, linewidth=1.4, linestyle='solid')
    
    plt.xlabel("t - t′")
    plt.ylabel("G")
    plt.legend()
    plt.grid(False)
    plt.box(True)
    
    plt.savefig(save_file)
    plt.close()
