
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

def time_level_curves(time, G, G0, xlabel, ylabel, title, save_path = None):
    plt.figure(figsize=(6, 4), dpi=200)
    
    plt.plot(time[-1] - time[:], G0, label="G₀", color='black', linewidth=2.4)
    
    # Corrected selection of probe times
    probe_times_idx = np.linspace(0, len(time) - 1, num=10, dtype=int)

    xlim_max = 0# time[-1] - time[0]  # Default to full range

    # Determine the 0.1% threshold of total amplitude (using absolute values)
    threshold = 0.001 * np.max(np.abs(G))
    for t_idx in probe_times_idx:
        if t_idx == 0:  # Skip empty slices
            continue
        
        color = to_rgba((t_idx / len(time), 0, 1 - (2 * t_idx / len(time) - 1) ** 2, 1))
        x_vals = time[t_idx] - time[:t_idx]
        y_vals = G[t_idx, :t_idx]

        if y_vals.size == 0:  # Avoid empty array errors
            continue
        
        # Find where |y_vals| still above the threshold
        valid_idx = np.where(np.abs(y_vals) >= threshold)[0]
        
        if valid_idx.size > 0:
            xlim_max = max(xlim_max, x_vals[valid_idx[0]])

        plt.plot(x_vals, y_vals, label=f"t = {round(time[t_idx], 2)}", color=color, alpha=0.9, linewidth=1.4, linestyle='solid')
    
    if xlim_max == 0 : xlim_max = time[-1] - time[0]  # Default to full range

    if xlabel == None: 
        plt.xlabel("t - t′ (s)") 
    else: 
        plt.xlabel(xlabel)
    if ylabel == None:
       plt.ylabel("g(t,t')")
    else:
        plt.ylabel(ylabel)
    if title == None:  
        plt.title("Neuron pair functional connectivity")
    else:
        plt.title(title)
    
    # Adjust x-axis limits
    plt.xlim([0, xlim_max])

    # Legend at top-right, outside the plot
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

    plt.grid(False)
    plt.box(True)
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def time_level_curves_single_kernel(time, G, save_path= None):
    plt.figure(figsize=(6, 4), dpi=200)
    
    plt.plot()
    
    # Corrected selection of probe times
    probe_times_idx = np.linspace(0, len(time) - 1, num=16, dtype=int)

    for t_idx in probe_times_idx:
        color = to_rgba((t_idx /len(time), 0, 1 - (2 * t_idx / len(time) - 1) ** 2, 1))
        plt.plot(time[t_idx] - time[:t_idx], G[t_idx, :t_idx],
                 label=f"t = {round(time[t_idx], 2)}",
                 color=color, alpha=0.9, linewidth=1.4, linestyle='solid')
    
    plt.xlabel("t - t′ (s)")
    plt.ylabel("G")
     # Legend at top-right, outside the plot
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)

    plt.grid(False)
    plt.box(True)
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def t_t_heatmap(time, G, save_path= None):
    plt.figure()
    plt.imshow(G[:, :], aspect='auto', cmap='viridis',
        extent=[time.min(), time.max(), time.min(), time.max()])
    plt.colorbar()
    plt.xlabel('t’ (s)')  # Axis 0 is t
    plt.ylabel('t (s)')  # Axis 1 is t'
    plt.tight_layout()  # Ensures proper layout
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()