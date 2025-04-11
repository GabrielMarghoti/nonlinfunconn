import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def neural_network(Ggap, Gsyn, Esyn, labels=None, positions = None, save_path=None):
    """
    Plots and saves a network graph based on gap junction (electrical) and synaptic (chemical) connectivity,
    with excitation and inhibition shown in shades of red and blue respectively. Node sizes are proportional
    to their connectivity strength.

    Parameters:
    - Ggap: np.ndarray or list of lists (Adjacency matrix for gap junctions, undirected connections)
    - Gsyn: np.ndarray or list of lists (Adjacency matrix for chemical synapses, directed connections)
    - Esyn: np.ndarray or list of lists (Adjacency matrix for synaptic strengths, continuous values)
    - labels: np.ndarray or dict (Optional, mapping of node indices to labels)
    - save_path: str (Filename to save the plotted network)
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    num_nodes = len(Ggap)
    G.add_nodes_from(range(num_nodes))
    
    # Create lists to store edges for gap junctions and chemical synapses
    gap_edges = []
    gap_edge_colors = []
    gap_edge_weights = []

    syn_edges = []
    syn_edge_colors = []
    syn_edge_weights = []

    # Add gap junctions (undirected edges, black)
    for i in range(num_nodes):
        for j in range(num_nodes):  # Avoid double-adding edges
            if Ggap[i, j] > 0:
                gap_edges.append((j, i))  # Store as (source, target, weight)
                gap_edge_weights.append(Ggap[i, j])  # Store weight
                gap_edge_colors.append('black')  # Color for gap junctions

    # Normalize Esyn values for colormap scaling
    min_val, max_val = np.min([np.min(Esyn), -50]), np.max([np.max(Esyn), 0])  # -55 to 10 mV
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = cm.get_cmap('coolwarm')  # Single colormap ranging from blue to red

    # Add chemical synapses with color intensity based on Esyn values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if Gsyn[i, j] != 0:  # Only add edges where there is a synaptic connection
                color = cmap(norm(Esyn[i, j]))
                syn_edges.append((j, i))  # Store as (source, target, weight, color)
                syn_edge_weights.append(Gsyn[i, j])  # Store weight
                syn_edge_colors.append(color)  # Color for synaptic connections


    # Get edge colors
    # Normalize edge widths to a reasonable range (e.g., 1 to 10)

    def normalize_weights(weights, min_width=0.6, max_width=6):
        if weights.size == 0:
            return weights  # or return np.zeros_like(weights) depending on context
        if np.max(weights) > 0 and np.max(weights) != np.min(weights):
            return min_width + (max_width - min_width) * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return np.full(len(weights), min_width)  # Default width if all weights are zero

    gap_edge_weights = normalize_weights(np.array(gap_edge_weights))
    syn_edge_weights = normalize_weights(np.array(syn_edge_weights))

    # Define layout
    pos = positions if positions is not None else nx.spring_layout(
        G, k=0.1, fixed=[0], pos={0: (1, 0)}
    ) 
    
    # Node size is proportional to the sum of weights of incoming and outgoing edges
    node_strengths = np.zeros(num_nodes)
    for j in range(num_nodes):
        # Sum of weights for outgoing edges (Gsyn) and gap junctions (Ggap) from presynaptic neuron j
        node_strengths[j] = np.sum(np.abs(Ggap[:, j])) + np.sum(np.abs(Gsyn[:, j]))
    
    # Normalize node sizes to a reasonable range (e.g., 100 to 1000)
    min_size, max_size = 100, 500
    if np.max(node_strengths) > 0:
        node_sizes = min_size + (max_size - min_size) * (node_strengths - np.min(node_strengths)) / (np.max(node_strengths) - np.min(node_strengths))
    else:
        node_sizes = np.full(num_nodes, min_size)  # Default size if all strengths are zero
    # Make the first node have a bold stroke
    node_border_colors = ['black'] * num_nodes
    node_border_colors[0] = 'magenta'  # Set the first node's border color to red
    node_border_widths = [1] * num_nodes
    node_border_widths[0] = 3  # Set the first node's border width to 3


    
    if isinstance(labels, np.ndarray):
        labels = {i: label for i, label in enumerate(labels) if label}  # Filter out empty labels
    # Ensure only existing nodes are labeled
    labels = {i: label for i, label in labels.items() if i in G.nodes()}
    
    fig, ax = plt.subplots(figsize=(10, 10) if num_nodes <= 20 else (20, 20))
    
    # Draw nodes with sizes proportional to their strength
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color='lightgray', edgecolors=node_border_colors, 
        linewidths=node_border_widths, node_size=node_sizes
    )
    # Draw edges for gap junctions without arrows
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=gap_edge_colors, arrows=False, width=gap_edge_weights, edgelist=gap_edges
    )
    # Draw edges for synaptic connections with arrows
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=syn_edge_colors, arrows=True, width=syn_edge_weights, edgelist=syn_edges,
        arrowstyle='->', arrowsize=10, connectionstyle='arc3,rad=0.2'  # Adjust arrow size and style here
    )

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Stimulated node', 
               markerfacecolor='lightgray', markeredgecolor='magenta', markersize=8, markeredgewidth=1.6),
        Line2D([0], [0], marker='o', color='w', label='Responsive node', 
               markerfacecolor='lightgray', markeredgecolor='black', markersize=8, markeredgewidth=1),
        Line2D([0], [0], color='black', lw=2, label='Gap junction electrical synapse'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.0), lw=2, label='Inhibitory chemical synapse'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(1.0), lw=2, label='Excitatory chemical synapse')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # Draw labels if provided
    if labels is not None and len(labels) > 0:    
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=8, font_color='black')
    
    # Remove axis
    ax.axis('off')
    
    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close()



def connect_matrices_heatmap(gamma_g=None, gamma_s=None, labels=None, save_path=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 8))

    # Plot gamma_g heatmap
    cax1 = ax[0].imshow(gamma_g, cmap='viridis', aspect='equal')
    ax[0].set_title('gamma_g Heatmap')
    ax[0].set_xlabel('Presynaptic Neuron')
    ax[0].set_ylabel('Postsynaptic Neuron')
    if labels is not None:
        ax[0].set_xticks(range(len(labels)))
        ax[0].set_yticks(range(len(labels)))
        ax[0].set_xticklabels(labels, rotation=90)
        ax[0].set_yticklabels(labels)
    fig.colorbar(cax1, ax=ax[0])

    # Plot gamma_s heatmap
    cax2 = ax[1].imshow(gamma_s, cmap='viridis', aspect='equal')
    ax[1].set_title('gamma_s Heatmap')
    ax[1].set_xlabel('Presynaptic Neuron')
    ax[1].set_ylabel('Postsynaptic Neuron')
    if labels is not None:
        ax[1].set_xticks(range(len(labels)))
        ax[1].set_yticks(range(len(labels)))
        ax[1].set_xticklabels(labels, rotation=90)
        ax[1].set_yticklabels(labels)
    fig.colorbar(cax2, ax=ax[1])
    
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()