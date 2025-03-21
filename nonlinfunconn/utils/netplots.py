import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def neural_network(Ggap, Gsyn, Esyn, labels=None, save_path=None):
    """
    Plots and saves a network graph based on gap junction (electrical) and synaptic (chemical) connectivity,
    with excitation and inhibition shown in shades of red and blue respectively.

    Parameters:
    - Ggap: np.ndarray or list of lists (Adjacency matrix for gap junctions, undirected connections)
    - Gsyn: np.ndarray or list of lists (Adjacency matrix for chemical synapses, directed connections)
    - Esyn: np.ndarray or list of lists (Adjacency matrix for synaptic strengths, continuous values)
    - labels: np.ndarray or dict (Optional, mapping of node indices to labels)
    - filename: str (Filename to save the plotted network)
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    num_nodes = len(Ggap)
    G.add_nodes_from(range(num_nodes))
    
    # Add gap junctions (undirected edges, black)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid double-adding edges
            if Ggap[i, j] > 0:
                G.add_edge(i, j, weight=Ggap[i, j], color='black')
                G.add_edge(j, i, weight=Ggap[j, i], color='black')
    
    # Normalize Esyn values for colormap scaling
    if np.any(Esyn):
        min_val, max_val = np.min(Esyn[Esyn != 0]), np.max(Esyn)
    else:
        min_val, max_val = 0, 1  # Default range if Esyn is all zero
    
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap_red = cm.Reds
    cmap_blue = cm.Blues
    
    # Add chemical synapses with color intensity based on Esyn values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if Gsyn[i, j] > 0:  # Excitatory (shades of red)
                color = cmap_red(norm(Esyn[i, j]))
                G.add_edge(i, j, weight=Gsyn[i, j], color=color)
            elif Gsyn[i, j] < 0:  # Inhibitory (shades of blue)
                color = cmap_blue(norm(abs(Esyn[i, j])))
                G.add_edge(i, j, weight=abs(Gsyn[i, j]), color=color)
    
    
    # Get edge colors
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    
    # Define layout
    pos = nx.circular_layout(G) # nx.spring_layout(G, k=0.6)  # Force-directed layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightgray', edgecolors='black', node_size=400)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)
    
    if isinstance(labels, np.ndarray):
        labels = {i: label for i, label in enumerate(labels) if label}  # Filter out empty labels
    # Ensure only existing nodes are labeled
    labels = {i: label for i, label in labels.items() if i in G.nodes()}

    # Draw labels if provided
    if labels is not None and len(labels) > 0:    
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')
    
    # Remove axis
    plt.axis('off')
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def connect_matrices_heatmap(gamma_g=None, gamma_s=None, save_path = None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot gamma_g heatmap
    cax1 = ax[0].imshow(gamma_g, cmap='viridis', aspect='auto')
    ax[0].set_title('gamma_g Heatmap')
    ax[0].set_xlabel('Neuron Index')
    ax[0].set_ylabel('Neuron Index')
    fig.colorbar(cax1, ax=ax[0])

    # Plot gamma_s heatmap
    cax2 = ax[1].imshow(gamma_s, cmap='viridis', aspect='auto')
    ax[1].set_title('gamma_s Heatmap')
    ax[1].set_xlabel('Neuron Index')
    ax[1].set_ylabel('Neuron Index')
    fig.colorbar(cax2, ax=ax[1])
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()
