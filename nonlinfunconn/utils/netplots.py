import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
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
    
    # Add gap junctions (undirected edges, black)
    for i in range(num_nodes):
        for j in range(num_nodes):  # Avoid double-adding edges
            if Ggap[i, j] > 0:
                G.add_edge(j, i, weight=Ggap[i, j], color='black')
    
    # Normalize Esyn values for colormap scaling
    min_val, max_val = np.min([np.min(Esyn), -1]), np.max([np.max(Esyn), 0]) # -70 to 10 mV
 
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = cm.get_cmap('coolwarm')  # Single colormap ranging from blue to red
    
    # Add chemical synapses with color intensity based on Esyn values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if Gsyn[i, j] != 0:  # Only add edges where there is a synaptic connection
                color = cmap(norm(Esyn[i, j]))
                G.add_edge(j, i, weight=Gsyn[i, j], color=color)  # 'j' is the origin (source) to 'i' is the target (destination)
 
    # Get edge colors
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    
    # Normalize edge widths to a reasonable range (e.g., 1 to 10)
    min_width, max_width = 1, 8
    if np.max(edge_weights) > 0:
        edge_widths = min_width + (max_width - min_width) * (edge_weights - np.min(edge_weights)) / (np.max(edge_weights) - np.min(edge_weights))
    else:
        edge_widths = np.full(len(edge_weights), min_width)  # Default width if all weights are zero

    # Define layout
    pos = positions if positions is not None else nx.circular_layout(G) #nx.spring_layout(G, k=None)  # Force-directed layout with increased k value
    
    # Calculate node sizes based on connectivity strength
    # Node size is proportional to the sum of weights of incoming and outgoing edges
    node_strengths = np.zeros(num_nodes)
    for j in range(num_nodes):
        # Sum of weights for outgoing edges (Gsyn) and gap junctions (Ggap) from presynaptic neuron j
        node_strengths[j] = np.sum(np.abs(Ggap[:, j])) + np.sum(np.abs(Gsyn[:, j]))
    
    # Normalize node sizes to a reasonable range (e.g., 100 to 1000)
    min_size, max_size = 400, 1000
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

    plt.figure(figsize=(12, 12))
    if num_nodes>20:
        plt.figure(figsize=(80, 80))
    
    # Draw nodes with sizes proportional to their strength
    nx.draw_networkx_nodes(
        G, pos, node_color='lightgray', edgecolors=node_border_colors, 
        linewidths=node_border_widths, node_size=node_sizes
    )
    # Draw edges with arrows
    nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, arrows=True, width=edge_widths,
        arrowstyle='->', arrowsize=20  # Adjust arrow size and style here
    )

    # Draw labels if provided
    if labels is not None and len(labels) > 0:    
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    
    # Remove axis
    plt.axis('off')
    
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
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