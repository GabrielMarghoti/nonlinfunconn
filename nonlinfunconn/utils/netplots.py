import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA



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

    def normalize_weights(weights, min_width=0.0, max_width=5):
        if weights.size == 0:
            return weights  # or return np.zeros_like(weights) depending on context
        if np.max(weights) > 0 and np.max(weights) != np.min(weights):
            return min_width + (max_width - min_width) * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return np.full(len(weights), min_width)  # Default width if all weights are zero

    gap_edge_weights = normalize_weights(np.array(gap_edge_weights))
    syn_edge_weights = normalize_weights(np.array(syn_edge_weights))

    # Define layout
    pos = positions if positions is not None else nx.circular_layout(G)
    # nx.spring_layout(G, k=0.1, fixed=[0], pos={0: (1, 0)})

    # Node size is proportional to the sum of weights of incoming and outgoing edges
    node_strengths = np.zeros(num_nodes)
    for j in range(num_nodes):
        # Sum of weights for outgoing edges (Gsyn) and gap junctions (Ggap) from presynaptic neuron j
        node_strengths[j] = np.sum(np.abs(Ggap[:, j])) + np.sum(np.abs(Gsyn[:, j]))
    
    # Normalize node sizes to a reasonable range (e.g., 100 to 1000)
    min_size, max_size = 1200, 2000
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
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # Draw labels if provided
    if labels is not None and len(labels) > 0:    
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=12, font_color='black')
    
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





def dynamics_network(A, labels=None, positions = None, save_path=None):
    """
    Plots and saves a network graph based on cumulative signal propagation kernels,
    Parameters:
    - Ggap: np.ndarray or list of lists (Adjacency matrix for gap junctions, undirected connections)
    - Gsyn: np.ndarray or list of lists (Adjacency matrix for chemical synapses, directed connections)
    - Esyn: np.ndarray or list of lists (Adjacency matrix for synaptic strengths, continuous values)
    - labels: np.ndarray or dict (Optional, mapping of node indices to labels)
    - save_path: str (Filename to save the plotted network)
    """
    # Create a directed graph
    G = nx.DiGraph()
    
    num_nodes = len(A)
    G.add_nodes_from(range(num_nodes))
    
    # Create lists to store edges for gap junctions and chemical synapses
    edges = []
    edge_colors = []
    edge_weights = []


    # Normaliz colormap scaling
    min_val, max_val =  min(A.min(),-1), max(A.max(),1)  # -55 to 10 mV
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = cm.get_cmap('coolwarm')  # Single colormap ranging from blue to red

    # Add chemical synapses with color intensity based on Esyn values
    for i in range(num_nodes):
        for j in range(num_nodes):
            if A[i, j] != 0:  # Only add edges where there is a synaptic connection
                color = cmap(norm(A[i, j]))
                edges.append((j, i))  # Store as (source, target, weight, color)
                edge_weights.append(np.abs(A[i, j]))  # Store weight
                edge_colors.append(color)  # Color for synaptic connections


    # Get edge colors
    # Normalize edge widths to a reasonable range (e.g., 1 to 10)

    def normalize_weights(weights, min_width=0.2, max_width=2):
        if weights.size == 0:
            return weights  # or return np.zeros_like(weights) depending on context
        if np.max(weights) > 0 and np.max(weights) != np.min(weights):
            return min_width + (max_width - min_width) * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        return np.full(len(weights), min_width)  # Default width if all weights are zero

    edge_weights = normalize_weights(np.array(edge_weights))

    # Define layout
    if positions is not None:
        pos = positions
    else:
        pos = nx.spring_layout(G, seed=42)

    # Node size is proportional to the sum of weights of incoming and outgoing edges
    node_strengths = np.zeros(num_nodes)
    for j in range(num_nodes):
        # Sum of weights for outgoing edges (Gsyn) and gap junctions (Ggap) from presynaptic neuron j
        node_strengths[j] = np.sum(np.abs(A[:, j])) + np.sum(np.abs(A[:, j]))
    
    # Normalize node sizes to a reasonable range (e.g., 100 to 1000)
    min_size, max_size = 800, 1000
    if np.max(node_strengths) > 0:
        node_sizes = min_size + (max_size - min_size) * (node_strengths - np.min(node_strengths)) / (np.max(node_strengths) - np.min(node_strengths))
    else:
        node_sizes = np.full(num_nodes, min_size)  # Default size if all strengths are zero
    # Make the first node have a bold stroke
    node_border_colors = ['black'] * num_nodes
    node_border_colors[0] = 'red'  # Set the first node's border color to red
    node_border_widths = [1] * num_nodes
    node_border_widths[0] = 3  # Set the first node's border width to 3


    
    if isinstance(labels, np.ndarray):
        labels = {i: label for i, label in enumerate(labels) if label}  # Filter out empty labels
    # Ensure only existing nodes are labeled
    labels = {i: label for i, label in labels.items() if i in G.nodes()}
    
    fig, ax = plt.subplots(figsize=(8, 8) if num_nodes <= 20 else (20, 20))
    
    # Draw nodes with sizes proportional to their strength
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color='lightgray', edgecolors=node_border_colors, 
        linewidths=node_border_widths, node_size=node_sizes
    )
    
    # Draw edges for synaptic connections with arrows
    nx.draw_networkx_edges(
        G, pos, ax=ax, edge_color=edge_colors, arrows=True, width=edge_weights, edgelist=edges,
        arrowstyle='->', arrowsize=30  # Adjust arrow size and style here
    )

    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Stimulated node', 
               markerfacecolor='lightgray', markeredgecolor='red', markersize=8, markeredgewidth=1.6),
        Line2D([0], [0], marker='o', color='w', label='Responsive nodes', 
               markerfacecolor='lightgray', markeredgecolor='black', markersize=8, markeredgewidth=1),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(1.0), lw=2, label='Signal amplification'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.0), lw=2, label='Signal suppresion')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=12)

    # Draw labels if provided
    if labels is not None and len(labels) > 0:    
        nx.draw_networkx_labels(G, pos, ax=ax, labels=labels, font_size=10, font_color='black')
    
    # Remove axis
    ax.axis('off')
    
    if save_path is None:
        fig.show()
    else:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close()

def dynamics_network_3d(A, labels=None, positions=None, save_path=None, auto_angle=True):
    """
    3D network plot using signal propagation matrix A.

    Parameters:
    - A: np.ndarray, weighted adjacency matrix
    - labels: list or dict, node labels
    - positions: np.ndarray of shape (N, 3), 3D positions of nodes
    - save_path: str or None, where to save the plot
    - auto_angle: bool, whether to auto-select best viewing angle (PCA)
    """
    G = nx.DiGraph()
    num_nodes = A.shape[0]
    G.add_nodes_from(range(num_nodes))

    edges = []
    edge_colors = []
    edge_weights = []

    # Normalize for colormap
    min_val, max_val = min(A.min(), 0), max(A.max(), 0)
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    cmap = cm.get_cmap('coolwarm')

    for i in range(num_nodes):
        for j in range(num_nodes):
            if A[i, j] != 0:
                color = cmap(norm(A[i, j]))
                edges.append((j, i))  # from j to i
                edge_weights.append(np.abs(A[i, j]))
                edge_colors.append(color)

    def normalize_weights(weights, min_width=0.5, max_width=4):
        weights = np.array(weights)
        if weights.size == 0 or np.max(weights) == np.min(weights):
            return np.full_like(weights, min_width)
        return min_width + (max_width - min_width) * (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

    edge_weights = normalize_weights(np.array(edge_weights))

    # Handle positions
    if positions is None:
        # Generate synthetic 3D positions if none provided
        pos_2d = nx.spring_layout(G, seed=42, dim=2)
        positions = np.array([[x, y, 0] for x, y in pos_2d.values()])
    positions = np.asarray(positions)

    # Node sizes
    node_strengths = np.sum(np.abs(A), axis=0) + np.sum(np.abs(A), axis=1)
    node_sizes = 100 + 200 * (node_strengths - node_strengths.min()) / (node_strengths.ptp() + 1e-6)

    node_border_colors = ['black'] * num_nodes
    node_border_colors[0] = 'red'
    node_border_widths = [1] * num_nodes
    node_border_widths[0] = 3

    if isinstance(labels, (list, np.ndarray)):
        labels = {i: label for i, label in enumerate(labels) if label}
    labels = {i: label for i, label in labels.items() if i in G.nodes()} if labels else {}

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    ax.scatter(xs, ys, zs, s=node_sizes, c='lightgray', edgecolors=node_border_colors,
               linewidths=node_border_widths)

    # Draw edges
    for (j, i), color, width in zip(edges, edge_colors, edge_weights):
        x = [positions[j, 0], positions[i, 0]]
        y = [positions[j, 1], positions[i, 1]]
        z = [positions[j, 2], positions[i, 2]]
        ax.plot(x, y, z, color=color, linewidth=width, alpha=0.8)

    # Label nodes
    for idx, label in labels.items():
        x, y, z = positions[idx]
        ax.text(x, y, z + 0.02, label, fontsize=10, ha='center', va='bottom')

    # Optimize view
    if auto_angle:
        pca = PCA(n_components=2)
        xy = pca.fit_transform(positions)
        elev = 30
        azim = np.rad2deg(np.arctan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0]))
        ax.view_init(elev=elev, azim=azim)
    else:
        ax.view_init(elev=20, azim=45)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Stimulated node',
               markerfacecolor='lightgray', markeredgecolor='red', markersize=8, markeredgewidth=1.6),
        Line2D([0], [0], marker='o', color='w', label='Responsive nodes',
               markerfacecolor='lightgray', markeredgecolor='black', markersize=8, markeredgewidth=1),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(1.0), lw=2, label='Signal amplification'),
        Line2D([0], [0], color=cm.get_cmap('coolwarm')(0.0), lw=2, label='Signal suppression')
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.axis('off')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()