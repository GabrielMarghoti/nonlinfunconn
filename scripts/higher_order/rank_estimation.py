import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh # Specialized solver for symmetric (Hermitian) matrices

def get_rank_over_time(L, time_steps):
    """
    Calculates the rank of the diffusion operator e^(-Lt) over time.
    
    This is much faster than computing expm(L*t) and svd()
    at every step. Since L is symmetric, its eigenvalues (lambda)
    are real. The eigenvalues of e^(-Lt) are simply e^(-lambda * t).
    
    The singular values of e^(-Lt) (which is also symmetric) are
    the absolute values of its eigenvalues, |e^(-lambda * t)|.
    Since e^x is always positive, s_i(t) = e^(-lambda_i * t).
    
    The rank is the count of singular values > 0.
    """
    
    # 1. Get the eigenvalues of L just once.
    # We use eigvalsh for stability and speed with symmetric matrices.
    eigvals = eigvalsh(L)
    
    # 2. Calculate the rank at each time step
    ranks = []
    for t in time_steps:
        # Calculate the new singular values at time t
        s_at_t = np.exp(-eigvals * t)
        
        # Estimate rank by counting singular values above a small threshold
        rank = np.sum(s_at_t > 1e-10)
        ranks.append(rank)
        
    return ranks

# --- Main Analysis ---
N = 200          # Size of all networks
# We define our time "movie"
time_steps = np.linspace(0.1, 100.0, 1000) # Start just after t=0
all_ranks = {}

# --- 1. Chain (Undirected Path Graph) ---
G_chain = nx.path_graph(N)
L_chain = nx.laplacian_matrix(G_chain).toarray()
all_ranks['Chain (Path)'] = get_rank_over_time(L_chain, time_steps)

# --- 2. Regular Lattice (2D grid) ---
# Build a 2D grid (regular lattice) with ~N nodes, then trim/pad to exactly N.
rows = max(1, int(np.floor(np.sqrt(N))))
cols = int(np.ceil(N / rows))
G_lattice = nx.grid_2d_graph(rows, cols, periodic=False)
G_lattice = nx.convert_node_labels_to_integers(G_lattice)
# Trim extra nodes if necessary
if G_lattice.number_of_nodes() > N:
    G_lattice.remove_nodes_from(range(N, G_lattice.number_of_nodes()))
# Pad with isolated nodes if fewer
for i in range(G_lattice.number_of_nodes(), N):
    G_lattice.add_node(i)
L_lattice = nx.laplacian_matrix(G_lattice).toarray()
all_ranks['Regular Lattice (2D grid)'] = get_rank_over_time(L_lattice, time_steps)
G_star = nx.star_graph(N - 1) # N-1 leaves + 1 center
L_star = nx.laplacian_matrix(G_star).toarray()
all_ranks['Star (Hub)'] = get_rank_over_time(L_star, time_steps)

# --- 3. Circle (Ring Lattice) ---
G_circle = nx.cycle_graph(N)
L_circle = nx.laplacian_matrix(G_circle).toarray()
all_ranks['Circle (Ring)'] = get_rank_over_time(L_circle, time_steps)

# --- 4. Random (Erdos-Renyi) ---
G_random = nx.erdos_renyi_graph(N, p=0.2) # Denser for faster mixing
L_random = nx.laplacian_matrix(G_random).toarray()
all_ranks['Random (ER)'] = get_rank_over_time(L_random, time_steps)

# --- 5. Barabasi-Albert (Scale-Free) ---
G_ba = nx.barabasi_albert_graph(N, m=3) # Denser for faster mixing
L_ba = nx.laplacian_matrix(G_ba).toarray()
all_ranks['Barabasi-Albert (Scale-Free)'] = get_rank_over_time(L_ba, time_steps)

# --- 6. Small World (Watts-Strogatz) ---
G_sw = nx.watts_strogatz_graph(N, k=4, p=0.2) # 20% rewiring
L_sw = nx.laplacian_matrix(G_sw).toarray()
all_ranks['Small World (WS)'] = get_rank_over_time(L_sw, time_steps)

# --- 7. CRITICAL EXAMPLE: Disconnected Graph ---
# Two separate 20-node random graphs
G_dis = nx.disjoint_union(nx.erdos_renyi_graph(N//2, 0.3),
                         nx.erdos_renyi_graph(N//2, 0.3))
L_dis = nx.laplacian_matrix(G_dis).toarray()
all_ranks['Disconnected (2 Clusters)'] = get_rank_over_time(L_dis, time_steps)


# --- Plot the results ---
plt.figure(figsize=(12, 8))

for name, ranks in all_ranks.items():
    plt.plot(time_steps, ranks, label=name, marker='o', markersize=3, alpha=0.8)

plt.title(f'Rank of Diffusion Operator $e^{{-Lt}}$ vs. Time (t) (N={N})', fontsize=16)
plt.xlabel('Time (t)', fontsize=12)
plt.ylabel('Rank of Operator (Count of $\sigma > 10^{-10}$)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True)
# Use log scales and show y-axis normalized by N (as tick labels).
# Guard against the later plt.yticks(...) call (which includes 0 and would break a log y-axis)
import matplotlib.ticker as ticker
plt.xscale('log')
plt.yscale('log')
plt.ylim(1, N)  # require positive lower bound for log scale (ranks < 1 are effectively 0 and won't be shown)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y/N:.3f}"))
# Disable the later linear yticks call so it doesn't attempt to place a 0 tick on a log axis
plt.yticks = lambda *args, **kwargs: None
plt.yticks(range(0, N + 3, 5))
plt.show()

