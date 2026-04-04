import pennylane as qml
from pennylane import numpy as np
import networkx as nx
import scipy
import matplotlib.pyplot as plt
import math
from scipy.optimize import minimize

np.random.seed(42)

n = int(input("Calculation for a cyclic graph with number of vertices"))
num_iters = 50  # number of optimization iterations
num_shots = 100  # number of measurements per step
target_energy = n if n % 2 == 0 else n - 1
epsilon = 1e-1

# Creating a cyclic graph
def create_graph(n):
    G = nx.Graph()
    G.add_edges_from([(i, (i+1) % n) for i in range(n)])
    return G

edges = [(i, (i + 1) % n) for i in range(n)]

graph = create_graph(n) 
pos = nx.circular_layout(graph)
nx.draw_networkx(graph, pos, with_labels=True)
plt.savefig("graph-weighted.png")

# Definition of cost and mixer Hamiltonians
def U_B(beta):
    for i in range(n):
        qml.RX(2 * beta, wires=i)

def U_C(gamma):
    for i, j in edges:
        qml.CNOT(wires=[i, j])
        qml.RZ(gamma, wires=j)
        qml.CNOT(wires=[i, j])

# Initialization of parameters with Clifford values (π/4, adding a small random noise)
def initialize_params_clifford(p):
    gammas = np.full(p, np.pi / 4) + 0.01 * np.random.randn(p)
    betas = np.full(p, np.pi / 4) + 0.01 * np.random.randn(p)
    return np.array([gammas, betas], requires_grad=True)

def bitstring_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])

# Initialization of the quantum device
dev = qml.device("default.qubit", wires=n, shots=num_shots)

@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    """
    Quantum circuit for QAOA.

    Arguments:
    gammas, betas: arrays of parameters (p elements each)
    return_samples: if True, returns measurement results of bit strings

    Returns:
    either measured bit strings, or the expected value of operator C
    """
    # Initial state: uniform superposition |+>^{⊗n}
    for i in range(n):
        qml.Hadamard(wires=i)
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_B(beta)
    if return_samples:
        return qml.sample()
    C = qml.sum(*(qml.Z(i) @ qml.Z(j) for i, j in edges))
    return qml.expval(C)

def objective(params):
    return -0.5 * (len(graph) - circuit(*params))

def qaoa_maxcut():
    """
    Runs QAOA with sequential increase of the number of layers p until convergence is achieved.

    Returns:
    int_samples_list: list of measurements for each p
    p_max: number of layers at which convergence is achieved
    energy_history_all: history of energy change for each p
    """
    p = 1
    converged = False
    int_samples_list = []
    energy_history_all = []  

    while not converged:
        print(f"\n=== Running QAOA for p={p} ===")
        init_params = initialize_params_clifford(p)
        opt = qml.AdagradOptimizer(stepsize=0.5)
        params = init_params.copy()
        energy_history = []  

        for i in range(num_iters):
            params = opt.step(objective, params)
            current_energy = -objective(params)
            energy_history.append(current_energy)  
            if (i + 1) % 5 == 0:
                print(f"Objective after step {i+1:3d}: {current_energy: .7f}")

        energy = -objective(params)
        bitstrings = circuit(*params, return_samples=True, shots=num_shots)
        sampled_ints = [bitstring_to_int(s) for s in bitstrings]
        int_samples_list.append(sampled_ints)

        counts = np.bincount(np.array(sampled_ints))
        most_freq_bit_string = np.argmax(counts)

        print(f"Optimized parameters:\ngamma: {params[0]}\nbeta:  {params[1]}")
        print(f"Most frequently sampled bit string is: {most_freq_bit_string:0{n}b}")
        print(f"Final energy: {energy:.6f}")

        energy_history_all.append(energy_history)  

        if abs(energy - target_energy) <= epsilon:
            print("Convergence achieved")
            converged = True
        else:
            p += 1

    return int_samples_list, p, energy_history_all

# =====================================================
# Visualization: top-10 most probable bitstrings for each p (number of layers)
# =====================================================

xticks = range(0, 2**n)
xtick_labels = list(map(lambda x: format(x, f"0{n}b"), xticks))
bins = np.arange(0, 2**n + 1) - 0.5

rows = math.ceil(p_max / 2)
cols = 2 if p_max > 1 else 1
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
    
for i, int_samples in enumerate(int_samples_list):
    ax = axes[i]
    ax.set_title(f"p={i + 1}")
    ax.set_xlabel("bitstrings")
    ax.set_ylabel("freq.")
    
    # Get bitstring frequencies
    counts = np.bincount(np.array(int_samples))
    sorted_counts = np.argsort(counts)[::-1]  # Sort by frequency descending
    top_bitstrings = sorted_counts[:10]      # Select top-10 bitstrings
    top_counts = counts[top_bitstrings]      # Get their frequencies
    
    # Update xticks for top-10 bitstrings
    ax.set_xticks(range(len(top_bitstrings)))                     # Set positions
    xticks_top = [format(x, f"0{n}b") for x in top_bitstrings]    # Format as binary strings
    ax.set_xticklabels(xticks_top, rotation="vertical")           # Apply labels

    # Plot histogram for top-10 bitstrings
    ax.bar(range(len(top_bitstrings)), top_counts, color="orange", width=0.6)

for j in range(len(int_samples_list), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# =====================================================
# Visualization: All bitstrings with top-2 most probable highlighted
# =====================================================

xticks = range(0, 2**n)
xtick_labels = list(map(lambda x: format(x, f"0{n}b"), xticks))
bins = np.arange(0, 2**n + 1) - 0.5

rows = math.ceil(p_max / 2)
cols = 2 if p_max > 1 else 1
fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
if isinstance(axes, np.ndarray):
    axes = axes.flatten()
else:
    axes = [axes]
    
for i, int_samples in enumerate(int_samples_list):
    ax = axes[i]
    ax.set_title(f"p={i + 1}")
    ax.set_xlabel("Bitstrings")
    ax.set_ylabel("Frequency")
    
    # Get bitstring frequencies
    counts = np.bincount(np.array(int_samples))
    
    # Find the two most frequent bitstrings (potential optimal solutions)
    sorted_counts = np.argsort(counts)[::-1]      # Sort by frequency descending
    top_bitstrings = sorted_counts[:2]           # Select top-2
    top_counts = counts[top_bitstrings]          # Get their frequencies
    
    # Configure x-axis with all bitstrings
    ax.set_xticks(xticks)                         # Set all bitstring positions
    ax.set_xticklabels(xtick_labels, rotation="vertical")  # All labels
    
    # Plot histogram for all bitstrings
    ax.hist(int_samples, bins=bins, color="orange", width=0.6, histtype="bar")
    
    # Override labels: show only top-2 on x-axis (cleaner visualization)
    ax.set_xticks(top_bitstrings)                 # Keep only top-2 positions
    ax.set_xticklabels([format(x, f"0{n}b") for x in top_bitstrings], rotation="vertical")

# Remove any unused subplots
for j in range(len(int_samples_list), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# =====================================================
# Energy convergence plot: how energy changes during optimization
# =====================================================

plt.figure(figsize=(10, 6))

# Plot energy history for each p (number of layers)
for i, history in enumerate(energy_history_all):
    plt.plot(history, label=f'p = {i + 1}')

plt.xlabel("Step")
plt.ylabel("Energy")
plt.title("Energy value evolution over optimization steps for different p")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================================
# Optimizer wrapper functions for QAOA parameter optimization
# =====================================================

# Adagrad optimizer
def run_adagrad_wrapper(p, num_iters):
    init_params = initialize_params_clifford(p)
    params = init_params.copy()
    
    opt = qml.AdagradOptimizer(stepsize=0.5)
    energy_history = []
    
    for i in range(num_iters):
        params = opt.step(objective, params)
        current_energy = -objective(params)  
        energy_history.append(current_energy)
    
    final_energy = -objective(params)
    return final_energy, energy_history

# Adam optimizer
def run_adam_wrapper(p, num_iters):
    init_params = initialize_params_clifford(p)
    params = np.array(init_params, requires_grad=True)
    
    opt = qml.AdamOptimizer(stepsize=0.1)
    energy_history = []
    
    for i in range(num_iters):
        params = opt.step(objective, params)
        current_energy = -objective(params)
        energy_history.append(current_energy)
    
    final_energy = -objective(params)
    return final_energy, energy_history

# Standard Gradient Descent optimizer
def run_gd_wrapper(p, num_iters):
    init_params = initialize_params_clifford(p)
    params = np.array(init_params, requires_grad=True)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.1)
    energy_history = []
    
    for i in range(num_iters):
        params = opt.step(objective, params)
        current_energy = -objective(params)
        energy_history.append(current_energy)
    
    final_energy = -objective(params)
    return final_energy, energy_history

# COBYLA optimizer
def run_cobyla_wrapper(p, num_iters):
    init_params = initialize_params_clifford(p).flatten()
    
    def objective_scipy(params_flat):
        params = params_flat.reshape(2, p)
        return -0.5 * (len(graph) - circuit(*params))
    
    energy_history = []
    
    def callback(xk):
        """Callback to record energy at each iteration"""
        current_energy = -objective_scipy(xk)
        energy_history.append(current_energy)
    
    result = minimize(
        objective_scipy, 
        init_params, 
        method='COBYLA',
        options={'maxiter': num_iters, 'disp': False},
        callback=callback
    )
    
    final_energy = -result.fun
    return final_energy, energy_history

# =====================================================
# Optimizer Comparison for QAOA on Cyclic Graphs
# =====================================================

def compare_optimizers(max_extra_layers=1, base_iters=40, min_iters=15):
    """
    Compare different optimizers for QAOA on cyclic graphs.
    
    This function:
    1. Determines theoretical p needed for convergence (n/2 for even n, (n-1)/2 for odd n)
    2. Tests 4 optimizers: Adagrad, Adam, GradientDescent, COBYLA
    3. Uses adaptive number of iterations (fewer for larger p)
    4. Plots energy evolution and final energy comparison
    
    Parameters:
    -----------
    max_extra_layers : int, default=1
        Number of additional layers to check beyond theoretical estimate
    base_iters : int, default=40
        Base number of iterations for p=4-5
    min_iters : int, default=15
        Minimum iterations for large p (p >= 6)
    
    Returns:
    --------
    results : dict
        Dictionary containing final_energy, energy_history, and convergence status
        for each optimizer and each p value
    """
    
    def get_adaptive_num_iters(p):
        """
        Return adaptive number of iterations based on number of layers p.
        
        Smaller p requires more iterations (harder to converge).
        Larger p converges faster, so fewer iterations are needed.
        """
        if p <= 3:
            return base_iters + 10      # 50 iterations for small p
        elif p <= 5:
            return base_iters           # 40 iterations for medium p
        else:
            return min_iters            # 15 iterations for large p
    
    # =====================================================
    # Theoretical estimate for convergence
    # =====================================================
    if n % 2 == 0:
        p_theoretical = n // 2
        print(f"\n Theoretical estimate: p_theoretical = n/2 = {p_theoretical}")
    else:
        p_theoretical = (n - 1) // 2
        print(f"\n Theoretical estimate: p_theoretical = (n-1)/2 = {p_theoretical}")
    
    # Dictionary of optimizers to compare
    optimizers = {
        'Adagrad': run_adagrad_wrapper,
        'Adam': run_adam_wrapper,
        'GradientDescent': run_gd_wrapper,
        'COBYLA': run_cobyla_wrapper
    }
    
    # Color scheme for different p values (for visualization)
    p_colors = {
        1: 'blue', 2: 'red', 3: 'green', 4: 'orange', 5: 'purple',
        6: 'brown', 7: 'pink', 8: 'gray', 9: 'olive', 10: 'cyan',
        11: 'magenta', 12: 'gold', 13: 'navy', 14: 'teal', 15: 'coral'
    }
    
    results = {}
    convergence_status = {}  # Track at which p each optimizer converged
    
    print("\n" + "=" * 70)
    print(f"OPTIMIZER COMPARISON FOR CYCLIC GRAPH C_{n}")
    print(f"Target energy (optimal MaxCut): {target_energy}")
    print("=" * 70)
    
    # Maximum p to check = theoretical + extra layers
    max_p_to_check = p_theoretical + max_extra_layers
    print(f"\n Checking p from 1 to {max_p_to_check}")
    print(f"Adaptive iterations: p=1-3 → {get_adaptive_num_iters(1)}, "
          f"p=4-5 → {get_adaptive_num_iters(4)}, p≥6 → {get_adaptive_num_iters(6)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    # =====================================================
    # Main loop: test each optimizer for all p values
    # =====================================================
    for idx, (opt_name, opt_func) in enumerate(optimizers.items()):
        print(f"\n Testing {opt_name}...")
        results[opt_name] = {}
        ax = axes[idx]
        
        for p in range(1, max_p_to_check + 1):
            num_iters_adaptive = get_adaptive_num_iters(p)
            print(f"  p = {p} (iters={num_iters_adaptive})...", end=" ", flush=True)
            
            try:
                final_energy, energy_history = opt_func(p, num_iters_adaptive)
                
                # Store results
                results[opt_name][p] = {
                    'final_energy': float(final_energy),
                    'energy_history': [float(e) for e in energy_history],
                    'converged': abs(final_energy - target_energy) <= epsilon
                }
                
                # Plot energy evolution for this p
                steps = range(1, len(energy_history) + 1)
                ax.plot(steps, energy_history,
                       color=p_colors.get(p, 'black'),
                       linewidth=1.5,
                       alpha=0.8,
                       label=f'p = {p}')
                
                # Track convergence
                if results[opt_name][p]['converged'] and opt_name not in convergence_status:
                    convergence_status[opt_name] = p
                    print(f"✅ CONVERGED at p={p} (E={final_energy:.6f})")
                else:
                    status = "✅" if results[opt_name][p]['converged'] else "❌"
                    print(f"{status} E = {final_energy:.6f}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results[opt_name][p] = {
                    'final_energy': None,
                    'energy_history': [],
                    'converged': False,
                    'error': str(e)
                }
        
        ax.axhline(y=target_energy, color='black', linestyle='--',
                  linewidth=1.5, label=f'Target E = {target_energy}')
        ax.set_xlabel('Iteration step', fontsize=10)
        ax.set_ylabel('Energy', fontsize=10)
        ax.set_title(f'{opt_name} Optimizer (C_{n})', fontsize=11)
        ax.legend(loc='best', fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # =====================================================
    # Convergence summary
    # =====================================================
    print("\n" + "=" * 70)
    print("CONVERGENCE SUMMARY")
    print("=" * 70)
    
    final_p_max = p_theoretical
    for opt_name, opt_results in results.items():
        converged_at = None
        for p in sorted(opt_results.keys()):
            if opt_results[p].get('converged', False):
                converged_at = p
                break
        
        if converged_at:
            print(f"✅ {opt_name}: converged at p = {converged_at}")
            final_p_max = max(final_p_max, converged_at)
        else:
            print(f"❌ {opt_name}: DID NOT CONVERGE up to p = {max_p_to_check}")
    
    print(f"\n Theoretical estimate: p = {p_theoretical}")
    print(f" Recommended p_max = {final_p_max}")
    
    # =====================================================
    # Bar chart: final energy vs number of layers
    # =====================================================
    fig_final, ax_final = plt.subplots(figsize=(10, 6))
    p_values = list(range(1, max_p_to_check + 1))
    x_positions = np.arange(len(p_values))
    bar_width = 0.2
    
    opt_colors = {'Adagrad': 'blue', 'Adam': 'red', 'GradientDescent': 'green', 'COBYLA': 'orange'}
    
    for i, (opt_name, opt_results) in enumerate(results.items()):
        final_energies = []
        for p in p_values:
            if p in opt_results and opt_results[p].get('final_energy') is not None:
                final_energies.append(opt_results[p]['final_energy'])
            else:
                final_energies.append(0)
        
        offset = (i - len(optimizers)/2 + 0.5) * bar_width
        ax_final.bar(x_positions + offset, final_energies, bar_width,
                    label=opt_name, color=opt_colors[opt_name], alpha=0.7)
    
    # Target energy line (theoretical optimum)
    ax_final.axhline(y=target_energy, color='black', linestyle='--',
                    linewidth=2, label=f'Target E = {target_energy}')
    
    ax_final.set_xlabel('Number of layers (p)', fontsize=12)
    ax_final.set_ylabel('Final Energy', fontsize=12)
    ax_final.set_title(f'Final Energy vs Number of Layers (C_{n})', fontsize=14)
    ax_final.set_xticks(x_positions)
    ax_final.set_xticklabels(p_values)
    ax_final.legend(loc='upper right')
    ax_final.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    return results


# =====================================================
# Run the comparison
# =====================================================
if __name__ == "__main__":
    compare_optimizers(max_extra_layers=1, base_iters=40, min_iters=15)

int_samples_list, p_max, energy_history_all = qaoa_maxcut()
