import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Define cell states
M = 0  # Metal
OX = 1  # Oxide
EF = 2  # Electric Field
A = 3   # Anion-rich Oxide
S = 4   # Solvent

# Simulation parameters
L = 20  # Lattice size (LxLxL)
STEPS = 500  # Number of simulation steps (reduced for clarity)
P_REACTION = 0.2  # Increased probability for reaction-like rules to see more transitions
P_DISSOLUTION = 0.1  # Probability for oxide dissolution
P_ANION = 0.1  # Probability for anion incorporation
P_BOND = 0.2  # Unbonding probability for surface reorganization

# Initialize 3D lattice
def initialize_lattice(L):
    lattice = np.full((L, L, L), S, dtype=int)  # Fill with solvent
    # Bottom layer (z=0) is metal
    lattice[:, :, 0] = M
    # Initial oxide layer at z=1
    lattice[:, :, 1] = OX
    # Add some initial EF and A states to observe transitions
    for i in range(L):
        for j in range(L):
            if random.random() < 0.1:
                lattice[i, j, 2] = EF
            if random.random() < 0.05:
                lattice[i, j, 2] = A
    return lattice

# Get Moore neighborhood indices (26 neighbors in 3D)
def get_moore_neighborhood(i, j, k, L):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                ni, nj, nk = (i + di) % L, (j + dj) % L, (k + dk) % L  # Periodic in x, y
                if nk >= 0 and nk < L:  # Fixed boundaries in z
                    neighbors.append((ni, nj, nk))
    return neighbors

# Apply rules
def apply_rules(lattice, i, j, k, ni, nj, nk):
    state1 = lattice[i, j, k]
    state2 = lattice[ni, nj, nk]
    
    # Reaction-like rules
    if random.random() < P_REACTION:
        if (state1 == M and state2 == S) or (state2 == M and state1 == S):
            return (OX, OX)  # M + S -> OX + OX
        if (state1 == EF and state2 == S) or (state2 == EF and state1 == S):
            if random.random() < P_DISSOLUTION:
                return (S, S)  # EF + S -> S + S
            if random.random() < P_ANION:
                return (A, S)  # EF + S -> A + S
        if (state1 == M and state2 == A) or (state2 == M and state1 == A):
            return (OX, OX)  # M + A -> OX + OX
        if (state1 == M and state2 == OX) or (state2 == M and state1 == OX):
            return (M, EF)  # M + OX -> M + EF
    
    # Diffusion-like rules
    if (state1 == EF and state2 == OX) or (state2 == EF and state1 == OX):
        return (OX, EF)  # EF + OX -> OX + EF
    if (state1 == EF and state2 == A) or (state2 == EF and state1 == A):
        return (A, EF)  # EF + A -> A + EF
    if (state1 == S and state2 == OX) or (state2 == S and state1 == OX):
        return (OX, S)  # S + OX -> OX + S
    if (state1 == S and state2 == A) or (state2 == S and state1 == A):
        return (A, S)  # S + A -> A + S
    
    # No change
    return (state1, state2)

# Count oxide-like neighbors (OX, EF, A)
def count_oxide_neighbors(lattice, i, j, k):
    neighbors = get_moore_neighborhood(i, j, k, lattice.shape[0])
    return sum(1 for ni, nj, nk in neighbors if lattice[ni, nj, nk] in [OX, EF, A])

# Surface reorganization (simplified)
def surface_reorganization(lattice, i, j, k, ni, nj, nk):
    state1 = lattice[i, j, k]
    state2 = lattice[ni, nj, nk]
    if (state1 in [OX, A] and state2 == S) or (state2 in [OX, A] and state1 == S):
        n1 = count_oxide_neighbors(lattice, i, j, k)
        n2 = count_oxide_neighbors(lattice, ni, nj, nk)
        N = n1 - n2 if state1 in [OX, A] else n2 - n1
        if N < 0 or random.random() < P_BOND ** max(N, 0):
            return (state2, state1)
    return (state1, state2)

# Count states in the lattice
def count_states(lattice):
    counts = defaultdict(int)
    for state in [M, OX, EF, A, S]:
        counts[state] = np.sum(lattice == state)
    return counts

# Main simulation loop
def simulate_anodization():
    lattice = initialize_lattice(L)
    state_counts = {state: [] for state in [M, OX, EF, A, S]}
    labels = {M: 'Metal (M)', OX: 'Oxide (OX)', EF: 'Electric Field (EF)', A: 'Anion (A)', S: 'Solvent (S)'}
    colors = {M: 'red', OX: 'blue', EF: 'green', A: 'purple', S: 'yellow'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for step in range(STEPS):
        # Update all sites asynchronously
        indices = [(i, j, k) for i in range(L) for j in range(L) for k in range(L)]
        random.shuffle(indices)
        
        for i, j, k in indices:
            neighbors = get_moore_neighborhood(i, j, k, L)
            if not neighbors:
                continue
            ni, nj, nk = random.choice(neighbors)
            
            # Apply rules
            new_state1, new_state2 = apply_rules(lattice, i, j, k, ni, nj, nk)
            lattice[i, j, k], lattice[ni, nj, nk] = new_state1, new_state2
            
            # Apply surface reorganization
            new_state1, new_state2 = surface_reorganization(lattice, i, j, k, ni, nj, nk)
            lattice[i, j, k], lattice[ni, nj, nk] = new_state1, new_state2
        
        # Record state counts
        counts = count_states(lattice)
        for state in [M, OX, EF, A, S]:
            state_counts[state].append(counts[state])
        
        # Visualize every 50 steps
        if step % 50 == 0:
            # Plot cross-section
            ax1.clear()
            cross_section = lattice[:, :, L//2]  # Middle z-plane
            ax1.imshow(cross_section, cmap='viridis', interpolation='nearest')
            ax1.set_title(f'Cross-section at Step {step}')
            
            # Plot state counts
            ax2.clear()
            for state in [M, OX, EF, A, S]:
                ax2.plot(state_counts[state], label=labels[state], color=colors[state])
            ax2.set_title('State Counts Over Time')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Number of Cells')
            ax2.legend()
            ax2.grid(True)
            
            plt.pause(0.1)
    
    plt.show()
    return lattice

# Run simulation
if __name__ == "__main__":
    simulate_anodization()