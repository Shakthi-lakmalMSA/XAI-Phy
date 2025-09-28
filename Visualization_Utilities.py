import matplotlib.pyplot as plt
import numpy as np

def visualize_simulation(positions, tokens, attention_matrix, save_path=None):
    """
    Visualizes the final state of the simulation.

    Args:
        positions (np.array): Final positions of the tokens.
        tokens (list): List of tokens corresponding to the positions.
        attention_matrix (np.array): The attention matrix to visualize connections.
        save_path (str, optional): If provided, saves the plot to this file path.
    """
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Normalize positions to fit in the plot
    pos_min = np.min(positions, axis=0)
    pos_max = np.max(positions, axis=0)
    positions = (positions - pos_min) / (pos_max - pos_min + 1e-6)

    # Draw attention lines
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            if i == j: continue
            alpha = min(1.0, attention_matrix[i, j] * 2.0) # Amplify alpha for visibility
            if alpha > 0.1:
                ax.plot([positions[i, 0], positions[j, 0]], [positions[i, 1], positions[j, 1]],
                        color='cyan', alpha=alpha, linewidth=attention_matrix[i,j] * 5)

    # Draw particles (tokens)
    ax.scatter(positions[:, 0], positions[:, 1], s=200, color='magenta', zorder=5)

    # Add token labels
    for i, token in enumerate(tokens):
        ax.text(positions[i, 0], positions[i, 1] + 0.02, token,
                ha='center', va='bottom', color='white', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="magenta", lw=1, alpha=0.8))

    ax.set_title('LLM Insight: Reasoning Map', fontsize=20, color='white')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
