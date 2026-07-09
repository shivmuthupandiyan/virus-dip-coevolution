import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pdesolver import create_fft_exp_kernel_padded
import time
import os

# --- Helper and Data Packaging Functions ---
def create_snapshots_at_timepoints(solver_results, num_snapshots=100):
    """
    Creates a series of analysis 'snapshots' by downsampling the continuous-time simulation results.
    """
    required_keys = ['time_points', 'v_dist_time', 'd_dist_time']
    if not all(key in solver_results for key in required_keys):
        raise KeyError("Results missing density series or time_points.")
        
    time_points = solver_results['time_points']
    v_series, d_series = solver_results['v_dist_time'], solver_results['d_dist_time']
    n_total_steps = len(time_points)
    
    # Select evenly spaced indices from the full simulation time series
    snapshot_indices = np.linspace(0, n_total_steps - 1, num_snapshots, dtype=int)
    print(f"  Downsampling full {n_total_steps}-step time series to {len(snapshot_indices)} representative snapshots.")
    
    snapshots = []
    for i, time_idx in enumerate(snapshot_indices):
        snapshots.append({
            'time': time_points[time_idx],
            'snapshot_idx': i,  # The index in our downsampled list (0, 1, 2...)
            'virus_pop': v_series[:, :, time_idx],
            'dip_pop': d_series[:, :, time_idx]
        })
    return snapshots

def compute_normalized_interference(virus_pop, dip_pop, params, fft_K_exp):
    dx, dy = params['dx'], params['dy']; V_total = np.sum(virus_pop) * dx * dy; D_total = np.sum(dip_pop) * dx * dy
    if V_total < 1e-9 or D_total < 1e-9: return 0.0
    v_norm, d_norm = virus_pop / V_total, dip_pop / D_total
    from numpy.fft import fft2, ifft2
    ny, nx = params['nx'], params['nx']; padded_ny, padded_nx = params['padded_ny'], params['padded_ny']
    d_norm_padded = np.zeros((padded_ny, padded_nx)); d_norm_padded[:ny, :nx] = d_norm
    fft_d_norm_padded = fft2(d_norm_padded); conv_d_norm_spatial = np.real(ifft2(fft_K_exp * fft_d_norm_padded))[:ny, :nx]
    normalized_rate_field = params['kappa'] * conv_d_norm_spatial * (dx * dy)
    return np.sum(normalized_rate_field * v_norm) * dx * dy

def compute_interference_matrix(snapshots, params):
    n_snapshots = len(snapshots); interference_matrix = np.zeros((n_snapshots, n_snapshots))
    padded_ny, padded_nx = params['padded_ny'], params['padded_ny']
    fft_K_exp = create_fft_exp_kernel_padded(params, 'sigma', padded_nx, padded_ny)
    for i in range(n_snapshots):
        for j in range(n_snapshots):
            interference_matrix[i, j] = compute_normalized_interference(snapshots[i]['virus_pop'], snapshots[j]['dip_pop'], params, fft_K_exp)
    return interference_matrix

def plot_resistance_to_dip(interference_matrix, snapshots, ref_dip_snapshot_idx, save_path=None):
    """
    Generates a line graph showing the resistance of virus snapshots
    against a single reference DIP snapshot over a continuous time window.
    """
    n_snapshots = interference_matrix.shape[0]
    if not (0 <= ref_dip_snapshot_idx < n_snapshots):
        print(f"Error: ref_dip_snapshot_idx {ref_dip_snapshot_idx} is out of bounds (0-{n_snapshots-1}).")
        return
        
    ref_dip_time = snapshots[ref_dip_snapshot_idx]['time']
    print(f"\n  Generating resistance profile plot for DIP at T={ref_dip_time:.1f} (Index {ref_dip_snapshot_idx})...")

    interference_col = interference_matrix[:, ref_dip_snapshot_idx]
    max_interference = np.max(interference_col)
    if max_interference < 1e-12:
        print(f"  -> DIP at T={ref_dip_time:.1f} has no effect. Skipping plot.")
        return
        
    resistance_col = max_interference / (interference_col + 1e-15)

    # Define the continuous window: 10 snapshots before to 30 after the reference
    start_idx = max(0, ref_dip_snapshot_idx - 10)
    end_idx = min(n_snapshots, ref_dip_snapshot_idx + 31)  # +30 inclusive
    
    virus_indices_to_plot = np.arange(start_idx, end_idx)
    
    if len(virus_indices_to_plot) < 2:
        print("  -> Not enough valid snapshots in the window to draw a line. Skipping plot.")
        return

    resistance_values_to_plot = resistance_col[virus_indices_to_plot]
    virus_times_to_plot = [snapshots[i]['time'] for i in virus_indices_to_plot]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot the resistance profile as a continuous line
    ax.plot(virus_times_to_plot, resistance_values_to_plot, marker='.', linestyle='-', color='black', label='Virus Resistance Profile')
    
    # Add a vertical dotted line for the reference DIP time
    ax.axvline(x=ref_dip_time, color='gray', linestyle=':', linewidth=2.5, label=f'Reference DIP Time (T={ref_dip_time:.0f})')
    
    # Add the baseline for comparison
    ax.axhline(1, color='red', linestyle='--', linewidth=2, label='Baseline (Most Sensitive)')
    
    # Set labels, title, and other aesthetics
    ax.set_yscale('linear')
    ax.set_xlabel("Virus Snapshot Time (T)", fontsize=18)
    ax.set_ylabel("Fold Resistance", fontsize=18)
    # ax.set_title(f"Resistance Profile of Evolving Virus to DIPs from Time T={ref_dip_time:.0f}", fontsize=18)
    
    # Improve readability
    ax.legend(fontsize=14)
    ax.grid(True, which='major', axis='x', alpha=0.8)
    ax.grid(True, which='major', axis='y', alpha=0.8)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xlim(virus_times_to_plot[0], virus_times_to_plot[-1])
    
    plt.tight_layout()
    
    if save_path:
        filename = f"resistance_profile_vs_dip_t{ref_dip_time:.0f}.svg"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  -> Saved plot to {filepath}")
        plt.close(fig)
    else:
        plt.show()

def plot_cross_snapshot_resistance(interference_matrix, snapshots, save_path=None):
    print("  Generating cross-snapshot resistance heatmap...")
    n = len(snapshots)
    snapshot_times = [s['time'] for s in snapshots]
    resistance_matrix = np.zeros_like(interference_matrix)
    for j in range(n):
        interference_col = interference_matrix[:, j]; max_interference = np.max(interference_col)
        if max_interference > 1e-12: resistance_matrix[:, j] = max_interference / (interference_col + 1e-15)
        else: resistance_matrix[:, j] = 1.0
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use the actual time values for the axes
    vmax = np.nanmax(resistance_matrix)
    if vmax <= 1: vmax = 1.1 # Avoid log scale errors if there's no resistance
    im = ax.pcolormesh(snapshot_times, snapshot_times, resistance_matrix, cmap='plasma', norm=mcolors.LogNorm(vmin=1, vmax=vmax))
    
    # Setup plot aesthetics
    ax.plot([snapshot_times[0], snapshot_times[-1]], [snapshot_times[0], snapshot_times[-1]], 'w--', alpha=0.7, label='Contemporary (Time i=j)')
    ax.set_xlabel('DIP Snapshot Time (T)', fontsize=18)
    ax.set_ylabel('Virus Snapshot Time (T)', fontsize=18)
    # ax.set_title('Cross-Snapshot Fold Resistance', fontsize=18)
    ax.legend(fontsize=14)
    ax.set_aspect('equal')
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add colorbar with explicit ticks for powers of 10
    log_max_power = np.ceil(np.log10(vmax))
    tick_values = [10**i for i in range(int(log_max_power) + 1)]
    if not tick_values:
        cbar = fig.colorbar(im, ax=ax) # Fallback to default
    else:
        cbar = fig.colorbar(im, ax=ax, ticks=tick_values)
        
    cbar.set_label('Fold Resistance (Log Scale)', fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    
    if save_path:
        filename = "cross_snapshot_resistance_heatmap.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        print(f"  -> Saved plot to {filepath}")
        plt.close(fig)
    else:
        plt.show()