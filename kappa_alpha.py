import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import concurrent.futures
from tqdm import tqdm
from modified_FFT import sol_dip_virus_pde_fft
import os

def run_single_simulation(task):
    """Worker function for multiprocessing pool."""
    params, ij_indices = task
    # *** IMPORTANT: Replace this dummy function with your real solver ***
    result = sol_dip_virus_pde_fft(params)
    # result = sol_dip_virus_pde_fft(params) 
    return result, ij_indices

def sweep_parameters_multiprocess(alpha_list, kappa_list, override_params=None):
    """
    Run the solver across alpha_list x kappa_list in parallel, returning a grid of results.
    """
    if override_params is None: override_params = {}
    
    tasks = []
    for i, a in enumerate(alpha_list):
        for j, k in enumerate(kappa_list):
            params = override_params.copy()
            params.update({'alpha': float(a), 'kappa': float(k)})
            tasks.append((params, (i, j)))

    results_grid = np.empty((len(alpha_list), len(kappa_list)), dtype=object)
    
    # Use ProcessPoolExecutor for true CPU parallelism
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Use tqdm for a progress bar
        future_to_task = {executor.submit(run_single_simulation, task): task for task in tasks}
        for future in tqdm(concurrent.futures.as_completed(future_to_task), total=len(tasks), desc="Running Simulations"):
            try:
                result, (i, j) = future.result()
                results_grid[i, j] = result
            except Exception as exc:
                print(f'A simulation generated an exception: {exc}')

    return results_grid

def summarize_endstate(res, d_thresh=1.0, final_frac=0.1):
    """Calculates summary metrics from a single simulation result."""
    # Ensure all required keys exist, return NaNs otherwise
    required_keys = ['time_points', 'V_total_time', 'params', 'shape_stats_v']
    if not all(k in res for k in required_keys) or 'mean_distance_from_origin' not in res['shape_stats_v']:
        return {'X_extent': np.nan, 'E_escape': np.nan, 'S_suppress': np.nan}

    t = res['time_points']
    V = res['V_total_time']
    K = res['params']['K_cap']
    d = res['shape_stats_v']['mean_distance_from_origin']
    
    # Handle cases where simulation might have failed and returned empty lists/arrays
    if len(t) == 0 or len(V) == 0 or len(d) == 0:
        return {'X_extent': np.nan, 'E_escape': np.nan, 'S_suppress': np.nan}
        
    n = max(1, int(len(t) * final_frac))
    d_final_mean = float(np.nanmean(d[-n:]))
    S = float(np.log10(max(np.nanmean(V[-n:]), 1e-12) / K))
    E = int(d_final_mean >= d_thresh)  # 1 = escape present at end-state
    return {'X_extent': d_final_mean, 'E_escape': E, 'S_suppress': S}

# --- NEW PLOTTING FUNCTIONS ---

def plot_line_trajectories_dual_panel(alpha_list, kappa_list, X_grid, S_grid, outpath=None):
    """
    Figure 3A: Two-panel line plot showing log population and escape vs κ, 
    with separate line styles and markers for different α values (black and white).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Select a subset of α values for clarity (3-5 representative curves)
    alpha_indices = np.linspace(0, len(alpha_list)-1, min(5, len(alpha_list)), dtype=int)
    
    # Define line styles and markers for black and white plotting
    line_styles = [':', (0, (3, 1, 1, 1)), '-.', '--', '-']  # solid, dashed, dotted, dash-dot, custom
    
    for idx, i in enumerate(alpha_indices):
        alpha_val = alpha_list[i]
        linestyle = line_styles[idx % len(line_styles)]
        
        # Filter out NaN values for clean lines
        mask = np.isfinite(S_grid[i, :]) & np.isfinite(X_grid[i, :])
        if np.sum(mask) < 2:  # Need at least 2 points for a line
            continue
            
        kappa_masked = np.array(kappa_list)[mask]
        S_masked = S_grid[i, :][mask]
        X_masked = X_grid[i, :][mask]
        
        # Panel 1: Log population vs κ
        ax1.plot(kappa_masked, S_masked, color='black', linestyle=linestyle,
                label=f'α = {alpha_val:.3f}', linewidth=2)
        
        # Panel 2: Escape extent vs κ  
        ax2.plot(kappa_masked, X_masked, color='black', linestyle=linestyle,
                label=f'α = {alpha_val:.3f}', linewidth=2)
    
    # Configure panel 1
    ax1.set_xscale('log')
    ax1.set_xlabel('κ (interference strength)')
    ax1.set_ylabel('Virus Population (log₁₀(V_final))')
    ax1.set_title('Virus Population vs κ')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Configure panel 2
    ax2.set_xscale('log')
    ax2.set_xlabel('κ (interference strength)')
    ax2.set_ylabel('Escape Extent (final mean distance)')
    ax2.set_title('Escape Extent vs κ')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.suptitle('Parameter Dependencies: How α Affects the κ Response')
    fig.tight_layout()
    if outpath: 
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
    return fig, (ax1, ax2)

def plot_tradeoff_parametric_curves(alpha_list, kappa_list, X_grid, S_grid, outpath=None):
    """
    Figure 3B: Trade-off plot showing log population vs escape extent,
    with parametric curves traced by κ for each α value (black and white).
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Select representative α values
    alpha_indices = np.linspace(0, len(alpha_list)-1, min(5, len(alpha_list)), dtype=int)
    
    # Define line styles and markers for black and white plotting
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'v', 'D']
    
    for idx, i in enumerate(alpha_indices):
        alpha_val = alpha_list[i]
        linestyle = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        
        # Filter out NaN values
        mask = np.isfinite(S_grid[i, :]) & np.isfinite(X_grid[i, :])
        if np.sum(mask) < 2:
            continue
            
        S_masked = S_grid[i, :][mask]
        X_masked = X_grid[i, :][mask]
        kappa_masked = np.array(kappa_list)[mask]
        
        # Sort by κ to ensure proper curve tracing
        sort_idx = np.argsort(kappa_masked)
        S_sorted = S_masked[sort_idx]
        X_sorted = X_masked[sort_idx]
        kappa_sorted = kappa_masked[sort_idx]
        
        # Plot the parametric curve (κ parameter implicit)
        ax.plot(X_sorted, S_sorted, linestyle=linestyle, marker=marker, color='black',
               label=f'α = {alpha_val:.3f}', linewidth=2.5, markersize=4, alpha=0.8, markevery=5)
        
        # Add arrow to show direction of increasing κ
        if len(X_sorted) > 1:
            # Arrow at midpoint
            mid_idx = len(X_sorted) // 2
            dx = X_sorted[mid_idx+1] - X_sorted[mid_idx-1] if mid_idx > 0 and mid_idx < len(X_sorted)-1 else 0
            dy = S_sorted[mid_idx+1] - S_sorted[mid_idx-1] if mid_idx > 0 and mid_idx < len(S_sorted)-1 else 0
            if dx != 0 or dy != 0:
                ax.annotate('', xy=(X_sorted[mid_idx] + 0.1*dx, S_sorted[mid_idx] + 0.1*dy),
                           xytext=(X_sorted[mid_idx], S_sorted[mid_idx]),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax.set_xlabel('Escape Extent (final mean distance)')
    ax.set_ylabel('Log Population (log₁₀(V_final/K))')
    ax.set_title('Trade-off Landscape: Log Population vs Escape\n(curves trace κ, arrows show increasing κ)')
    ax.grid(True, alpha=0.3)
    ax.legend(title='α (fitness penalty)', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    fig.tight_layout()
    if outpath: 
        fig.savefig(outpath, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_escape_extent_map_updated(alpha_list, kappa_list, X_grid, S_grid, E_grid, outpath=None):
    """Figure 1: Heatmap of escape extent with log population contours and no-escape hatching."""
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(X_grid, origin='lower', aspect='auto', cmap='viridis',
                   extent=[0, len(kappa_list)-1, 0, len(alpha_list)-1])
    # Log population contours:
    X_mesh, Y_mesh = np.meshgrid(np.arange(len(kappa_list)), np.arange(len(alpha_list)))
    levels = [-2, -1, -0.5, 0]  # adjust for your regime
    cs = ax.contour(X_mesh, Y_mesh, S_grid, levels=levels, colors='white', linewidths=1.5)
    ax.clabel(cs, fmt=lambda v: f"LogPop={v:.1f}", inline=True, fontsize=9)
    # Hatch where no escape (E=0):
    ax.contourf(X_mesh, Y_mesh, 1-E_grid, levels=[0.5, 1.5], hatches=['///'], alpha=0, colors='none')
    
    ax.set_xticks(np.arange(len(kappa_list)))
    ax.set_xticklabels([f"{k:.0e}" for k in kappa_list], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(alpha_list)))
    ax.set_yticklabels([f"{a:.3f}" for a in alpha_list])
    ax.set_xlabel('κ (interference strength)'); ax.set_ylabel(r'$\alpha$ (fitness penalty)')
    ax.set_title('Escape Extent (heat) & Log Population (contours)')
    cbar = fig.colorbar(im, ax=ax, pad=0.02); cbar.set_label(r'Escape Extent (final mean distance)')
    fig.tight_layout()
    if outpath: fig.savefig(outpath, dpi=300, bbox_inches='tight')
    return fig, ax

def plot_safe_kappa_bands_updated(alpha_list, kappa_list, E_grid, S_grid, S_target=-1.0, outpath=None):
    """Figure 2: Horizontal bands showing the 'safe' kappa window for each alpha."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, a in enumerate(alpha_list):
        # Indices where no escape and adequate log population
        ok = (E_grid[i,:] == 0) & (S_grid[i,:] <= S_target)
        # Find contiguous spans in log-κ index space
        j = 0
        while j < len(kappa_list):
            if ok[j]:
                j0 = j
                while j < len(kappa_list) and ok[j]: j += 1
                j1 = j - 1
                ax.hlines(y=a, xmin=kappa_list[j0], xmax=kappa_list[j1],
                          linewidth=8, alpha=0.8, color='C0')
            else:
                j += 1
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_yticks(alpha_list); ax.set_yticklabels([f"{a:.3f}" for a in alpha_list])
    ax.set_xlabel('κ (interference strength)'); ax.set_ylabel('α (fitness penalty)')
    ax.set_title(f'Safe κ Windows per α (No Escape & Log Population ≤ {S_target})')
    ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.5)
    fig.tight_layout()
    if outpath: fig.savefig(outpath, dpi=300, bbox_inches='tight')
    return fig, ax

# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    # 1. Define simulation parameters
    alpha_list = np.array([1e-2, 5e-2, 1e-1, 5e-1, 1])
    kappa_list = np.logspace(-9, -6.5, 200)
    output_dir = 'all_runs/threshold_pareto'
    # For quick previews, override spatial/time resolution:
    override = {}

    # 2. Run the parameter sweep (in parallel)
    print("Starting parameter sweep...")
    results_grid = sweep_parameters_multiprocess(alpha_list, kappa_list, override_params=override)
    print("Parameter sweep complete.")

    # 3. Calculate summary metrics for all runs
    X_grid = np.full(results_grid.shape, np.nan)
    E_grid = np.zeros_like(X_grid)
    S_grid = np.full_like(X_grid, np.nan)

    for i in range(len(alpha_list)):
        for j in range(len(kappa_list)):
            r = results_grid[i, j]
            if r is None or not r.get('success', False): continue
            summ = summarize_endstate(r, d_thresh=1.0, final_frac=0.2) # Using last 20%
            X_grid[i, j] = summ['X_extent']
            E_grid[i, j] = summ['E_escape']
            S_grid[i, j] = summ['S_suppress']

    # 4. Generate all figures
    print("\nGenerating figures...")
   
    # New line trajectory visualizations
    plt.rcParams['svg.fonttype'] = 'none'

    plot_line_trajectories_dual_panel(alpha_list, kappa_list, X_grid, S_grid, 
                                     outpath=os.path.join(output_dir,'fig3a_line_trajectories.svg'))
    plot_tradeoff_parametric_curves(alpha_list, kappa_list, X_grid, S_grid, 
                                   outpath=os.path.join(output_dir,'fig3b_tradeoff_curves.png'))
    
    print("All figures generated and saved to PNG files.")
    plt.show()
