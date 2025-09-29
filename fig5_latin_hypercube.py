import os
import time
import numpy as np
import pandas as pd
from scipy.stats import qmc
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from pdesolver import sol_dip_virus_pde_fft, default_params


def classify_outcome(result, params, chase_std_dev_threshold=0.1):
    """
    Classify the outcome of a simulation run.

    Parameters
    ----------
    result : dict
        Output from `sol_dip_virus_pde_fft`.
    params : dict
        The parameters used for the simulation run.
    chase_std_dev_threshold : float, optional
        Standard deviation threshold of V-D distance to classify as 'Chase'.

    Returns
    -------
    str
        Outcome category: 'Chase', 'Static Coexistence', 'DIP Extinction',
        'Coextinction'.
    """
    # Check for early virus extinction based on solver termination
    if result['termination_reason'] == 'Virus Extinction':
        return 'Coextinction'

    # Get extinction thresholds from the simulation parameters
    ext_V = params.get('extinction_threshold_V', 1)
    ext_D = params.get('extinction_threshold_D', 1)

    # Check final populations against the specified thresholds
    final_V, final_D = result['V_total_time'][-1], result['D_total_time'][-1]
    if final_V < ext_V:
        return 'Coextinction'
    elif final_D < ext_D:
        return 'DIP Extinction'

    # Analyze coexistence dynamics (second half of simulation only)
    num_points = len(result['time_points'])
    start_idx = num_points // 2
    if start_idx < 2:
        return 'Static Coexistence (Short Sim)'

    v_mean = result['mean_phenotype_v'][start_idx:]
    d_mean = result['mean_phenotype_d'][start_idx:]
    distance = np.sqrt(np.sum((v_mean - d_mean) ** 2, axis=1))

    if np.isnan(distance).any():
        return 'Static Coexistence'

    return 'Chase' if np.std(distance) > chase_std_dev_threshold else 'Static Coexistence'


def run_simulation_worker(params_dict):
    """
    Run a single simulation with given parameters.

    Returns
    -------
    pandas.Series
        Parameters and outcome, plus final virus and DIP counts.
    """
    try:
        result = sol_dip_virus_pde_fft(params_dict)
        # Pass the params_dict to the classifier
        outcome = classify_outcome(result, params_dict)

        output = pd.Series(params_dict)
        output['outcome'] = outcome
        output['final_V'] = result['V_total_time'][-1]
        output['final_D'] = result['D_total_time'][-1]
        return output
    except Exception as e:
        print(f"[ERROR] Worker failed with params {params_dict}: {e}")
        return None


def run_lhs_sweep(param_ranges, n_samples, base_params=None, log_params=None, n_cpu=None):
    """
    Perform Latin Hypercube Sampling (LHS) sweep over parameter space.
    """
    base_params = base_params or default_params.copy()
    log_params = log_params or []
    n_cpu = n_cpu or os.cpu_count()

    print(f"[INFO] Starting LHS sweep with {n_samples} samples on {n_cpu} cores.")

    sampler = qmc.LatinHypercube(d=len(param_ranges), seed=42)
    unit_samples = sampler.random(n=n_samples)

    all_sim_params = []
    for sample in unit_samples:
        params = base_params.copy()
        for (p_name, (p_min, p_max)), sample_val in zip(param_ranges.items(), sample):
            if p_name in log_params:
                scaled_val = 10 ** (np.log10(p_min) + sample_val * (np.log10(p_max) - np.log10(p_min)))
            else:
                scaled_val = p_min + sample_val * (p_max - p_min)
            params[p_name] = scaled_val
        all_sim_params.append(params)

    start_time = time.time()
    with multiprocessing.Pool(processes=n_cpu) as pool:
        results_list = pool.map(run_simulation_worker, all_sim_params)
    print(f"[INFO] Sweep finished in {(time.time() - start_time) / 60:.2f} minutes.")

    return pd.DataFrame([res for res in results_list if res is not None])


def plot_lhs_results(df, sweep_params, log_params, save_dir, n_bins=10, use_hatch=False):
    """
    Greyscale stacked outcome proportions with direct labels placed slightly
    inside the right edge so the stack fills exactly to the edges.
    """
    outcome_order = ['Chase', 'Static Coexistence', 'DIP Extinction', 'Coextinction']
    
    param_xlabels = {
        'kappa': r'Interference Strength ($\kappa$)',
        'eta':   r'De novo Generation Rate ($\eta$)',
        'alpha': r'Centering Term ($\alpha$)',
        'beta':  r'Aggregation Term ($\beta$)',
        'gamma': r'Decay Rate ($\gamma$)',
        'mu':    r'Mutation Rate ($\mu$)'
    }

    cmap = cm.get_cmap('Greys')
    greys = [cmap(v) for v in np.linspace(0.85, 0.25, len(outcome_order))]

    df['outcome'] = pd.Categorical(df['outcome'], categories=outcome_order, ordered=True)

    png_save_dir = os.path.join(save_dir, 'png')
    svg_save_dir = os.path.join(save_dir, 'svg')
    os.makedirs(png_save_dir, exist_ok=True)
    os.makedirs(svg_save_dir, exist_ok=True)

    print(f"[INFO] Generating and saving greyscale labeled plots to {png_save_dir} and {svg_save_dir}...")

    hatch_patterns = ['///', '\\\\\\', 'xxx', '...']

    for param_name in sweep_params:
        fig, ax = plt.subplots(figsize=(6, 5))

        data_min, data_max = df[param_name].min(), df[param_name].max()
        if param_name in log_params:
            bins = np.logspace(np.log10(data_min), np.log10(data_max), n_bins + 1)
            ax.set_xscale('log')
            log_min = np.floor(np.log10(data_min))
            log_max = np.ceil(np.log10(data_max))
            major_ticks = 10.0 ** np.arange(log_min, log_max + 1)
            plot_min, plot_max = bins[0], bins[-1]
            visible_ticks = major_ticks[(major_ticks >= plot_min) & (major_ticks <= plot_max)]
            if len(visible_ticks) > 0:
                ax.set_xticks(visible_ticks)
                ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        else:
            bins = np.linspace(data_min, data_max, n_bins + 1)

        df['bin'] = pd.cut(df[param_name], bins, right=False, labels=bins[:-1])
        proportions = df.groupby('bin', observed=False)['outcome'].value_counts(normalize=True).unstack(fill_value=0)
        for out in outcome_order:
            if out not in proportions.columns:
                proportions[out] = 0.0
        proportions = proportions[outcome_order]

        # --- FIX & REFACTOR ---
        # Calculate bin_centers once and use it for both plot_x and annotations
        if param_name in log_params: # <<<< THIS IS THE CORRECTED LINE
            bin_centers = np.sqrt(bins[:-1] * bins[1:])
        else:
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
        # --- END FIX ---

        plot_x = np.concatenate(([bins[0]], bin_centers, [bins[-1]]))
        y_vals = [proportions[o].values for o in outcome_order]
        plot_y = [np.concatenate(([y[0]], y, [y[-1]])) for y in y_vals]
        
        # Note: calling stackplot twice is unnecessary. The returned polys can be used directly.
        polys = ax.stackplot(plot_x, plot_y, colors=greys, alpha=0.95)

        if use_hatch:
            for poly, hatch in zip(polys, hatch_patterns):
                poly.set_hatch(hatch)
                poly.set_edgecolor('k')
                poly.set_linewidth(0.25)
        else:
            for poly in polys:
                poly.set_edgecolor('none')

        xlabel_text = param_xlabels.get(param_name, param_name)
        ax.set_xlabel(xlabel_text, fontsize=12)
        ax.set_ylabel("Proportion of Outcomes", fontsize=12)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

        # Position for annotation text labels
        if param_name in log_params:
            log_a, log_b = np.log10(bins[0]), np.log10(bins[-1])
            x_text = 10 ** (log_b - 0.05 * (log_b - log_a))
        else:
            x_text = bins[-1] - 0.05 * (bins[-1] - bins[0])

        for i, outcome in enumerate(outcome_order):
            arr = proportions[outcome].values
            if np.all(np.isnan(arr)) or arr.sum() == 0:
                continue

            peak_idx = int(np.nanargmax(arr))
            peak_val = arr[peak_idx]
            idx_for_label = -1 if peak_val < 1e-3 else peak_idx

            x_anchor = bin_centers[idx_for_label]
            row = proportions.iloc[idx_for_label]
            bottom_before = row.iloc[:i].sum()
            center_y = bottom_before + row.iloc[i] / 2.0
            
            ha = 'right' if x_text < x_anchor else 'left' 
            
            ax.annotate(outcome,
                        xy=(x_anchor, center_y),
                        xytext=(x_text, center_y),
                        va='center', ha=ha,
                        fontsize=9, weight='bold',
                        arrowprops=dict(arrowstyle='-', lw=0.6, color='0.35', shrinkA=5, shrinkB=2))

        plt.tight_layout()
        filename = f"lhs_sensitivity_{param_name}"
        fig_path_png = os.path.join(png_save_dir, f"{filename}.png")
        fig_path_svg = os.path.join(svg_save_dir, f"{filename}.svg")

        fig.savefig(fig_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(fig_path_svg, bbox_inches='tight')
        plt.close(fig)

    print("[INFO] All individual greyscale labeled plots saved.")


def generate_legend_svg(save_dir):
    """
    Generates and saves a standalone SVG legend for the plots into the svg/ subdirectory.
    """
    outcome_order = ['Chase', 'Static Coexistence', 'DIP Extinction', 'Coextinction']
    outcome_colors = dict(zip(outcome_order, sns.color_palette("viridis", len(outcome_order))))

    legend_handles = [mpatches.Patch(color=color, label=label)
                      for label, color in outcome_colors.items()]

    fig_legend, ax_legend = plt.subplots(figsize=(8, 0.5))
    
    ax_legend.legend(handles=legend_handles, ncol=len(outcome_order),
                     mode="expand", frameon=False, loc='center')
    
    ax_legend.axis('off')
    
    svg_save_dir = os.path.join(save_dir, 'svg')
    png_save_dir = os.path.join(save_dir, 'png')
    os.makedirs(svg_save_dir, exist_ok=True)
    os.makedirs(png_save_dir, exist_ok=True)
    legend_path = os.path.join(svg_save_dir, "lhs_sensitivity_legend.svg")
    png_legend_path = os.path.join(png_save_dir, "lhs_sensitivity_legend.png")

    fig_legend.savefig(legend_path, bbox_inches='tight')
    fig_legend.savefig(png_legend_path, bbox_inches='tight')

    plt.close(fig_legend)


if __name__ == '__main__':
    # Sweep configuration
    param_ranges_to_sweep = {
        'kappa': (9e-2, 1.1e2),
        'eta':   (9e-5, 1.1e-1),
        'alpha': (9e-4, 1.1),
        'beta': (9e-4, 1.1),
        'gamma': (0, 0.91),
        'mu': (9e-5, 1.1e-2)
    }
    log_scale_params = ['kappa', 'eta', 'mu', 'alpha', 'beta']
    num_samples = 10004
    base_simulation_params = default_params.copy()
    base_simulation_params.update({
        "T": 150,
        "r_v": 1,
        "K_cap": 1,
        "sigma": 1,
        "V0_total": 0.01,
        "D0_total": 0.01,
        "extinction_threshold_V": 1e-6,
        "extinction_threshold_D": 1e-6
    })

    results_dir = f"all_runs/lhs/lhs_{num_samples}"
    results_filename = os.path.join(results_dir, f"{num_samples}.pkl")
    os.makedirs(results_dir, exist_ok=True)

    if os.path.exists(results_filename):
        print(f"[INFO] Loading results from {results_filename}")
        results_df = pd.read_pickle(results_filename)
    else:
        results_df = run_lhs_sweep(param_ranges_to_sweep, num_samples,
                                   base_params=base_simulation_params,
                                   log_params=log_scale_params,
                                   n_cpu=multiprocessing.cpu_count() - 1)
        # results_df.to_pickle(results_filename)
        print(f"[INFO] Results saved to {results_filename}")

    plot_save_dir = os.path.dirname(results_filename)
    
    plot_lhs_results(results_df, list(param_ranges_to_sweep.keys()), log_scale_params, save_dir=plot_save_dir)
    
   # generate_legend_svg(save_dir=plot_save_dir)