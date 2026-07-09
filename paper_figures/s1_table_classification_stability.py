"""
S1 Table diagnostic: classification stability.

Visualize where convergence mismatches (especially Chase -> Coexistence)
fall in parameter space to check if they cluster near regime boundaries.
Reads the LHS sweep output produced by fig4_latin_hypercube.py.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# --- Load data ---
df = pd.read_pickle('all_runs/lhs/lhs_10000/10000.pkl')
conv = df[df['convergence_match'].notna()].copy()
conv['transition'] = conv['outcome'] + ' → ' + conv['outcome_2T'].astype(str)

matched = conv[conv['convergence_match'].astype(bool)]
mismatched = conv[~conv['convergence_match'].astype(bool)]
chase_to_coex = mismatched[
    (mismatched['outcome'] == 'Chase') &
    (mismatched['outcome_2T'] == 'Static Coexistence')
]

sweep_params = ['kappa', 'eta', 'alpha', 'beta', 'gamma', 'mu']
log_params = ['kappa', 'eta', 'mu', 'alpha', 'beta']

param_labels = {
    'kappa': r'$\kappa$ (interference)',
    'eta':   r'$\eta$ (de novo generation)',
    'alpha': r'$\alpha$ (centering)',
    'beta':  r'$\beta$ (aggregation)',
    'gamma': r'$\gamma$ (decay)',
    'mu':    r'$\mu$ (mutation)'
}

save_dir = 'all_runs/lhs/lhs_10000'
os.makedirs(os.path.join(save_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(save_dir, 'svg'), exist_ok=True)

# =========================================================================
# Panel 1: 1D marginal distributions — where do Chase→Coex mismatches sit?
# =========================================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.ravel()

for idx, param in enumerate(sweep_params):
    ax = axes[idx]
    is_log = param in log_params

    # All convergence samples (background)
    stable_chase = matched[matched['outcome'] == 'Chase']
    stable_coex = matched[matched['outcome'] == 'Static Coexistence']

    vals_chase = stable_chase[param].values
    vals_coex = stable_coex[param].values
    vals_mismatch = chase_to_coex[param].values

    if is_log:
        bins = np.logspace(
            np.log10(conv[param].min()),
            np.log10(conv[param].max()), 30)
    else:
        bins = np.linspace(conv[param].min(), conv[param].max(), 30)

    ax.hist(vals_chase, bins=bins, alpha=0.35, color='C0',
            label=f'Stable Chase (n={len(vals_chase)})', density=True)
    ax.hist(vals_coex, bins=bins, alpha=0.35, color='C1',
            label=f'Stable Coexistence (n={len(vals_coex)})', density=True)
    ax.hist(vals_mismatch, bins=bins, alpha=0.8, color='C3',
            label=f'Chase→Coex mismatch (n={len(vals_mismatch)})',
            density=True, histtype='step', linewidth=2)

    ax.set_xlabel(param_labels.get(param, param))
    ax.set_ylabel('Density')
    if is_log:
        ax.set_xscale('log')
    if idx == 0:
        ax.legend(fontsize=7, loc='upper right')

fig.suptitle(
    'Chase→Coexistence Mismatches: Marginal Distributions\n'
    '(Do mismatches cluster near the Chase/Coexistence boundary?)',
    fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'png', 'mismatch_marginals.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(save_dir, 'svg', 'mismatch_marginals.svg'),
            bbox_inches='tight')
plt.close(fig)

# =========================================================================
# Panel 2: Pairwise scatter — κ vs each other param, colored by mismatch
# =========================================================================
other_params = [p for p in sweep_params if p != 'kappa']
fig, axes = plt.subplots(1, len(other_params), figsize=(20, 4))

for idx, param in enumerate(other_params):
    ax = axes[idx]

    # Background: all convergence samples
    ax.scatter(conv['kappa'], conv[param],
               c='0.85', s=4, alpha=0.4, label='Matched', rasterized=True)

    # Overlay: all mismatches (any type)
    other_mm = mismatched[
        ~((mismatched['outcome'] == 'Chase') &
          (mismatched['outcome_2T'] == 'Static Coexistence'))
    ]
    ax.scatter(other_mm['kappa'], other_mm[param],
               c='C0', s=15, alpha=0.6, marker='s',
               label='Other mismatch', zorder=3)

    # Highlight: Chase → Coexistence
    ax.scatter(chase_to_coex['kappa'], chase_to_coex[param],
               c='C3', s=20, alpha=0.8, marker='o', edgecolors='k',
               linewidths=0.3, label='Chase→Coex', zorder=4)

    ax.set_xscale('log')
    if param in log_params:
        ax.set_yscale('log')
    ax.set_xlabel(param_labels['kappa'])
    ax.set_ylabel(param_labels.get(param, param))
    if idx == 0:
        ax.legend(fontsize=7, loc='best')

fig.suptitle(
    'Mismatches in Parameter Space (κ vs others)\n'
    'Red = Chase→Coexistence transitions',
    fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'png', 'mismatch_kappa_scatter.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(save_dir, 'svg', 'mismatch_kappa_scatter.svg'),
            bbox_inches='tight')
plt.close(fig)

# =========================================================================
# Panel 3: κ-focused view — CDF / strip showing boundary concentration
# =========================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})

# Top: stacked histogram of stable outcomes + mismatch overlay
kappa_bins = np.logspace(np.log10(conv['kappa'].min()),
                         np.log10(conv['kappa'].max()), 25)

ax1.hist(stable_chase['kappa'], bins=kappa_bins, alpha=0.4, color='C0',
         label='Stable Chase', density=False)
ax1.hist(stable_coex['kappa'], bins=kappa_bins, alpha=0.4, color='C1',
         label='Stable Coexistence', density=False)
ax1.hist(chase_to_coex['kappa'], bins=kappa_bins, alpha=0.9, color='C3',
         label='Chase→Coex mismatch', density=False,
         histtype='step', linewidth=2.5)
ax1.set_xscale('log')
ax1.set_ylabel('Count')
ax1.legend(fontsize=9)
ax1.set_title('Where do Chase→Coexistence transitions occur along κ?')

# Bottom: strip plot of individual mismatch κ values
ax2.scatter(chase_to_coex['kappa'],
            np.random.uniform(0.2, 0.8, size=len(chase_to_coex)),
            c='C3', s=30, alpha=0.7, edgecolors='k', linewidths=0.3)
ax2.set_xscale('log')
ax2.set_xlabel(param_labels['kappa'])
ax2.set_ylabel('')
ax2.set_yticks([])
ax2.set_ylim(0, 1)

fig.tight_layout()
fig.savefig(os.path.join(save_dir, 'png', 'mismatch_kappa_focus.png'),
            dpi=300, bbox_inches='tight')
fig.savefig(os.path.join(save_dir, 'svg', 'mismatch_kappa_focus.svg'),
            bbox_inches='tight')
plt.close(fig)

# =========================================================================
# Print summary statistics
# =========================================================================
print("=" * 60)
print("MISMATCH ANALYSIS SUMMARY")
print("=" * 60)
print(f"Total convergence samples: {len(conv)}")
print(f"Total mismatches: {len(mismatched)} ({len(mismatched)/len(conv)*100:.1f}%)")
print(f"Chase→Coexistence: {len(chase_to_coex)}")
print()

print("Chase→Coex κ statistics:")
print(f"  median: {chase_to_coex['kappa'].median():.3f}")
print(f"  mean:   {chase_to_coex['kappa'].mean():.3f}")
print(f"  range:  [{chase_to_coex['kappa'].min():.3f}, {chase_to_coex['kappa'].max():.3f}]")
print(f"  IQR:    [{chase_to_coex['kappa'].quantile(0.25):.3f}, {chase_to_coex['kappa'].quantile(0.75):.3f}]")
print()

print("For comparison — stable Chase κ stats:")
print(f"  median: {stable_chase['kappa'].median():.3f}")
print(f"  IQR:    [{stable_chase['kappa'].quantile(0.25):.3f}, {stable_chase['kappa'].quantile(0.75):.3f}]")
print()
print("Stable Coexistence κ stats:")
print(f"  median: {stable_coex['kappa'].median():.3f}")
print(f"  IQR:    [{stable_coex['kappa'].quantile(0.25):.3f}, {stable_coex['kappa'].quantile(0.75):.3f}]")
print()

# Quick check: are mismatches between the two regimes?
chase_med = stable_chase['kappa'].median()
coex_med = stable_coex['kappa'].median()
mm_med = chase_to_coex['kappa'].median()
lo, hi = sorted([chase_med, coex_med])
between = ((chase_to_coex['kappa'] >= lo) & (chase_to_coex['kappa'] <= hi)).sum()
print(f"Mismatches with κ between stable Chase median ({lo:.2f}) and Coex median ({hi:.2f}): "
      f"{between}/{len(chase_to_coex)} ({between/len(chase_to_coex)*100:.0f}%)")

print("\nPlots saved to:", save_dir)
