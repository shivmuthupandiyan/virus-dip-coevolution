# Coevolutionary dynamics of viruses and their defective interfering particles

This repository contains the source code and simulation framework for the paper:

**Coevolutionary dynamics of viruses and their defective interfering particles**  
*Shiv Muthupandiyan, John Yin*  
Wisconsin Institute for Discovery, Chemical and Biological Engineering, University of Wisconsin-Madison.  
**DOI:** [10.1371/journal.pcbi.1014300](https://doi.org/10.1371/journal.pcbi.1014300)

## Overview

Defective interfering particles (DIPs) are viral mutants that arise naturally during infection. Because they lack essential functions, they parasitize intact viruses during co-infection. This project implements the mathematical model described in the paper above—a continuous phenotype-space model using coupled partial differential equations (PDEs)—to explore the evolutionary interplay between viruses and DIPs.

Unlike traditional strong-selection models, this framework captures **strong-mutation regimes** where both populations diffuse through trait space.


## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the environment and install all dependencies
uv sync
```

Run scripts and notebooks inside the managed environment with `uv run`:

```bash
uv run python paper_figures/fig4_latin_hypercube.py
uv run jupyter lab figures.ipynb
```

## Usage

### Basic Example

```python
from pdesolver import sol_dip_virus_pde_fft, default_params

# Run with baseline parameters (pass {} to use all defaults)
results = sol_dip_virus_pde_fft({})

# Total virus / DIP populations over time
t = results['time_points']
V = results['V_total_time']
D = results['D_total_time']
```

### Custom Parameters

```python
# Define custom parameters (only specify what you want to change)
params = {
    'T': 200,           # Run for 200 time steps
    'D0_total': 1e5,    # Start with 100,000 DIPs
    'kappa': 1e-7,      # Interference strength
    'mu': 1e-2          # Higher mutation rate
}

# Run simulation with full density field output
results = sol_dip_virus_pde_fft(params, save_density_series=True)
```

### Output Structure

The solver returns a dictionary containing:

- `time_points` — Time points
- `V_total_time`, `D_total_time` — Total virus and DIP population sizes over time
- `mean_phenotype_v`, `mean_phenotype_d` — Mean phenotype positions (x, y coordinates), shape `(T, 2)`
- `mean_dist_from_origin_v` — Mean virus distance from the fitness optimum over time
- `termination_reason`, `success` — Solver status
- `v_dist_time`, `d_dist_time`, `PHENOx`, `PHENOy` — (Optional, when `save_density_series=True`) full spatial density fields and phenotype coordinate grids

## Model Parameters

The model is controlled by a parameter dictionary with the following options:

### Biological Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `mu` | 4e-3 | Mutation rate (phenotypic diffusion) |
| `gamma` | 0.3 | Population decay rate |
| `eta` | 1e-3 | De novo generation rate |
| `r_v` | 1 | Base viral replication rate |
| `alpha` | 0.025 | Centering term (fitness landscape) |
| `beta` | 0.025 | Aggregation term (fitness landscape) |
| `K_cap` | 1e8 | Carrying capacity |
| `kappa` | 1e-7 | Interference strength (scaled inversely with K_cap) |
| `sigma` | 1 | Interference kernel standard deviation |

### Initial Conditions
| Parameter | Default | Description |
|-----------|---------|-------------|
| `V0_total` | 1e6 | Initial virus population size |
| `v_init_pos` | [1, 1] | Initial virus position in phenotype space |
| `D0_total` | 1e6 | Initial DIP population size |
| `d_init_pos` | [1.4, 0] | Initial DIP position in phenotype space |
| `init_spread` | 0.4 | Initial population standard deviation |

### Simulation Control
| Parameter | Default | Description |
|-----------|---------|-------------|
| `nx` | 50 | Grid resolution (number of rows/columns) |
| `bord` | 5 | Domain boundaries (spans [-bord, bord]) |
| `T` | 100 | Number of time steps to simulate |
| `monitor_extinction` | True | Stop simulation if virus goes extinct |
| `extinction_threshold_V` | 1e2 | Virus extinction threshold |
| `extinction_threshold_D` | 1e2 | DIP extinction threshold |

## Project Structure

```
├── pdesolver.py              # Core PDE solver module
├── figures.ipynb             # Main notebook: Figs 1-3 (dynamics, heatmaps) + Fig 5C, 6C panels
├── pyproject.toml            # Project metadata and dependencies (uv)
├── uv.lock                   # Pinned dependency versions
├── README.md                 # This file
└── paper_figures/            # Publication figure generation scripts
    ├── fig4_latin_hypercube.py               # Fig 4: global sensitivity analysis (LHS) + S1 Table
    ├── fig5_cross_passage.py                 # Fig 5: time-shift / fold-resistance analysis
    ├── fig6_escape.py                        # Fig 6: escape vs. fitness-cost trade-off
    ├── s1_table_classification_stability.py  # S1 Table: LHS classification-stability diagnostic
    ├── s1_ndim.ipynb                         # S1 Fig: higher-dimensional model
    └── s2_kernels.ipynb                      # S2 Fig: alternative interference kernels
```

## Mathematical Framework

The model solves the following dimensionless system:

$$ \frac{\partial v}{\partial t} = \mu \nabla^2 v + [1 - \alpha|x|^2 - \beta|x - \bar{x}_V|^2 - \eta - I_V(x,t)]L(t)v - \gamma v $$

$$ \frac{\partial d}{\partial t} = \mu \nabla^2 d + [I_D(x,t)d + \eta v]L(t) - \gamma d $$

Where $I_V$ and $I_D$ represent the phenotype-dependent interference cost and benefit, calculated via convolution integrals over the trait space.

## Citation

If you use this code in your research, please cite:

> Muthupandiyan S, Yin J (2026). **Coevolutionary dynamics of viruses and their defective interfering particles**. *PLOS Computational Biology* 22(5): e1014300. https://doi.org/10.1371/journal.pcbi.1014300

**BibTeX:**
```bibtex
@article{muthupandiyan2026coevolutionary,
  title={Coevolutionary dynamics of viruses and their defective interfering particles},
  author={Muthupandiyan, Shiv and Yin, John},
  journal={PLOS Computational Biology},
  volume={22},
  number={5},
  pages={e1014300},
  year={2026},
  doi={10.1371/journal.pcbi.1014300},
  publisher={Public Library of Science}
}
```

## License

This project is available under the **CC-BY-NC 4.0 International license**.

## Contact

**Corresponding Author:** John Yin (john.yin@wisc.edu)  
*Wisconsin Institute for Discovery, University of Wisconsin-Madison*
