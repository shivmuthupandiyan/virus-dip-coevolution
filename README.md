# Coevolutionary dynamics of viruses and their defective interfering particles

This repository contains the source code and simulation framework for the paper:

**Coevolutionary dynamics of viruses and their defective interfering particles**  
*Shiv Muthupandiyan, John Yin*  
Wisconsin Institute for Discovery, Chemical and Biological Engineering, University of Wisconsin-Madison.  
**DOI:** [10.1101/2025.10.22.683971](https://doi.org/10.1101/2025.10.22.683971)

## Overview

Defective interfering particles (DIPs) are viral mutants that arise naturally during infection. Because they lack essential functions, they parasitize intact viruses during co-infection. This project implements the mathematical model described in the paper above—a continuous phenotype-space model using coupled partial differential equations (PDEs)—to explore the evolutionary interplay between viruses and DIPs.

Unlike traditional strong-selection models, this framework captures **strong-mutation regimes** where both populations diffuse through trait space.


## Installation

### Dependencies

```bash
pip install numpy scipy pandas matplotlib seaborn ipykernel imageio tqdm
```

Or use the included virtual environment:

```bash
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate  # On Windows
```

## Usage

### Basic Example

```python
from pdesolver import sol_dip_virus_pde_fft, plot_mean_phenotypes

# Use default parameters to reproduce baseline dynamics
results = sol_dip_virus_pde_fft()

# Plot the results
plot_mean_phenotypes(results)
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

- `t` - Time points
- `V_total`, `D_total` - Total population sizes over time
- `V_mean`, `D_mean` - Mean phenotype positions (x, y coordinates)
- `density_series` - (Optional) Full spatial density fields at each time point

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
├── figures.ipynb             # Main analysis and visualization notebook
├── README.md                 # This file
├── paper_figures/            # Publication figure generation scripts
│   ├── fig5_latin_hypercube.py   # Parameter sensitivity analysis
│   ├── fig6_cross_passage.py     # Time-shift resistances
│   ├── fig7_escape.py            # Escape dynamics analysis
│   ├── s5_ndim.ipynb             # Supplementary: N-dimensional analysis
│   └── s6_kernels.ipynb          # Supplementary: Kernel method analysis
└── venv/                     # Python virtual environment
```

## Mathematical Framework

The model solves the following dimensionless system:

$$ \frac{\partial v}{\partial t} = \mu \nabla^2 v + [1 - \alpha|x|^2 - \beta|x - \bar{x}_V|^2 - \eta - I_V(x,t)]L(t)v - \gamma v $$

$$ \frac{\partial d}{\partial t} = \mu \nabla^2 d + [I_D(x,t)d + \eta v]L(t) - \gamma d $$

Where $I_V$ and $I_D$ represent the phenotype-dependent interference cost and benefit, calculated via convolution integrals over the trait space.

## Citation

If you use this code in your research, please cite the following preprint:

> Muthupandiyan, S., & Yin, J. (2025). **Coevolutionary dynamics of viruses and their defective interfering particles**. *bioRxiv*. DOI: https://doi.org/10.1101/2025.10.22.683971

**BibTeX:**
```bibtex
@article{muthupandiyan2025coevolutionary,
  title={Coevolutionary dynamics of viruses and their defective interfering particles},
  author={Muthupandiyan, Shiv and Yin, John},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.10.22.683971},
  publisher={Cold Spring Harbor Laboratory}
}
```

## License

This project is available under the **CC-BY-NC 4.0 International license**.

## Contact

**Corresponding Author:** John Yin (john.yin@wisc.edu)  
*Wisconsin Institute for Discovery, University of Wisconsin-Madison*
