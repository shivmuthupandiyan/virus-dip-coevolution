Dependencies:
    numpy,
    scipy,
    pandas,
    matplotlib,
    seaborn,
    ipykernel,
    imageio (gif),
    tqdm (progress bar)

Main solver functionality is pdesolver.py 

Solves a discretized version of the model using RK45

Takes params dictionary:

default_params = {
        "nx": 50 (number of discretized rows/columns), "bord": 5 (spans [-bord, bord]), "T": 100 (number of time steps),
        "mu": 4e-3 (mutation rate), "gamma": 0.3 (decay rate), 'eta': 1e-3 (de novo generation rate),
        "r_v": 1 (base viral replication rate), "alpha": 0.025 (centering term), "beta": 0.025 (aggregation term),
        "K_cap": 1e8 (carrying capacity), "kappa": 1e-7 (interference strength, must be scaled inversely with K_cap), "sigma": 1 (interference kernel standard deviation),
        "V0_total": 1e6 (initial virus population), "v_init_pos": [1, 1] (initial virus location),
        "D0_total": 1e6 (initial DIP population), "d_init_pos": [1.4, 0] (initial DIP location),
        "init_spread": 0.4 (initial population standard devation), "monitor_extinction": True (stops run if V goes below extinction thresholds),
        "extinction_threshold_V": 1e2, "extinction_threshold_D": 1e2,
    }

Define any different params, then call sol_dip_virus_pde_fft(params)

Ex.
    params = {'T': 200, 'D0_total': 0, 'kappa': 0, 'mu': 1e-2}
    results = sol.sol_dip_virus_pde_fft(params, save_density_series=True)
    plot_mean_phenotypes(results)