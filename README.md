Dependencies:
    numpy
    scipy
    pandas
    matplotlib
    seaborn
    ipykernel
    imageio (gif)
    tqdm (progress bar)

Main solver functionality is pdesolver.py

Best way to interact is to define params, then call sol_dip_virus_pde_fft(params)

Ex.
    params = {'T': 200, 'D0_total': 0, 'kappa': 0, 'mu': 1e-2}
    results = sol.sol_dip_virus_pde_fft(params, save_density_series=True)
    plot_mean_phenotypes(results)

Figs 5 and 7 should just be called from the command line