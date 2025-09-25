import numpy as np
from scipy.integrate import solve_ivp
from numpy.fft import fft2, ifft2 
from scipy.fft import next_fast_len 
from scipy.sparse import diags 

default_params = {
        "nx": 50, "bord": 5, "T": 100,
        "mu": 4e-3, "gamma": 0.3, 'eta': 1e-3,
        "r_v": 1, "alpha": 0.025, "beta": 0.025,
        "K_cap": 1e8, "kappa": 1e-7, "sigma": 1,
        "V0_total": 1e6, "v_init_pos": [1, 1],
        "D0_total": 1e6, "d_init_pos": [1.4, 0],
        "init_spread": 0.4, "monitor_extinction": True,
        "extinction_threshold_V": 1e2, "extinction_threshold_D": 1e2,
    }

def event_virus_extinction_simplified(t, QL, A, B, PHENOx, PHENOy, _fft_K_exp_argument, params_dict):
    nx, ny, dx, dy = params_dict['nx'], params_dict['nx'], params_dict['dx'], params_dict['dy']
    current_V_total = np.sum(np.maximum(0, QL[:nx * ny])) * dx * dy
    return current_V_total - params_dict.get('extinction_threshold_V', 1e-2)
event_virus_extinction_simplified.terminal = True
event_virus_extinction_simplified.direction = -1

def event_dip_extinction_simplified(t, QL, A, B, PHENOx, PHENOy, _fft_K_exp_argument, params_dict):
    nx, ny, dx, dy = params_dict['nx'], params_dict['nx'], params_dict['dx'], params_dict['dy']
    current_D_total = np.sum(np.maximum(0, QL[nx * ny:])) * dx * dy
    return current_D_total - params_dict.get('extinction_threshold_D', 1e-2)
event_dip_extinction_simplified.terminal = False
event_dip_extinction_simplified.direction = -1

def count_local_maxima(density, threshold):
    if np.max(density) < threshold: return 0
    ny, nx = density.shape; count = 0
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            if density[i, j] >= threshold and np.all(density[i, j] > density[i-1:i+2, j-1:j+2]):
                count += 1
    return max(1, count) if np.max(density) >= threshold and count == 0 else count

def calculate_distribution_shape_stats(density, PHENOx, PHENOy, dx, dy, min_total=1e-9):
    total_mass = np.sum(density) * dx * dy
    nan_keys = ['variance_x', 'variance_y', 'skewness_x', 'skewness_y', 'kurtosis_x', 'kurtosis_y', 'mean_distance_from_origin', 'variance_distance_from_origin', 'anisotropy_ratio', 'effective_radius', 'compactness', 'max_density']
    if total_mass < min_total:
        return {k: np.nan for k in nan_keys} | {'num_clusters': 0}
    prob_density = density / np.sum(density)
    mean_x = np.sum(PHENOx * prob_density); mean_y = np.sum(PHENOy * prob_density)
    dx_centered, dy_centered = PHENOx - mean_x, PHENOy - mean_y
    variance_x, variance_y = np.sum(dx_centered**2 * prob_density), np.sum(dy_centered**2 * prob_density)
    skewness_x = np.sum(dx_centered**3 * prob_density) / (variance_x**1.5) if variance_x > 1e-12 else 0.0
    skewness_y = np.sum(dy_centered**3 * prob_density) / (variance_y**1.5) if variance_y > 1e-12 else 0.0
    kurtosis_x = np.sum(dx_centered**4 * prob_density) / (variance_x**2) - 3.0 if variance_x > 1e-12 else 0.0
    kurtosis_y = np.sum(dy_centered**4 * prob_density) / (variance_y**2) - 3.0 if variance_y > 1e-12 else 0.0
    distances = np.sqrt(PHENOx**2 + PHENOy**2); mean_distance = np.sum(distances * prob_density)
    variance_distance = np.sum((distances - mean_distance)**2 * prob_density)
    cov_xy = np.sum(dx_centered * dy_centered * prob_density)
    trace, det = variance_x + variance_y, variance_x * variance_y - cov_xy**2
    if trace > 1e-12 and det >= 0:
        sqrt_discriminant = np.sqrt(max(0, trace**2 - 4 * det))
        eigenval1, eigenval2 = (trace + sqrt_discriminant) / 2, (trace - sqrt_discriminant) / 2
        anisotropy_ratio = eigenval1 / eigenval2 if eigenval2 > 1e-12 else np.inf
    else: anisotropy_ratio = 1.0
    effective_radius = np.sqrt(variance_x + variance_y)
    max_density = np.max(density)
    compactness = max_density / (effective_radius + 1e-12)
    num_clusters = count_local_maxima(density, 0.1 * max_density)
    return {'variance_x': variance_x, 'variance_y': variance_y, 'skewness_x': skewness_x, 'skewness_y': skewness_y, 'kurtosis_x': kurtosis_x, 'kurtosis_y': kurtosis_y, 'mean_distance_from_origin': mean_distance, 'variance_distance_from_origin': variance_distance, 'anisotropy_ratio': anisotropy_ratio, 'effective_radius': effective_radius, 'compactness': compactness, 'num_clusters': num_clusters, 'max_density': max_density}

def create_fft_exp_kernel_padded(params, scale_param_name, padded_nx, padded_ny):
    dx, dy = params['dx'], params['dy']
    scale = params[scale_param_name]
    if scale <= 1e-9: return np.ones((padded_ny, padded_nx), dtype=complex)
    scale_sq = scale**2
    ix, iy = np.arange(padded_nx), np.arange(padded_ny)
    ixx, iyy = np.meshgrid(ix, iy)
    dist_x_sq = (np.minimum(ixx, padded_nx - ixx) * dx)**2
    dist_y_sq = (np.minimum(iyy, padded_ny - iyy) * dy)**2
    exp_kernel = np.exp(-(dist_x_sq + dist_y_sq) / (2 * scale_sq))
    kernel_sum = np.sum(exp_kernel) * dx * dy
    if kernel_sum > 1e-9: exp_kernel /= kernel_sum
    return fft2(exp_kernel)

def dip_virus_pde_fft_padded(t, QL, A, B, PHENOx, PHENOy, passed_fft_K_exp, params):
    nx, ny, dx, dy = params['nx'], params['nx'], params['dx'], params['dy']
    padded_nx, padded_ny = params['padded_nx'], params['padded_ny']
    mu, K_cap, kappa, gamma, eta = params["mu"], params["K_cap"], params["kappa"], params["gamma"], params['eta']
    v_density = np.maximum(0, QL[:nx * ny]).reshape(ny, nx)
    d_density = np.maximum(0, QL[nx * ny:]).reshape(ny, nx)
    V_total, D_total = np.sum(v_density) * dx * dy, np.sum(d_density) * dx * dy
    xbarV = [np.sum(PHENOx * v_density) * dx * dy / V_total, np.sum(PHENOy * v_density) * dx * dy / V_total] if V_total > 1e-9 else [0.0, 0.0]
    logistic_factor = (1 - (V_total + D_total) / max(K_cap, 1e-12))
    v_padded, d_padded = np.zeros((padded_ny, padded_nx)), np.zeros((padded_ny, padded_nx))
    v_padded[:ny, :nx], d_padded[:ny, :nx] = v_density, d_density
    fft_v_padded, fft_d_padded = fft2(v_padded), fft2(d_padded)
    conv_exp_d_spatial = np.real(ifft2(passed_fft_K_exp * fft_d_padded))[:ny, :nx]
    conv_exp_v_spatial = np.real(ifft2(passed_fft_K_exp * fft_v_padded))[:ny, :nx]
    interference_rate_on_v = kappa * conv_exp_d_spatial
    interference_gain_for_d = kappa * conv_exp_v_spatial
    r_v_map = (params["r_v"] - eta - params["alpha"] * (PHENOx**2 + PHENOy**2) - params["beta"] * ((xbarV[0]-PHENOx)**2 + (xbarV[1]-PHENOy)**2))
    dv_dt = (mu * (v_density @ A + B @ v_density) + (r_v_map * v_density - v_density * interference_rate_on_v) * logistic_factor - gamma * v_density)
    dd_dt = (mu * (d_density @ A + B @ d_density) + (d_density * interference_gain_for_d + v_density * eta) * logistic_factor - gamma * d_density)
    return np.concatenate([dv_dt.flatten(), dd_dt.flatten()])

def sol_dip_virus_pde_fft(params, initial_v_density=None, initial_d_density=None, 
                          save_density_series=False):
    """
    Main solver function with robust initial condition setup.
    """
    upd_params = default_params.copy(); upd_params.update(params)
    upd_params.update({'save_density_series': save_density_series})
    nx, ny, bord = upd_params['nx'], upd_params['nx'], upd_params['bord']
    padded_nx, padded_ny = next_fast_len(2*nx-1), next_fast_len(2*ny-1)
    upd_params.update({'padded_nx': padded_nx, 'padded_ny': padded_ny})
    dx = dy = (2.0 * bord) / nx
    x_centers, y_centers = np.linspace(-bord+dx/2, bord-dx/2, nx), np.linspace(-bord+dy/2, bord-dy/2, ny)
    PHENOx, PHENOy = np.meshgrid(x_centers, y_centers)
    upd_params.update({"dx": dx, "dy": dy})
    
    # --- START OF LOGIC CHANGE ---

    init_spread = upd_params.get("init_spread", 0.2)
    
    # --- If spread is very small, use a robust delta function ---
    if init_spread < (dx / 2):
        print(f"Warning: init_spread ({init_spread:.2e}) is smaller than half the grid spacing ({dx/2:.2e}). Using a single-point initial condition to avoid numerical instability.")

        # For Virus
        v_init = np.zeros_like(PHENOx)
        # Find the index of the grid cell closest to the desired initial position
        v_idx_y = np.argmin(np.abs(y_centers - upd_params["v_init_pos"][1]))
        v_idx_x = np.argmin(np.abs(x_centers - upd_params["v_init_pos"][0]))
        # Place all the population mass there, remembering that the value is a DENSITY.
        # Density = Total Mass / Area
        v_init[v_idx_y, v_idx_x] = upd_params["V0_total"] / (dx * dy)
        
        # For DIP
        d_init = np.zeros_like(PHENOx)
        d_idx_y = np.argmin(np.abs(y_centers - upd_params["d_init_pos"][1]))
        d_idx_x = np.argmin(np.abs(x_centers - upd_params["d_init_pos"][0]))
        d_init[d_idx_y, d_idx_x] = upd_params["D0_total"] / (dx * dy)

    # --- If spread is large enough, use the original Gaussian method ---
    else:
        # For Virus
        v_init_gauss = np.exp(-((PHENOx - upd_params["v_init_pos"][0])**2 + (PHENOy - upd_params["v_init_pos"][1])**2) / (2 * init_spread**2))
        v_init = v_init_gauss * (upd_params["V0_total"] / max(np.sum(v_init_gauss) * dx * dy, 1e-12))
        
        # For DIP
        d_init_gauss = np.exp(-((PHENOx - upd_params["d_init_pos"][0])**2 + (PHENOy - upd_params["d_init_pos"][1])**2) / (2 * init_spread**2))
        d_init = d_init_gauss * (upd_params["D0_total"] / max(np.sum(d_init_gauss) * dx * dy, 1e-12))
    
    # --- END OF LOGIC CHANGE ---

    Q0 = np.concatenate([v_init.flatten(), d_init.flatten()])
    
    A_1D = diags([-2,1,1], [0,-1,1], shape=(nx,nx)).toarray(); A_1D[0,1]=2; A_1D[-1,-2]=2; A=A_1D/dx**2
    B_1D = diags([-2,1,1], [0,-1,1], shape=(ny,ny)).toarray(); B_1D[0,1]=2; B_1D[-1,-2]=2; B=B_1D/dy**2
    
    fft_K_exp = create_fft_exp_kernel_padded(upd_params, 'sigma', padded_nx, padded_ny)
    
    events, event_names = [], []
    if upd_params.get('monitor_extinction', False):
        events.extend([event_virus_extinction_simplified, event_dip_extinction_simplified])
        
    sol = solve_ivp(fun=dip_virus_pde_fft_padded, t_span=[0, upd_params['T']], y0=Q0, method='RK45', 
                    rtol=1e-6, atol=1e-9, args=(A, B, PHENOx, PHENOy, fft_K_exp, upd_params), 
                    events=events or None)
                    
    termination_reason = 't_end_reached'
    if sol.status == 1:
        # Check which event was triggered, if any
        for i, event_list in enumerate(sol.t_events):
            if event_list.size > 0:
                if i == 0: termination_reason = 'Virus Extinction'
                # DIP extinction is not terminal, so we just count it.
                break
    elif sol.status < 0: 
        termination_reason = 'Solver Error'
        
    dip_extinction_count = len(sol.t_events[1]) if events and len(sol.t_events) > 1 and sol.t_events[1] is not None else 0
    
    # --- Data processing remains the same ---
    num_t = len(sol.t)
    V_total_time, D_total_time = np.zeros(num_t), np.zeros(num_t)
    mean_phenotype_v, mean_phenotype_d = np.full((num_t, 2), np.nan), np.full((num_t, 2), np.nan)
    shape_stats_v_list, shape_stats_d_list = [], []
    
    # Pre-calculate indices for speed
    v_indices = slice(0, nx * ny)
    d_indices = slice(nx * ny, None)
    
    for i in range(num_t):
        v_density = np.maximum(0, sol.y[v_indices, i]).reshape(ny, nx)
        d_density = np.maximum(0, sol.y[d_indices, i]).reshape(ny, nx)
        
        V_total_time[i] = np.sum(v_density) * dx * dy
        D_total_time[i] = np.sum(d_density) * dx * dy

        if V_total_time[i] > 1e-12:
            mean_phenotype_v[i, 0] = np.sum(PHENOx * v_density) * dx * dy / V_total_time[i]
            mean_phenotype_v[i, 1] = np.sum(PHENOy * v_density) * dx * dy / V_total_time[i]
        
        if D_total_time[i] > 1e-12:
            mean_phenotype_d[i, 0] = np.sum(PHENOx * d_density) * dx * dy / D_total_time[i]
            mean_phenotype_d[i, 1] = np.sum(PHENOy * d_density) * dx * dy / D_total_time[i]
            
        shape_stats_v_list.append(calculate_distribution_shape_stats(v_density, PHENOx, PHENOy, dx, dy))
        shape_stats_d_list.append(calculate_distribution_shape_stats(d_density, PHENOx, PHENOy, dx, dy))
        
    shape_keys = shape_stats_v_list[0].keys() if shape_stats_v_list else {}
    shape_stats_v = {key: np.array([s[key] for s in shape_stats_v_list]) for key in shape_keys}
    shape_stats_d = {key: np.array([s[key] for s in shape_stats_d_list]) for key in shape_keys}
    
    base_results = {
        "time_points": sol.t, "V_total_time": V_total_time, "D_total_time": D_total_time,
        "mean_phenotype_v": mean_phenotype_v, "mean_phenotype_d": mean_phenotype_d,
        "shape_stats_v": shape_stats_v, "shape_stats_d": shape_stats_d,
        "params": upd_params, "success": sol.success, "termination_reason": termination_reason,
        "dip_extinction_count": dip_extinction_count
    }
    
    if upd_params['save_density_series']:
        v_series_3d = sol.y[v_indices, :].T.reshape(num_t, ny, nx)
        d_series_3d = sol.y[d_indices, :].T.reshape(num_t, ny, nx)
        v_series_transposed = v_series_3d.transpose((1, 2, 0))
        d_series_transposed = d_series_3d.transpose((1, 2, 0))
        
        base_results.update({
            "v_dist_time": v_series_transposed,
            "d_dist_time": d_series_transposed,
            "PHENOx": PHENOx, "PHENOy": PHENOy
        })
        
    return base_results

