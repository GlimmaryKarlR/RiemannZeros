import numpy as np
import pandas as pd
from decimal import Decimal, getcontext
import os

# === HIGH PRECISION CONFIGURATION ===
# Set precision extremely high to prevent underflow at macro-scale distances.
getcontext().prec = 150

def estimate_riemann_zero(n):
    """
    Estimates the imaginary height of the n-th Riemann zero (gamma_n)
    using the Riemann-von Mangoldt formula via Newton-Raphson approximation.
    """
    if n < 1:
        raise ValueError("Zero index must be >= 1")
        
    T = 2 * np.pi * n / np.log(n)
    
    for _ in range(10):
        term = T / (2 * np.pi)
        N_T = term * np.log(term) - term
        dN_dT = np.log(term) / (2 * np.pi)
        T = T - (N_T - n) / dN_dT
        
    return T

def project_macroscopic_scattering(a_target, c_fit, k_fit):
    """
    Projects the unshielded Scattered_RHS noise profile utilizing the topological
    exponential decay formula, evaluated at arbitrary precision.
    """
    c = Decimal(str(c_fit))
    k = Decimal(str(k_fit))
    a = Decimal(str(a_target))
    
    exponent = -k * a
    scattered_rhs = c * exponent.exp()
    
    return scattered_rhs

if __name__ == "__main__":
    # === GLOBAL TOPOLOGICAL CONSTANTS ===
    C_FIT = 3.2537e+05
    K_FIT = 2.3771
    
    # === RANGE CONFIGURATION UPDATE HERE===
    start_index = 19980
    end_index = 20000
    
    print("\n=== MACRO-SCALE HILBERT SPACE BATCH PROJECTION ===")
    print(f"Targeting Zero Indices: {start_index:,} to {end_index:,}")
    print(f"Total computations to run: {(end_index - start_index) + 1:,}")
    print("Processing...\n")
    
    results = []
    
    for n in range(start_index, end_index + 1):
        # 1. Project the geometric coordinate axis
        gamma_n = estimate_riemann_zero(n)
        
        # 2. Evaluate the bulk scattering noise at that metric
        projected_noise = project_macroscopic_scattering(gamma_n, C_FIT, K_FIT)
        
        # Format the extremely small noise value cleanly for the CSV
        noise_str = f"{projected_noise:.15E}"
        
        results.append({
            'Index_n': n,
            'Projected_Metric_a': gamma_n,
            'Scattering_Noise_RHS': noise_str
        })
        
        # Print a status update every 1,000 iterations
        if n % 1000 == 0:
            print(f"Processed index {n:,}...")

    # === DATA EXPORT ===
    df_results = pd.DataFrame(results)
    
    output_filename = f"topological_projection_{start_index}_to_{end_index}.csv"
    df_results.to_csv(output_filename, index=False)
    
    print("\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Dataset saved to: {os.path.abspath(output_filename)}")

     # UPDATE HERE FOR DISPLAY===
    
    print("\n--- First 5 Projections ---")
    print(df_results.head(5).to_string(index=False))
    
    print("\n--- Last 5 Projections ---")
    print(df_results.tail(5).to_string(index=False))