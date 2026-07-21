import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

# === PATH CONFIGURATION ===
LOCAL_CSV_PATH = "/Users/karlroesch/Desktop/karl/VSworking/RZ/first100000.csv"
WORKSPACE_DIR = "/Users/karlroesch/Desktop/karl/VSworking/RZ"
FINAL_OUTPUT_FILE = os.path.join(WORKSPACE_DIR, "macro_100k_extrapolation.csv")

# === PHASE 1: LOAD ALL 100,000 ZEROS ===
print("=== PHASE 1: LOADING ENTIRE LOCAL DATASET (100K) ===")
if not os.path.exists(LOCAL_CSV_PATH):
    print(f"ERROR: File not found at {LOCAL_CSV_PATH}")
    exit()

print(f"-> Reading all zeros from local source...")
try:
    df_raw = pd.read_csv(LOCAL_CSV_PATH, header=None)
    gamma_zeros = pd.to_numeric(df_raw.iloc[:, 0], errors='coerce').dropna().values
    
    total_zeros = len(gamma_zeros)
    print(f"-> Successfully loaded {total_zeros:,} zeros.")
    print(f"   First zero: {gamma_zeros[0]:.5f} | Last zero: {gamma_zeros[-1]:.5f}")
except Exception as e:
    print(f"Parsing error: {e}")
    exit()

# === PHASE 2: DOUBLE-PRECISION LANDSCAPE SWEEP ===
print("\n=== PHASE 2: DOUBLE-PRECISION LANDSCAPE SWEEP ===")
a_start, a_end = 5.0, 60.0
step_size = 0.001 
steps = int((a_end - a_start) / step_size) + 1
a_space = np.linspace(a_start, a_end, steps)

print(f"-> Scanning landscape (a = {a_start} to {a_end}) over {steps:,} grid points...")

# Vectorized logarithmic accumulation utilizing full 64-bit float registers
log_filters = np.zeros(len(a_space))
for i, a in enumerate(a_space):
    cos_terms = np.cos(np.pi * a / gamma_zeros)**2
    log_filters[i] = np.sum(np.log(cos_terms + 1e-300))

# Convert out of log space cleanly
filter_values = np.exp(log_filters)

# === PHASE 3: GRID RESOLUTION PEAK DETECTION ===
print("\n=== PHASE 3: GRID RESOLUTION PEAK DETECTION ===")
peaks, _ = find_peaks(log_filters, prominence=0.5, distance=10)
print(f"-> Isolated {len(peaks)} resonance peak windows.")

records = []
for idx, p_idx in enumerate(peaks):
    best_a = a_space[p_idx]
    f_val = filter_values[p_idx]
    
    # Standard precision scattering equations without arbitrary-precision overhead
    true_forward_T = 1.0 - f_val
    if true_forward_T <= 0:
        scattered_rhs = 0.0
    else:
        scattered_rhs = np.arcsinh(np.sqrt(f_val) / np.sqrt(true_forward_T))
    
    delta_transcendental = scattered_rhs - best_a

    records.append({
        'Peak_Index': idx + 1,
        'Thickness_a': best_a,
        'Filter_Value': f_val,
        'Scattered_RHS': scattered_rhs,
        'Delta': delta_transcendental
    })

df_peaks = pd.DataFrame(records)

# Spacing Metrics
if len(df_peaks) > 1:
    a_positions = df_peaks['Thickness_a'].values
    delta_a = np.diff(a_positions)
    df_peaks['Delta_a_to_Next'] = np.append(delta_a, np.nan)
    df_peaks['Interval_Ratio'] = np.append(delta_a[1:] / delta_a[:-1], [np.nan, np.nan])

# === PHASE 4: STRUCTURAL EXTRAPOLATION MODEL ===
print("\n=== PHASE 4: TREND EXTRAPOLATION MODEL ===")
# Fit an exponential decay trend line to the Scattered_RHS values: y = c * exp(-k * a)
valid_data = df_peaks.dropna(subset=['Scattered_RHS']).copy()
valid_data = valid_data[valid_data['Scattered_RHS'] > 0]

def exp_model(x, c, k):
    return c * np.exp(-k * x)

try:
    popt, _ = curve_fit(exp_model, valid_data['Thickness_a'].values, valid_data['Scattered_RHS'].values, p0=[1e-3, 0.5])
    c_fit, k_fit = popt
    print(f"-> Extrapolation Formula Found: Scattered_RHS = {c_fit:.4e} * exp(-{k_fit:.4f} * a)")
    
    # Calculate a projection for an extended scale value (e.g., a = 100)
    a_target = 100.0
    projected_rhs = exp_model(a_target, c_fit, k_fit)
    print(f"-> Projected Scattered_RHS value at a = {a_target}: {projected_rhs:.4e}")
except Exception as e:
    print(f"-> Trend modeling skipped: {e}")

# Save results
df_peaks.to_csv(FINAL_OUTPUT_FILE, index=False)
print(f"\nResults successfully saved to: {FINAL_OUTPUT_FILE}")

print("\n=== FULL 100K RESCALED MATRIX ===")
print(df_peaks.head(15).to_string(index=False, formatters={
    'Thickness_a': '{:,.3f}'.format,
    'Filter_Value': '{:.4e}'.format,
    'Scattered_RHS': '{:.4e}'.format,
    'Delta': '{:,.4f}'.format,
    'Delta_a_to_Next': '{:,.3f}'.format,
    'Interval_Ratio': '{:,.4f}'.format
}))