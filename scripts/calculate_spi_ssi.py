import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
import os

# Paths
output_dir = r"D:\Research Datasets\Additional Datasets\Defence 08\Deraniyagala\SPI and SSI"
os.makedirs(output_dir, exist_ok=True)
rainfall_file = r"D:\Research Datasets\Additional Datasets\Defence 08\Deraniyagala\Datasets\Deraniyagala_RF_1990-2015.xlsx"
discharge_file = r"D:\Research Datasets\Additional Datasets\Defence 08\Deraniyagala\Datasets\Deraniyagala_SF_1990-2015.xlsx"

# Read and clean
rf = pd.read_excel(rainfall_file, usecols=['Date', 'Rainfall'])
q = pd.read_excel(discharge_file, usecols=['Date', 'Discharge'])

rf['Date'] = pd.to_datetime(rf['Date'], errors='coerce')
q['Date'] = pd.to_datetime(q['Date'], errors='coerce')

rf = rf.dropna(subset=['Date'])
q = q.dropna(subset=['Date'])

# Monthly aggregation with Date as index
rf_monthly = rf.resample('ME', on='Date').sum()
q_monthly = q.resample('ME', on='Date').sum()

# Unified SPI/SSI function
def calculate_index(data, col, timescales):
    result = pd.DataFrame(index=data.index)
    series = data[col]
    prefix = 'SPI' if col == 'Rainfall' else 'SSI'
    
    for ts in timescales:
        rolled = series.rolling(ts, min_periods=ts).sum()
        fit_data = rolled.dropna()
        
        index_vals = np.full(len(rolled), np.nan)
        
        # Only fit if we have enough data
        if len(fit_data) >= 10:
            # Remove zeros and negatives for fitting (standard practice)
            fit_pos = fit_data[fit_data > 0]
            
            if len(fit_pos) >= 10:  # Need positive values to fit gamma
                try:
                    shape, loc, scale = gamma.fit(fit_pos, floc=0)
                except:
                    # Fallback: use method of moments if MLE fails
                    mean_pos = fit_pos.mean()
                    var_pos = fit_pos.var()
                    shape = (mean_pos ** 2) / var_pos
                    scale = var_pos / mean_pos
                    loc = 0
                
                # Compute CDF: zeros get CDF = 0, positives use fitted gamma
                cdf = np.zeros(len(rolled))
                nonzero_idx = rolled > 0
                cdf[nonzero_idx] = gamma.cdf(rolled[nonzero_idx], shape, loc, scale)
                
                # Probability of zero (for mixed distribution)
                p_zero = (rolled == 0).sum() / len(rolled.dropna())
                # Adjust CDF for zero-inflation: P(X=0) = p_zero, P(X>0) = 1 - p_zero
                # Standard SPI: CDF at zero = p_zero (empirical)
                # But we use: CDF(x) = p_zero + (1 - p_zero) * gamma_cdf(x)
                cdf = p_zero + (1 - p_zero) * cdf
                
                index_vals = norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
            else:
                # Not enough positive values → fallback to empirical
                pass  # leave as NaN
        
        result[f'{prefix}_{ts}'] = index_vals
    
    result = result.reset_index()
    return result

# Compute
timescales = [1, 3, 12]
spi = calculate_index(rf_monthly, 'Rainfall', timescales)
ssi = calculate_index(q_monthly, 'Discharge', timescales)

# Merge and enforce order
results = pd.merge(spi, ssi, on='Date', how='inner')
cols_order = ['Date', 'SPI_1', 'SPI_3', 'SPI_12', 'SSI_1', 'SSI_3', 'SSI_12']
results = results[[col for col in cols_order if col in results.columns]]

# Save
output_file = os.path.join(output_dir, 'Deraniyagala_SPI_SSI.xlsx')
results.to_excel(output_file, index=False)
print(f"Done. Saved: {output_file}")
