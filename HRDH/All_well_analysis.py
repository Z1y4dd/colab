from dlisio import dlis
import pandas as pd
from pathlib import Path
import traceback
import numpy as np
from scipy.spatial import cKDTree
import glob
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

########################################################################################
# loading joined data + finding correlations

#1. Data Loading Function
def load_all_wells_data(base_path="."):
    
    """
    Load all joined CSV files from HRDH_* folders and combine them into a single DataFrame.
    Handles duplicate columns automatically.
    
    Parameters:
    -----------
    base_path : str or Path
        Base directory containing HRDH_* folders with joined CSV files
    
    Returns:
    --------
    DataFrame
        Combined DataFrame with data from all wells
    """
    from pathlib import Path
    import pandas as pd
    
    # Find all joined CSV files
    base_path = Path(base_path)
    joined_files = list(base_path.glob('HRDH_*/HRDH_*_joined.csv'))
    
    print(f"Found {len(joined_files)} joined CSV files:")
    for file in joined_files:
        print(f"  {file}")
    
    if not joined_files:
        raise FileNotFoundError("No joined CSV files found in specified path")
    
    # Load each file and add well identifier
    all_data = []
    
    for file in joined_files:
        try:
            # Extract well name from folder name
            well_name = file.parent.name
            
            # Load data
            df = pd.read_csv(file)
            
            # Handle duplicate columns
            if df.columns.duplicated().any():
                print(f"  Found duplicate columns in {well_name}")
                
                # Process each duplicate column
                for col in df.columns[df.columns.duplicated(keep=False)]:
                    # Get all columns with this name
                    dup_cols = [df[c] for c in df.columns if c == col]
                    
                    # Create a merged series starting with the first occurrence
                    merged = dup_cols[0].copy()
                    
                    # Fill NaN values from subsequent duplicates
                    for dup in dup_cols[1:]:
                        merged = merged.fillna(dup)
                    
                    # Replace the first occurrence with merged data
                    df[col] = merged
                
                # Keep only first occurrence of each column
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
                print(f"    Merged duplicate columns")
            
            # Add well identifier
            df['Well'] = well_name
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Add to list
            all_data.append(df)
            
            print(f"Loaded {well_name}: {len(df)} samples, {len(df.columns)} columns")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not all_data:
        raise ValueError("No files were successfully loaded")
    
    # Concatenate all dataframes
    df_all = pd.concat(all_data, ignore_index=True, sort=True)
    
    # Ensure Well column is a category for efficiency
    df_all['Well'] = df_all['Well'].astype('category')
    
    print(f"\nCombined dataset: {len(df_all)} total samples from {df_all['Well'].nunique()} wells")
    print(f"Wells: {', '.join(sorted(df_all['Well'].unique()))}")
    print(f"Total columns: {len(df_all.columns)}")
    
    return df_all
#2. Correlation Calculation Function
def calculate_well_correlations(df_all, min_samples=10):
    """
    Calculate correlations between log and lab measurements for each well.
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells
    min_samples : int
        Minimum number of samples required for correlation calculation
        
    Returns:
    --------
    tuple : (well_correlations, well_stats)
        - well_correlations: Dict with correlation matrices for each well
        - well_stats: Dict with statistics for each well
    """
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr
    
    # Extract log and lab columns
    lab_columns = [col for col in df_all.columns if col.startswith('Lab_') and 
                  col not in ['Lab_Depth', 'Lab_Sample_ID']]
    log_columns = [col for col in df_all.columns if col.startswith('Log_') and 
                  col not in ['Log_Depth', 'Log_FRAMENO']]
    
    print(f"Found {len(lab_columns)} lab variables and {len(log_columns)} log variables\n")
    
    well_correlations = {}
    well_stats = {}
    
    for well in sorted(df_all['Well'].unique()):
        print(f"Processing {well}...")
        
        # Filter data for this well
        well_data = df_all[df_all['Well'] == well].copy()
        
        # Select only columns that exist in this well's data
        available_log_cols = [col for col in log_columns if col in well_data.columns]
        available_lab_cols = [col for col in lab_columns if col in well_data.columns]
        
        if not available_log_cols or not available_lab_cols:
            print(f"  Warning: No valid log or lab columns for {well}")
            continue
            
        # Initialize correlation matrices
        correlation_matrix = pd.DataFrame(index=available_log_cols, columns=available_lab_cols, dtype=float)
        pvalue_matrix = pd.DataFrame(index=available_log_cols, columns=available_lab_cols, dtype=float)
        sample_size_matrix = pd.DataFrame(index=available_log_cols, columns=available_lab_cols, dtype=int)
        
        # Calculate correlations for each log-lab pair
        total_pairs = len(available_log_cols) * len(available_lab_cols)
        valid_correlations = 0
        correlation_values = []
        
        for log_col in available_log_cols:
            for lab_col in available_lab_cols:
                # Find samples with both measurements
                common_indices = well_data[log_col].notna() & well_data[lab_col].notna()
                common_data = well_data.loc[common_indices, [log_col, lab_col]]
                
                if len(common_data) >= min_samples:
                    x = common_data[log_col]
                    y = common_data[lab_col]
                    
                    # Only calculate if we have variation in the data
                    if x.std() > 0 and y.std() > 0:
                        try:
                            r, p = pearsonr(x, y)
                            correlation_matrix.loc[log_col, lab_col] = r
                            pvalue_matrix.loc[log_col, lab_col] = p
                            sample_size_matrix.loc[log_col, lab_col] = len(common_data)
                            
                            correlation_values.append(abs(r))
                            valid_correlations += 1
                            
                        except Exception as e:
                            print(f"    Error calculating correlation for {log_col} vs {lab_col}: {e}")
                            correlation_matrix.loc[log_col, lab_col] = np.nan
                            pvalue_matrix.loc[log_col, lab_col] = np.nan
                            sample_size_matrix.loc[log_col, lab_col] = 0
                    else:
                        # Skip if no variation
                        correlation_matrix.loc[log_col, lab_col] = np.nan
                        pvalue_matrix.loc[log_col, lab_col] = np.nan
                        sample_size_matrix.loc[log_col, lab_col] = len(common_data)
                else:
                    # Not enough samples
                    correlation_matrix.loc[log_col, lab_col] = np.nan
                    pvalue_matrix.loc[log_col, lab_col] = np.nan
                    sample_size_matrix.loc[log_col, lab_col] = len(common_data)
        
        # Store results for this well
        well_correlations[well] = {
            'correlation': correlation_matrix,
            'pvalue': pvalue_matrix,
            'sample_size': sample_size_matrix
        }
        
        # Calculate statistics for this well
        coverage = (valid_correlations / total_pairs) * 100 if total_pairs > 0 else 0
        mean_abs_corr = np.mean(correlation_values) if correlation_values else 0
        
        # Count strong and moderate correlations
        strong_corr = sum(1 for r in correlation_values if r >= 0.7)
        moderate_corr = sum(1 for r in correlation_values if 0.5 <= r < 0.7)
        
        well_stats[well] = {
            'total_pairs': total_pairs,
            'valid_correlations': valid_correlations,
            'coverage': coverage,
            'mean_abs_correlation': mean_abs_corr,
            'strong_correlations': strong_corr,
            'moderate_correlations': moderate_corr,
            'total_samples': len(well_data)
        }
        
        print(f"  {well}: {valid_correlations}/{total_pairs} valid correlations ({coverage:.1f}% coverage)")
        print(f"  Mean |r|: {mean_abs_corr:.3f}, Strong: {strong_corr}, Moderate: {moderate_corr}")
    
    return well_correlations, well_stats

#3. Common Correlation Analysis Function
def find_common_correlations(well_correlations, corr_threshold=0.5, min_wells=2):
    """
    Find correlations that appear in multiple wells above the specified threshold.
    
    Parameters:
    -----------
    well_correlations : dict
        Dictionary with correlation matrices for each well
    corr_threshold : float
        Minimum absolute correlation value to consider
    min_wells : int
        Minimum number of wells a correlation must appear in
    
    Returns:
    --------
    DataFrame
        Summary of common correlations
    """
    import pandas as pd
    import numpy as np
    
    # Track all correlations
    all_correlations = {}
    
    # Process each well
    for well, matrices in well_correlations.items():
        corr_matrix = matrices['correlation']
        
        # Find correlations above threshold
        for log_var in corr_matrix.index:
            for lab_var in corr_matrix.columns:
                r = corr_matrix.loc[log_var, lab_var]
                
                # Only consider valid correlations above threshold
                if pd.notna(r) and abs(r) >= corr_threshold:
                    pair_key = (log_var, lab_var)
                    
                    # Initialize if first time seeing this pair
                    if pair_key not in all_correlations:
                        all_correlations[pair_key] = {
                            'log_var': log_var,
                            'lab_var': lab_var,
                            'wells_found_in': [],
                            'r_values': {}
                        }
                    
                    # Add this well's correlation
                    all_correlations[pair_key]['wells_found_in'].append(well)
                    all_correlations[pair_key]['r_values'][well] = r
    
    # Filter to pairs found in multiple wells and calculate statistics
    common_pairs = []
    
    for pair_key, data in all_correlations.items():
        num_wells = len(data['wells_found_in'])
        
        if num_wells >= min_wells:
            r_values = list(data['r_values'].values())
            avg_r = np.mean(r_values)
            avg_abs_r = np.mean([abs(r) for r in r_values])
            std_r = np.std(r_values)
            
            # Check if correlation has consistent direction
            all_positive = all(r > 0 for r in r_values)
            all_negative = all(r < 0 for r in r_values)
            consistent_direction = all_positive or all_negative
            
            common_pairs.append({
                'log_var': data['log_var'],
                'lab_var': data['lab_var'],
                'num_wells': num_wells,
                'wells_found_in': sorted(data['wells_found_in']),
                'r_values': data['r_values'],
                'avg_r': avg_r,
                'avg_abs_r': avg_abs_r,
                'std_r': std_r,
                'consistent_direction': consistent_direction
            })
    
    # Convert to DataFrame
    if common_pairs:
        df_common = pd.DataFrame(common_pairs)
        
        # Sort by number of wells (descending) and average absolute correlation (descending)
        df_common = df_common.sort_values(['num_wells', 'avg_abs_r'], ascending=[False, False])
        
        return df_common
    else:
        return pd.DataFrame(columns=['log_var', 'lab_var', 'num_wells', 'wells_found_in', 
                                    'r_values', 'avg_r', 'avg_abs_r', 'std_r', 'consistent_direction'])
#4. Main Analysis Function
def analyze_wells(base_path=".", corr_threshold=0.5, min_samples=10, min_wells=2):
    """
    Complete multi-well correlation analysis with adjustable threshold.
    
    Parameters:
    -----------
    base_path : str
        Base path to search for HRDH_* folders
    corr_threshold : float
        Minimum correlation threshold (absolute value)
    min_samples : int
        Minimum samples needed for correlation calculation
    min_wells : int
        Minimum wells a correlation must appear in
    
    Returns:
    --------
    tuple
        (df_all, well_correlations, well_stats, common_correlations)
    """
    # Step 1: Load all well data
    df_all = load_all_wells_data(base_path)
    
    # Step 2: Calculate correlations for each well
    well_correlations, well_stats = calculate_well_correlations(df_all, min_samples)
    
    # Step 3: Find common correlations across wells
    common_correlations = find_common_correlations(
        well_correlations, 
        corr_threshold=corr_threshold,
        min_wells=min_wells
    )
    
    # Print summary
    print("\n" + "="*80)
    print(f"CORRELATION ANALYSIS SUMMARY (|r| ≥ {corr_threshold}, {min_wells}+ wells)")
    print("="*80)
    
    print(f"\nFound {len(common_correlations)} correlation pairs in {min_wells}+ wells")
    
    if not common_correlations.empty:
        # Count by number of wells
        well_counts = common_correlations['num_wells'].value_counts().sort_index(ascending=False)
        
        print("\nDistribution by number of wells:")
        for n_wells, count in well_counts.items():
            print(f"  {n_wells} wells: {count} correlations")
        
        # Count consistent vs. inconsistent correlations
        consistent = common_correlations['consistent_direction'].sum()
        inconsistent = len(common_correlations) - consistent
        
        print(f"\nConsistent direction: {consistent} ({consistent/len(common_correlations)*100:.1f}%)")
        print(f"Inconsistent direction: {inconsistent} ({inconsistent/len(common_correlations)*100:.1f}%)")
        
        # Show top correlations
        print("\nTop 10 strongest correlations:")
        top_10 = common_correlations.sort_values('avg_abs_r', ascending=False).head(10)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            log_name = row['log_var'].replace('Log_', '')
            lab_name = row['lab_var'].replace('Lab_', '')
            wells = [w.replace('HRDH_', '') for w in row['wells_found_in']]
            
            print(f"  {i}. {log_name} vs {lab_name}: r̄={row['avg_r']:.3f} ({row['num_wells']} wells: {', '.join(wells)})")
    
    return df_all, well_correlations, well_stats, common_correlations

#  visual for missningness of data  + data points count
def visualize_missing_data_by_well(df_all, lab_columns, log_columns):
    """
    Create missing data matrix and sample count visualizations grouped by well.
    """
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [3, 1]})
    
    # Get unique wells
    wells = sorted(df_all['Well'].unique())
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#ff7f0e', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#d62728'
    }
    
    # --- Subplot 1: Missing data matrix by well ---
    all_vars = lab_columns + log_columns
    missing_matrix = []
    
    for well in wells:
        well_data = df_all[df_all['Well'] == well][all_vars]
        missing_row = well_data.isnull().sum() / len(well_data) * 100
        missing_matrix.append(missing_row)
    
    missing_df = pd.DataFrame(missing_matrix, index=wells, columns=all_vars).T
    
    # Create heatmap
    sns.heatmap(missing_df, 
                cmap='RdYlGn_r',  # Red for high missing, Green for low missing
                cbar_kws={'label': 'Missing %'},
                annot=True, 
                fmt='.0f',
                annot_kws={'size': 8},
                linewidths=0.5,
                ax=ax1)
    
    ax1.set_title('Missing Data Percentage by Well and Variable', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Wells', fontsize=12)
    ax1.set_ylabel('Variables', fontsize=12)
    
    # Fix the tick labels
    yticks = ax1.get_yticks()
    if len(yticks) <= len(all_vars):
        step = max(1, len(all_vars) // len(yticks))
        sampled_vars = all_vars[::step][:len(yticks)]
        ax1.set_yticklabels([col.replace('Lab_', '').replace('Log_', '') for col in sampled_vars], 
                            rotation=0, fontsize=9)
    else:
        ax1.set_yticklabels([col.replace('Lab_', '').replace('Log_', '') for col in all_vars], 
                            rotation=0, fontsize=9)
    
    ax1.set_xticklabels(wells, rotation=45, ha='right')
    
    # --- Subplot 2: Sample count by well ---
    sample_counts = df_all['Well'].value_counts().loc[wells]
    bars = ax2.barh(range(len(wells)), sample_counts.values)
    
    # Color bars by well
    for i, (well, bar) in enumerate(zip(wells, bars)):
        bar.set_color(well_colors[well])
        ax2.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{sample_counts[well]:,}', va='center', fontsize=10)
    
    ax2.set_yticks(range(len(wells)))
    ax2.set_yticklabels(wells)
    ax2.set_xlabel('Number of Samples', fontsize=12)
    ax2.set_title('Sample Count by Well', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('Missing Data Analysis by Well', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig('imgs/missing_data_by_well_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("MISSING DATA ANALYSIS SUMMARY")
    print("="*80)
    
    # Calculate missing percentages by well
    print("\nOverall Missing Data by Well:")
    print("-" * 60)
    for well in wells:
        well_data = df_all[df_all['Well'] == well][all_vars]
        total_missing = well_data.isnull().sum().sum()
        total_cells = len(well_data) * len(all_vars)
        missing_pct = (total_missing / total_cells) * 100
        lab_missing = well_data[lab_columns].isnull().sum().sum() / (len(well_data) * len(lab_columns)) * 100 if lab_columns else 0
        log_missing = well_data[log_columns].isnull().sum().sum() / (len(well_data) * len(log_columns)) * 100 if log_columns else 0
        
        print(f"{well}:")
        print(f"  Total samples: {len(well_data):,}")
        print(f"  Overall missing: {missing_pct:.1f}%")
        print(f"  Lab data missing: {lab_missing:.1f}%")
        print(f"  Log data missing: {log_missing:.1f}%")
    
    # Variable completeness analysis
    var_completeness = []
    for var in all_vars:
        var_comps = []
        for well in wells:
            well_data = df_all[df_all['Well'] == well]
            if var in well_data.columns:
                completeness = (well_data[var].notna().sum() / len(well_data)) * 100
                var_comps.append(completeness)
        
        if var_comps:  # Only add if we have data
            var_completeness.append({
                'Variable': var.replace('Lab_', '').replace('Log_', ''),
                'Avg_Completeness': np.mean(var_comps),
                'Min_Completeness': min(var_comps),
                'Max_Completeness': max(var_comps),
                'Range': max(var_comps) - min(var_comps)
            })
    
    var_comp_df = pd.DataFrame(var_completeness)
    var_comp_df = var_comp_df.sort_values('Avg_Completeness', ascending=False)
    
    print("\nVariables with Highest Completeness (>90%):")
    print("-" * 60)
    high_complete = var_comp_df[var_comp_df['Avg_Completeness'] > 90]
    for _, row in high_complete.iterrows():
        print(f"  - {row['Variable']}: {row['Avg_Completeness']:.1f}% average completeness")
    
    print("\nVariables with Lowest Completeness (<50%):")
    print("-" * 60)
    low_complete = var_comp_df[var_comp_df['Avg_Completeness'] < 50]
    for _, row in low_complete.iterrows():
        print(f"  - {row['Variable']}: {row['Avg_Completeness']:.1f}% average completeness")
    
    print("\nVariables with Inconsistent Availability Across Wells:")
    print("-" * 60)
    inconsistent = var_comp_df[var_comp_df['Range'] > 50].sort_values('Range', ascending=False)
    for _, row in inconsistent.head(10).iterrows():
        print(f"  - {row['Variable']}: {row['Min_Completeness']:.1f}% to {row['Max_Completeness']:.1f}% (range: {row['Range']:.1f}%)")

#  visuals

LOG_DESCRIPTIONS = {
    'CN':   'Compensated Neutron Porosity',
    'CNC':  'Corrected Compensated Neutron Porosity',
    
    'GR':   'Gamma Ray',
    'GRSL': 'Gamma Ray from 1329 Spectrum',
    
    'HRD1': 'Long Space Spectrum Count Rate 140-200 keV',
    'HRD2': 'Long Space Spectrum Count Rate 200-540 keV',
    
    'K':    'Potassium Content',
    'KTH':  'Stripped Potassium-Thorium',
    
    'LSN':  'Long Space Neutron',
    'PE':   'Photo Electric Cross-Section',
    'SFT2': 'Long Space Spectrum Count Rate 100-140 keV',
    'SHR':  'Soft to Hard Count Rate Ratio',
    'SLTM': 'Delta Elapsed Time',
    'ZDNC': 'Borehole Size/Mud Weight Corrected Density'
}

def create_combined_correlation_heatmap(df_all, lab_columns, log_columns, well_correlations, min_samples=8):
    """
    Create heatmaps showing correlations calculated from combined data across all wells.
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells' data
    lab_columns : list
        Laboratory measurement columns to include
    log_columns : list
        Log measurement columns to include
    well_correlations : dict
        Dictionary with correlation matrices for each well (used for comparison)
    min_samples : int
        Minimum number of samples required per well for correlation calculation
        
    Returns:
    --------
    tuple:
        - combined_corr_matrix: DataFrame with combined correlations
        - sample_size_matrix: DataFrame with sample sizes for each pair
    
    Notes:
    ------
    "Combined" means all data points from all wells are pooled together into a single
    correlation calculation, rather than averaging the individual well correlations.
    This approach gives more weight to wells with more data points.
    """
    
    # Calculate combined correlations (pooling all data)
    combined_corr_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=float)
    sample_size_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=int)
    
    # Track which wells have sufficient data for each pair
    wells_with_data_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=object)
    
    for log_var in log_columns:
        for lab_var in lab_columns:
            # Get combined data
            combined_data = df_all[[log_var, lab_var, 'Well']].dropna()
            
            if len(combined_data) > 5:
                r, p = stats.pearsonr(combined_data[log_var], combined_data[lab_var])
                combined_corr_matrix.loc[log_var, lab_var] = r
                sample_size_matrix.loc[log_var, lab_var] = len(combined_data)
                
                # Count wells with sufficient data
                wells_with_sufficient_data = []
                for well in sorted(df_all['Well'].unique()):
                    well_data = df_all[df_all['Well'] == well][[log_var, lab_var]].dropna()
                    if len(well_data) >= min_samples:
                        wells_with_sufficient_data.append(well)
                
                wells_with_data_matrix.loc[log_var, lab_var] = wells_with_sufficient_data
            else:
                combined_corr_matrix.loc[log_var, lab_var] = np.nan
                sample_size_matrix.loc[log_var, lab_var] = 0
                wells_with_data_matrix.loc[log_var, lab_var] = []
    
    # Drop empty rows and columns
    combined_corr_matrix = combined_corr_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
    sample_size_matrix = sample_size_matrix.loc[combined_corr_matrix.index, combined_corr_matrix.columns]
    wells_with_data_matrix = wells_with_data_matrix.loc[combined_corr_matrix.index, combined_corr_matrix.columns]
    
    # --- Figure 1: Combined Correlations with Improved Annotations ---
    fig1, ax1 = plt.subplots(figsize=(18, 14))
    
    # Create custom annotations with correlation, number of wells, and sample size
    annot_text = np.empty_like(combined_corr_matrix, dtype=object)
    for i in range(len(combined_corr_matrix.index)):
        for j in range(len(combined_corr_matrix.columns)):
            r = combined_corr_matrix.iloc[i, j]
            n = sample_size_matrix.iloc[i, j]
            wells_list = wells_with_data_matrix.iloc[i, j]
            
            if pd.notna(r) and n > 0:
                n_wells = len(wells_list)
                # annot_text[i, j] = f'{r:.2f}\n({n_wells}w, n={n})'
                annot_text[i, j] = f'{r:.2f}\n({n_wells}w)'

            else:
                annot_text[i, j] = ''
    
    # Create heatmap with improved styling
    mask = pd.isna(combined_corr_matrix)
    
    sns.heatmap(
        combined_corr_matrix.astype(float),
        annot=annot_text,
        fmt='',
        annot_kws={'size': 8, 'va': 'center', 'ha': 'center', 'weight': 'bold'},
        cmap='RdYlGn',
        center=0, vmin=-1, vmax=1,
        cbar_kws={'label': 'Correlation (r)', 'shrink': 0.8, 'pad': 0.02},
        linewidths=0.8,
        linecolor='white',
        square=True,
        mask=mask,
        ax=ax1
    )
    
    # Improved title
    ax1.set_title('Log-Lab Correlations Across All Wells\n' + 
                  f'Values show: correlation (number of wells, sample size)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Laboratory Measurements', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Geophysical Log Measurements', fontsize=13, fontweight='bold')
    
    # Add explanation of "combined"
    explanation = ("Note: 'Combined' means all data points from all wells are pooled together\n"
                   "for a single correlation calculation, rather than averaging individual well correlations.")
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Clean tick labels
    ax1.set_xticklabels([col.replace('Lab_', '') for col in combined_corr_matrix.columns], 
                        rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels([row.replace('Log_', '') for row in combined_corr_matrix.index], 
                        rotation=0, fontsize=10)
    
    # Add grid for better readability
    ax1.set_facecolor('#f8f8f8')
    
    # Add LOG_DESCRIPTIONS legend in top left
    log_vars_in_plot = [row.replace('Log_', '') for row in combined_corr_matrix.index]
    legend_text = "Log Descriptions:\n" + "-"*50 + "\n"
    for log_var in log_vars_in_plot:
        if log_var in LOG_DESCRIPTIONS:
            legend_text += f"{log_var:<6}: {LOG_DESCRIPTIONS[log_var]}\n"
    
    # Add text box with descriptions in top left corner
    plt.figtext(0.02, 0.98, legend_text, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05)  # Make room for legend and explanation
    plt.savefig('imgs/heatmap_combined_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Figure 2: Strong correlations only with adjusted color scale ---
    fig2, ax2 = plt.subplots(figsize=(18, 14))
    
    # Create mask for weak correlations
    strong_threshold = 0.5
    mask_weak = np.abs(combined_corr_matrix) < strong_threshold
    
    # Determine if all strong correlations are positive or negative
    all_positive = (combined_corr_matrix[np.abs(combined_corr_matrix) >= strong_threshold] >= 0).all().all()
    all_negative = (combined_corr_matrix[np.abs(combined_corr_matrix) >= strong_threshold] <= 0).all().all()
    
    # Count strong correlations
    n_strong = (~mask_weak & ~pd.isna(combined_corr_matrix)).sum().sum()
    n_very_strong = (np.abs(combined_corr_matrix) >= 0.7).sum().sum()
    
    # Create annotations only for strong correlations
    strong_annot_text = np.empty_like(combined_corr_matrix, dtype=object)
    for i in range(len(combined_corr_matrix.index)):
        for j in range(len(combined_corr_matrix.columns)):
            r = combined_corr_matrix.iloc[i, j]
            n = sample_size_matrix.iloc[i, j]
            wells_list = wells_with_data_matrix.iloc[i, j]
            
            if pd.notna(r) and abs(r) >= strong_threshold and n > 0:
                n_wells = len(wells_list)
                # strong_annot_text[i, j] = f'{r:.2f}\n({n_wells}w, n={n})'
                strong_annot_text[i, j] = f'{r:.2f}\n({n_wells}w)'
            else:
                strong_annot_text[i, j] = ''
    
    # Use a narrower color range for better differentiation among strong correlations
    sns.heatmap(
        combined_corr_matrix.astype(float),
        annot=strong_annot_text,
        fmt='',
        annot_kws={'size': 8, 'va': 'center', 'ha': 'center', 'weight': 'bold'},
        cmap='RdYlGn',
        center=0, 
        vmin=0.5 if all_positive else -0.7 if all_negative else -1, 
        vmax=1 if all_positive else -0.5 if all_negative else 1,
        cbar_kws={'label': f'Strong Correlations (|r| ≥ {strong_threshold})', 'shrink': 0.8, 'pad': 0.02},
        linewidths=0.8,
        linecolor='white',
        square=True,
        mask=mask_weak,
        ax=ax2
    )
    
    # Improved title
    ax2.set_title(f'Strong Correlations Only (|r| ≥ {strong_threshold})\n' +
                  f'Values show: correlation (number of wells)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Laboratory Measurements', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Geophysical Log Measurements', fontsize=13, fontweight='bold')
    
    # Clean tick labels
    ax2.set_xticklabels([col.replace('Lab_', '') for col in combined_corr_matrix.columns], 
                        rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels([row.replace('Log_', '') for row in combined_corr_matrix.index], 
                        rotation=0, fontsize=10)
    
    # Add grid for better readability
    ax2.set_facecolor('#f8f8f8')
    
    # Add LOG_DESCRIPTIONS legend in top left for strong correlations plot too
    plt.figtext(0.02, 0.98, legend_text, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top', fontfamily='monospace')
    
    # Add explanation of "combined" here too
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.05)  # Make room for legend and explanation
    plt.savefig('imgs/heatmap_strong_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print improved summary statistics
    print("\n" + "="*80)
    print("COMBINED CORRELATION ANALYSIS SUMMARY")
    print("="*80)
    
    print("\nCORRELATION STRENGTH DISTRIBUTION:")
    print("-" * 60)
    total_valid = (~pd.isna(combined_corr_matrix)).sum().sum()
    for threshold, label in [(0.7, "Very Strong"), (0.5, "Strong"), (0.3, "Moderate")]:
        count = (np.abs(combined_corr_matrix) >= threshold).sum().sum()
        percentage = (count / total_valid) * 100 if total_valid > 0 else 0
        print(f"{label} (|r| ≥ {threshold}): {count} ({percentage:.1f}%)")
    
    print("\nSAMPLE SIZE DISTRIBUTION:")
    print("-" * 60)
    valid_sizes = sample_size_matrix[sample_size_matrix > 0].values.flatten()
    if len(valid_sizes) > 0:
        print(f"Range: {valid_sizes.min()} - {valid_sizes.max()} samples")
        print(f"Mean: {valid_sizes.mean():.0f} samples")
        print(f"Median: {np.median(valid_sizes):.0f} samples")
    
    # Find correlations with highest confidence (large n and strong r)
    confidence_scores = np.abs(combined_corr_matrix) * np.sqrt(sample_size_matrix)
    if not np.all(np.isnan(confidence_scores)):
        top_confidence_idx = np.unravel_index(np.nanargmax(confidence_scores), confidence_scores.shape)
        
        print("\nHIGHEST CONFIDENCE CORRELATION:")
        print("-" * 60)
        log_var = combined_corr_matrix.index[top_confidence_idx[0]]
        lab_var = combined_corr_matrix.columns[top_confidence_idx[1]]
        wells_for_top = wells_with_data_matrix.iloc[top_confidence_idx]
        print(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}")
        print(f"r = {combined_corr_matrix.iloc[top_confidence_idx]:.3f}, n = {sample_size_matrix.iloc[top_confidence_idx]}")
        print(f"Wells with sufficient data (≥{min_samples} samples): {len(wells_for_top)} wells")
    
    return combined_corr_matrix, sample_size_matrix

def create_comprehensive_correlation_scatter_plots(df_all, correlations_by_well_count, min_correlation, 
                                                 min_samples_per_well=8, max_plots_per_figure=20,
                                                 specific_logs=None, specific_labs=None):
    """
    Create comprehensive scatter plots for correlations that appear in multiple wells.
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells' data
    correlations_by_well_count : dict
        Dictionary of correlations organized by number of wells
    min_correlation : float
        Minimum absolute correlation threshold for inclusion
    min_samples_per_well : int
        Minimum number of samples required per well (default: 8)
    max_plots_per_figure : int
        Maximum number of plots to include in each summary figure
    specific_logs : list, optional
        List of specific log types to include (e.g., ['GR', 'PE']). If None, all logs are used.
    specific_labs : list, optional
        List of specific lab measurements to include. If None, all lab measurements are used.
        
    Returns:
    --------
    None
        Saves plots to imgs/scatter_plots/positive and imgs/scatter_plots/negative
    """
    
    # Define colors for each well
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#d62728', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#ff7f0e'
    }
    
    # Create directories for individual plots
    positive_dir = Path('imgs/scatter_plots/positive')
    negative_dir = Path('imgs/scatter_plots/negative')
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)
    
    # Function to sanitize filename
    def sanitize_filename(name):
        return name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Function to calculate combined correlation
    def calculate_combined_correlation(df_all, log_var, lab_var):
        """Calculate correlation using ALL data points pooled together"""
        valid_data = df_all[[log_var, lab_var]].dropna()
        if len(valid_data) >= min_samples_per_well:
            try:
                r, p = pearsonr(valid_data[log_var], valid_data[lab_var])
                return r
            except Exception as e:
                print(f"Error calculating correlation for {log_var} vs {lab_var}: {e}")
                return np.nan
        return np.nan
    
    # Get all possible log-lab combinations
    all_log_columns = [col for col in df_all.columns if col.startswith('Log_') and 
                   col not in ['Log_Depth', 'Log_FRAMENO']]
    all_lab_columns = [col for col in df_all.columns if col.startswith('Lab_') and 
                   col not in ['Lab_Depth', 'Lab_Sample_ID']]
    
    # Filter log columns based on specific_logs if provided
    if specific_logs:
        specific_log_columns = [f'Log_{log}' for log in specific_logs if f'Log_{log}' in df_all.columns]
        log_columns = specific_log_columns
        print(f"Using {len(log_columns)} specific log columns for scatter plots")
    else:
        log_columns = all_log_columns
        print(f"Using all {len(log_columns)} log columns for scatter plots")
    
    # Filter lab columns based on specific_labs if provided
    if specific_labs:
        specific_lab_columns = [f'Lab_{lab}' for lab in specific_labs if f'Lab_{lab}' in df_all.columns]
        lab_columns = specific_lab_columns
        print(f"Using {len(lab_columns)} specific lab columns for scatter plots")
    else:
        lab_columns = all_lab_columns
        print(f"Using all {len(lab_columns)} lab columns for scatter plots")
    
    # Calculate combined correlations for ALL pairs
    all_combined_correlations = []
    
    for log_var in log_columns:
        for lab_var in lab_columns:
            # Check which wells have sufficient data
            wells_with_data = []
            for well in df_all['Well'].unique():
                well_data = df_all[df_all['Well'] == well]
                valid_data = well_data[[log_var, lab_var]].dropna()
                if len(valid_data) >= min_samples_per_well:
                    try:
                        r, _ = pearsonr(valid_data[log_var], valid_data[lab_var])
                        wells_with_data.append((well, r))
                    except Exception as e:
                        print(f"Error with {well}, {log_var} vs {lab_var}: {e}")
            
            # Only proceed if we have data from 2+ wells
            if len(wells_with_data) >= 2:
                # Calculate combined correlation
                combined_r = calculate_combined_correlation(df_all, log_var, lab_var)
                
                # Only include if meets threshold
                if not np.isnan(combined_r) and abs(combined_r) >= min_correlation:
                    # Create info dictionary
                    correlations = [r for _, r in wells_with_data]
                    info = {
                        'avg_corr': np.mean(correlations),
                        'avg_abs_corr': np.mean([abs(r) for r in correlations]),
                        'std_corr': np.std(correlations),
                        'n_wells': len(wells_with_data),
                        'correlation_type': 'Positive' if combined_r > 0 else 'Negative',
                        'consistent_direction': all(r > 0 for _, r in wells_with_data) or all(r < 0 for _, r in wells_with_data),
                        'missing_wells': [well for well in df_all['Well'].unique() if well not in [w for w, _ in wells_with_data]]
                    }
                    
                    all_combined_correlations.append(((log_var, lab_var), wells_with_data, info, combined_r))
    
    if not all_combined_correlations:
        print("No correlations met the criteria. Try lowering min_correlation or min_samples_per_well.")
        return
        
    # Separate based on combined correlation sign
    positive_correlations = [(pair, wells_data, info, combined_r) 
                            for pair, wells_data, info, combined_r in all_combined_correlations 
                            if combined_r > 0]
    negative_correlations = [(pair, wells_data, info, combined_r) 
                            for pair, wells_data, info, combined_r in all_combined_correlations 
                            if combined_r < 0]
    
    # Sort by absolute combined correlation value (strongest first)
    positive_correlations.sort(key=lambda x: abs(x[3]), reverse=True)
    negative_correlations.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Remove combined_r from tuples for compatibility with rest of code
    all_positive_corrs = [(pair, wells_data, info) for pair, wells_data, info, combined_r in positive_correlations]
    all_negative_corrs = [(pair, wells_data, info) for pair, wells_data, info, combined_r in negative_correlations]
    
    print(f"\nCorrelation Classification Summary (2+ wells, |r| ≥ {min_correlation}):")
    print(f"Positive correlations: {len(all_positive_corrs)}")
    print(f"Negative correlations: {len(all_negative_corrs)}")
    print(f"Total correlations meeting threshold: {len(all_positive_corrs) + len(all_negative_corrs)}")
    
    # Function to create individual plot
    def create_individual_plot(pair, wells_data, info, correlation_type, save_path, combined_r):
        """Create and save an individual scatter plot."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        log_var, lab_var = pair
        all_x = []
        all_y = []
        
        # Get list of wells involved (shortened names)
        wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
        wells_str = ', '.join(wells_involved)
        
        # Colors based on correlation type
        edge_color = 'darkgreen' if correlation_type == 'positive' else 'darkred'
        bg_color = '#e8f5e9' if correlation_type == 'positive' else '#ffebee'
        line_color = 'darkgreen' if correlation_type == 'positive' else 'darkred'
        
        # Plot data for each well
        for well, r in wells_data:
            well_data = df_all[df_all['Well'] == well]
            
            mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
            x_data = well_data.loc[mask, log_var].values
            y_data = well_data.loc[mask, lab_var].values
            
            if len(x_data) > 0:
                well_short = well.replace('HRDH_', '')
                ax.scatter(x_data, y_data, 
                         color=well_colors[well], 
                         alpha=0.6, 
                         s=40,
                         label=f'{well_short} (r={r:.2f}, n={len(x_data)})',
                         edgecolors=edge_color,
                         linewidth=0.5)
                
                # Individual well regression line
                if len(x_data) > 1:
                    z = np.polyfit(x_data, y_data, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_data.min(), x_data.max(), 100)
                    ax.plot(x_line, p(x_line), color=well_colors[well], 
                           linestyle='--', alpha=0.5, linewidth=1.5)
                
                all_x.extend(x_data)
                all_y.extend(y_data)
        
        # Calculate combined statistics
        if len(all_x) > 3:
            all_x = np.array(all_x)
            all_y = np.array(all_y)
            
            # Use the passed combined_r instead of recalculating
            combined_p = stats.pearsonr(all_x, all_y)[1]
            
            # Calculate R-squared
            z = np.polyfit(all_x, all_y, 1)
            p = np.poly1d(z)
            y_pred = p(all_x)
            ss_res = np.sum((all_y - y_pred)**2)
            ss_tot = np.sum((all_y - np.mean(all_y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate 95% confidence interval
            n = len(all_x)
            t_val = stats.t.ppf(0.975, n-2)
            s_yx = np.sqrt(ss_res / (n-2))
            
            x_line = np.linspace(all_x.min(), all_x.max(), 100)
            y_line = p(x_line)
            
            x_mean = np.mean(all_x)
            se_line = s_yx * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((all_x - x_mean)**2))
            ci_upper = y_line + t_val * se_line
            ci_lower = y_line - t_val * se_line
            
            # Plot confidence interval
            ax.fill_between(x_line, ci_lower, ci_upper, 
                           color=line_color, alpha=0.1, 
                           label='95% CI')
            
            # Plot combined regression line
            ax.plot(x_line, y_line, color=line_color, 
                   linestyle='-', alpha=0.8, linewidth=3,
                   label=f'Combined: r={combined_r:.2f}, R²={r_squared:.2f}')
            
            # Add statistics box
            stats_text = f'p={combined_p:.2e}\nn={len(all_x)}\nSlope={z[0]:.3f}'
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9),
                   verticalalignment='bottom', horizontalalignment='right',
                   fontsize=10, color=line_color)
        
        # Styling
        ax.set_facecolor(bg_color)
        ax.set_xlabel(log_var.replace('Log_', ''), fontsize=13, fontweight='bold')
        ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=13, fontweight='bold')
        
        # Title
        title_line1 = f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}"
        title_line2 = f"Wells: {wells_str} | Combined r = {combined_r:.3f}"
        ax.set_title(f"{title_line1}\n{title_line2}", fontsize=12, pad=10, fontweight='bold')
        
        # Legend
        ax.legend(fontsize=9, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add subtle box around plot
        for spine in ax.spines.values():
            spine.set_edgecolor(line_color)
            spine.set_alpha(0.4)
            spine.set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Process positive correlations
    if all_positive_corrs:
        n_positive = min(len(all_positive_corrs), max_plots_per_figure)
        n_cols = min(4, n_positive) if n_positive > 1 else 1  # Changed n_plots to n_positive
        n_rows = (n_positive + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_positive == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        fig.suptitle(f'Positive Correlations in 2+ Wells (Top {n_positive} of {len(all_positive_corrs)} total)', 
                    fontsize=20, fontweight='bold', color='darkgreen', y=0.995)
        
        # Save individual positive plots
        print(f"\nSaving {len(all_positive_corrs)} individual positive correlation plots (2+ wells)...")
        
        for idx, ((pair, wells_data, info), (_, _, _, combined_r)) in enumerate(zip(all_positive_corrs, positive_correlations)):
            # Create filename with sanitized variable names
            log_var, lab_var = pair
            log_name = sanitize_filename(log_var.replace('Log_', ''))
            lab_name = sanitize_filename(lab_var.replace('Lab_', ''))
            filename = f"{idx+1:03d}_{log_name}_vs_{lab_name}_r{abs(combined_r):.2f}.png"
            save_path = positive_dir / filename
            
            # Create individual plot
            create_individual_plot(pair, wells_data, info, 'positive', save_path, combined_r)
            
            # Also create subplot in combined figure if within limit
            if idx < n_positive:
                ax = axes[idx]
                
                # Collect all data points for combined analysis
                all_x = []
                all_y = []
                
                # Get list of wells involved (shortened names)
                wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
                wells_str = ', '.join(wells_involved)
                
                # Plot data for each well with individual regression lines
                for well, r in wells_data:
                    well_data = df_all[df_all['Well'] == well]
                    mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                    x_data = well_data.loc[mask, log_var].values
                    y_data = well_data.loc[mask, lab_var].values
                    
                    if len(x_data) > 0:
                        well_short = well.replace('HRDH_', '')
                        ax.scatter(x_data, y_data, 
                                 color=well_colors[well], 
                                 alpha=0.6, 
                                 s=30,
                                 label=f'{well_short} (r={r:.2f})',
                                 edgecolors='darkgreen',
                                 linewidth=0.5)
                        
                        # Individual well regression line
                        if len(x_data) > 1:
                            z = np.polyfit(x_data, y_data, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_data.min(), x_data.max(), 100)
                            ax.plot(x_line, p(x_line), color=well_colors[well], 
                                   linestyle='--', alpha=0.5, linewidth=1)
                        
                        all_x.extend(x_data)
                        all_y.extend(y_data)
                
                # Calculate combined statistics from ALL data points
                if len(all_x) > 3:
                    all_x = np.array(all_x)
                    all_y = np.array(all_y)
                    
                    # Combined regression line
                    z = np.polyfit(all_x, all_y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(all_x.min(), all_x.max(), 100)
                    ax.plot(x_line, p(x_line), color='darkgreen', 
                           linestyle='-', alpha=0.8, linewidth=2.5,
                           label=f'Combined (r={combined_r:.2f})')
                
                # Shade background green for positive correlation
                ax.set_facecolor('#e8f5e9')
                
                # Formatting with improved title
                ax.set_xlabel(log_var.replace('Log_', ''), fontsize=11, fontweight='bold')
                ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=11, fontweight='bold')
                
                # Enhanced title with wells listed
                title_line1 = f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}"
                title_line2 = f"Wells: {wells_str} | Combined r = {combined_r:.3f}"
                ax.set_title(f"{title_line1}\n{title_line2}", fontsize=10, pad=8)
                
                # Improved legend
                ax.legend(fontsize=7, loc='best', framealpha=0.9, 
                         borderpad=0.3, columnspacing=0.5, handletextpad=0.3)
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                
                # Add subtle box around plot
                for spine in ax.spines.values():
                    spine.set_edgecolor('darkgreen')
                    spine.set_alpha(0.3)
                    spine.set_linewidth(1.5)
        
        # Hide empty subplots
        for idx in range(n_positive, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('imgs/scatter_all_positive_correlations_2plus_wells.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved {len(all_positive_corrs)} positive correlation plots to: {positive_dir}")
    
    # Process negative correlations
    if all_negative_corrs:
        n_negative = min(len(all_negative_corrs), max_plots_per_figure)
        n_cols = min(4, n_negative) if n_negative > 1 else 1  # Changed n_plots to n_negative
        n_rows = (n_negative + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_negative == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        fig.suptitle(f'Negative Correlations in 2+ Wells (Top {n_negative} of {len(all_negative_corrs)} total)', 
                    fontsize=20, fontweight='bold', color='darkred', y=0.995)
        
        # Save individual negative plots
        print(f"\nSaving {len(all_negative_corrs)} individual negative correlation plots (2+ wells)...")
        
        for idx, ((pair, wells_data, info), (_, _, _, combined_r)) in enumerate(zip(all_negative_corrs, negative_correlations)):
            # Create filename with sanitized variable names
            log_var, lab_var = pair
            log_name = sanitize_filename(log_var.replace('Log_', ''))
            lab_name = sanitize_filename(lab_var.replace('Lab_', ''))
            filename = f"{idx+1:03d}_{log_name}_vs_{lab_name}_r{abs(combined_r):.2f}.png"
            save_path = negative_dir / filename
            
            # Create individual plot
            create_individual_plot(pair, wells_data, info, 'negative', save_path, combined_r)
            
            # Also create subplot in combined figure if within limit
            if idx < n_negative:
                ax = axes[idx]
                
                # Collect all data points for combined analysis
                all_x = []
                all_y = []
                
                # Get list of wells involved (shortened names)
                wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
                wells_str = ', '.join(wells_involved)
                
                # Plot data for each well with individual regression lines
                for well, r in wells_data:
                    well_data = df_all[df_all['Well'] == well]
                    mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                    x_data = well_data.loc[mask, log_var].values
                    y_data = well_data.loc[mask, lab_var].values
                    
                    if len(x_data) > 0:
                        well_short = well.replace('HRDH_', '')
                        ax.scatter(x_data, y_data, 
                                 color=well_colors[well], 
                                 alpha=0.6, 
                                 s=30,
                                 label=f'{well_short} (r={r:.2f})',
                                 edgecolors='darkred',
                                 linewidth=0.5)
                        
                        # Individual well regression line
                        if len(x_data) > 1:
                            z = np.polyfit(x_data, y_data, 1)
                            p = np.poly1d(z)
                            x_line = np.linspace(x_data.min(), x_data.max(), 100)
                            ax.plot(x_line, p(x_line), color=well_colors[well], 
                                   linestyle='--', alpha=0.5, linewidth=1)
                        
                        all_x.extend(x_data)
                        all_y.extend(y_data)
                
                # Calculate combined statistics from ALL data points
                if len(all_x) > 3:
                    all_x = np.array(all_x)
                    all_y = np.array(all_y)
                    
                    # Combined regression line
                    z = np.polyfit(all_x, all_y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(all_x.min(), all_x.max(), 100)
                    ax.plot(x_line, p(x_line), color='darkred', 
                           linestyle='-', alpha=0.8, linewidth=2.5,
                           label=f'Combined (r={combined_r:.2f})')
                
                # Shade background red for negative correlation
                ax.set_facecolor('#ffebee')
                
                # Formatting with improved title
                ax.set_xlabel(log_var.replace('Log_', ''), fontsize=11, fontweight='bold')
                ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=11, fontweight='bold')
                
                # Enhanced title with wells listed
                title_line1 = f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}"
                title_line2 = f"Wells: {wells_str} | Combined r = {combined_r:.3f}"
                ax.set_title(f"{title_line1}\n{title_line2}", fontsize=10, pad=8)
                
                # Improved legend
                ax.legend(fontsize=7, loc='best', framealpha=0.9,
                         borderpad=0.3, columnspacing=0.5, handletextpad=0.3)
                ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                
                # Add subtle box around plot
                for spine in ax.spines.values():
                    spine.set_edgecolor('darkred')
                    spine.set_alpha(0.3)
                    spine.set_linewidth(1.5)
        
        # Hide empty subplots
        for idx in range(n_negative, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('imgs/scatter_all_negative_correlations_2plus_wells.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved {len(all_negative_corrs)} negative correlation plots to: {negative_dir}")
    
    # Create summary statistics
    print("\n" + "="*80)
    print(f"SCATTER PLOT GENERATION SUMMARY (2+ WELLS, {min_samples_per_well}+ SAMPLES/WELL)")
    print("="*80)
    
    print("\nINDIVIDUAL PLOTS SAVED:")
    print("-" * 60)
    print(f"Positive correlations: imgs/scatter_plots/positive/")
    print(f"   - {len(all_positive_corrs)} plots saved")
    print(f"Negative correlations: imgs/scatter_plots/negative/")
    print(f"   - {len(all_negative_corrs)} plots saved")

#pair plot
def create_advanced_well_pairplots_by_group(df_all, variable_groups, min_samples_per_well=5, sample_size=1000):
    """
    Create advanced pairplots for each variable group.
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells' data
    variable_groups : dict
        Dictionary of variable groups, each with 'logs' and 'labs' lists
    min_samples_per_well : int
        Minimum samples required for a well to be included in a correlation
    sample_size : int
        Maximum number of samples to use (for performance)
    """
    print("\nCreating advanced multi-well pairplots by variable group...")
    
    for group_name, group_info in variable_groups.items():
        print(f"\nProcessing group: {group_name}")
        
        # Skip Top_Correlations if empty
        if group_name == "Top_Correlations" and (not group_info['logs'] or not group_info['labs']):
            print(f"  Skipping {group_name}: No variables defined")
            continue
        
        # Combine logs and labs for this group
        variables = group_info['logs'] + group_info['labs']
        
        # Check data availability
        available_vars = [v for v in variables if v in df_all.columns]
        if len(available_vars) < 2:
            print(f"  Skipping {group_name}: Not enough variables available ({len(available_vars)})")
            continue
        
        print(f"  Creating pairplot with {len(available_vars)} variables: " + 
              ", ".join([v.replace('Log_', '').replace('Lab_', '') for v in available_vars]))
        
        # Create the advanced pairplot
        try:
            fig = create_advanced_well_pairplot(
                df_all, 
                available_vars, 
                min_samples_per_well=min_samples_per_well,
                sample_size=sample_size,
                filename=f"advanced_pairplot_{group_name.lower()}"
            )
            
            # Add the description as a subtitle
            fig.text(0.5, 0.96, group_info["description"], 
                   ha='center', fontsize=12, fontstyle='italic')
            
            # Save again with the updated title
            fig.savefig(f"imgs/advanced_pairplot_{group_name.lower()}.png", dpi=300, bbox_inches='tight')
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"  Error creating pairplot for {group_name}: {e}")
            

def create_advanced_well_pairplot(df_all, variables_to_plot, min_samples_per_well=5, sample_size=1000, 
                                 save_path='imgs/', filename=None):
    """
    Create an advanced pairplot for multi-well analysis with:
    - Histograms on diagonal (colored by well)
    - Scatter plots in upper triangle (colored by well with regression lines)
    - Correlation heatmap in lower triangle
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells' data
    variables_to_plot : list
        List of variables to include in the pairplot
    min_samples_per_well : int
        Minimum samples required for a well to be included in a correlation
    sample_size : int
        Maximum number of samples to use (for performance)
    save_path : str
        Directory to save the plot
    filename : str, optional
        Custom filename for the plot (without extension)
        
    Returns:
    --------
    fig : matplotlib Figure object
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr
    
    # Create variable friendly names (for display)
    var_friendly_names = {var: var.replace('Log_', '').replace('Lab_', '') for var in variables_to_plot}
    
    # Sample data if too large (but maintain well representation)
    wells = df_all['Well'].unique()
    if len(df_all) > sample_size:
        # Sample proportionally from each well
        sampled_data = []
        for well in wells:
            well_data = df_all[df_all['Well'] == well]
            well_sample_size = int(sample_size * (len(well_data) / len(df_all)))
            if well_sample_size > 0:
                sampled_data.append(well_data.sample(min(len(well_data), well_sample_size), random_state=42))
        
        df_plot = pd.concat(sampled_data)
        print(f"Sampled {len(df_plot)} points from {len(df_all)} total")
    else:
        df_plot = df_all.copy()
    
    # Define well colors (consistent with other visualizations)
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#d62728', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#ff7f0e'
    }
    
    # Create the grid of subplots (n×n for n variables)
    n_vars = len(variables_to_plot)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(2*n_vars, 2*n_vars))
    
    # Adjust for single variable case
    if n_vars == 1:
        axes = np.array([[axes]])
    
    # Calculate all correlations for each variable pair
    correlations = {}
    for i, var1 in enumerate(variables_to_plot):
        for j, var2 in enumerate(variables_to_plot):
            if i != j:  # Skip self-correlations
                # Calculate correlation using all data
                valid_data = df_all[[var1, var2]].dropna()
                if len(valid_data) >= min_samples_per_well * 2:
                    try:
                        r, p = pearsonr(valid_data[var1], valid_data[var2])
                        correlations[(var1, var2)] = (r, p, len(valid_data))
                    except:
                        correlations[(var1, var2)] = (np.nan, np.nan, 0)
    
    # Create histograms on diagonal
    for i, var in enumerate(variables_to_plot):
        ax = axes[i, i]
        # Create separate histograms for each well
        for well in wells:
            well_data = df_plot[df_plot['Well'] == well]
            if len(well_data) > 0:
                sns.histplot(well_data[var].dropna(), 
                           ax=ax, 
                           color=well_colors[well], 
                           alpha=0.5,
                           label=well)
        
        # Set labels
        ax.set_xlabel(var_friendly_names[var])
        ax.set_ylabel('Count')
        
        # Only show legend for the first diagonal plot
        if i == 0:
            ax.legend(fontsize=8)
        
        # Remove y-ticks for cleaner look
        ax.set_yticks([])
    
    # Create scatter plots in upper triangle, correlation info in lower
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:  # Skip diagonal
                var1 = variables_to_plot[j]  # x-axis
                var2 = variables_to_plot[i]  # y-axis
                ax = axes[i, j]
                
                if i < j:  # Upper triangle - scatter plots
                    # Plot each well with different color
                    for well in wells:
                        well_data = df_plot[df_plot['Well'] == well]
                        valid_data = well_data[[var1, var2]].dropna()
                        
                        if len(valid_data) > 0:
                            ax.scatter(valid_data[var1], valid_data[var2], 
                                     color=well_colors[well], 
                                     alpha=0.7, s=20, 
                                     label=well)
                            
                            # Add regression line for each well if enough points
                            if len(valid_data) >= min_samples_per_well:
                                try:
                                    x = valid_data[var1]
                                    y = valid_data[var2]
                                    
                                    # Fit line
                                    z = np.polyfit(x, y, 1)
                                    p = np.poly1d(z)
                                    
                                    # Plot line
                                    x_line = np.linspace(x.min(), x.max(), 100)
                                    ax.plot(x_line, p(x_line), color=well_colors[well], 
                                           linestyle='--', alpha=0.7, linewidth=1)
                                except Exception as e:
                                    pass
                    
                    # Set labels
                    ax.set_xlabel(var_friendly_names[var1])
                    ax.set_ylabel(var_friendly_names[var2])
                    
                    # Add grid
                    ax.grid(True, linestyle=':', alpha=0.3)
                
                else:  # Lower triangle - correlation heatmap
                    # Clear the axis for heatmap
                    ax.clear()
                    
                    # Get correlation if available
                    if (var1, var2) in correlations:
                        r, p, n = correlations[(var1, var2)]
                    else:
                        r, p, n = np.nan, np.nan, 0
                    
                    # Add correlation as a color-coded cell
                    if not np.isnan(r):
                        # Create a single-cell "heatmap"
                        # Color based on correlation strength and direction
                        if r > 0:
                            color = plt.cm.Greens(0.5 + r/2)  # Map [0,1] to [0.5,1] in colormap
                        else:
                            color = plt.cm.Reds(0.5 - r/2)    # Map [-1,0] to [1,0.5] in colormap
                        
                        # Fill the entire axis with this color
                        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                                 color=color, alpha=0.7))
                        
                        # Add correlation text
                        sig_stars = ''
                        if p < 0.001:
                            sig_stars = '***'
                        elif p < 0.01:
                            sig_stars = '**'
                        elif p < 0.05:
                            sig_stars = '*'
                        
                        # Show correlation value and significance stars
                        ax.text(0.5, 0.5, f'r = {r:.2f}{sig_stars}', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=10, fontweight='bold',
                               color='black' if abs(r) < 0.7 else 'white')
                        
                        # Add sample size
                        ax.text(0.5, 0.25, f'n = {n}', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=8,
                               color='black' if abs(r) < 0.7 else 'white')
                    
                    # Remove ticks and spines for clean look
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
    
    # Add a title explaining significance
    plt.figtext(0.5, 0.01, 
               "* p<0.05, ** p<0.01, *** p<0.001", 
               ha='center', fontsize=10)
    
    # Add an overall title
    fig.suptitle("Multi-Well Correlation Analysis", fontsize=16, y=0.98)
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save the plot
    if filename is None:
        # Create a filename based on the variables
        var_names = [var_friendly_names[v] for v in variables_to_plot]
        filename = f"advanced_pairplot_{'_'.join(var_names[:3])}"
        if len(var_names) > 3:
            filename += "_etc"
    
    plt.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight')
    
    # Return the figure
    return fig