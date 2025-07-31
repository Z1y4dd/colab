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
#combined r
def create_combined_correlation_heatmap(df_all, lab_columns, log_columns, well_correlations, min_samples=8, common_correlations=None):
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
    common_correlations : DataFrame, optional
        DataFrame from find_common_correlations to ensure consistent well counts
        
    Returns:
    --------
    tuple:
        - combined_corr_matrix: DataFrame with combined correlations
        - sample_size_matrix: DataFrame with sample sizes for each pair
    """
    
    # Calculate combined correlations (pooling all data)
    combined_corr_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=float)
    sample_size_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=int)
    
    # Track which wells have sufficient data for each pair
    wells_with_data_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=object)
    
    # NEW: Create a lookup for well counts from common_correlations
    well_count_lookup = {}
    if common_correlations is not None:
        for _, row in common_correlations.iterrows():
            key = (row['log_var'], row['lab_var'])
            well_count_lookup[key] = row['num_wells']
    
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
                # NEW: Use well count from common_correlations if available
                log_var = combined_corr_matrix.index[i]
                lab_var = combined_corr_matrix.columns[j]
                
                if (log_var, lab_var) in well_count_lookup:
                    n_wells = well_count_lookup[(log_var, lab_var)]
                else:
                    n_wells = len(wells_list)
                
                # Color-code the well count markers
                if n_wells == 2:
                    well_text_color = 'blue'
                elif n_wells == 3:
                    well_text_color = 'green'
                else:  # 4 wells
                    well_text_color = 'purple'
                
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
                  f'Values show: correlation (number of wells from common_correlations analysis)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Laboratory Measurements', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Geophysical Log Measurements', fontsize=13, fontweight='bold')
    
    # Add explanation of "combined"
    explanation = ("Note: 'Combined' means all data points from all wells are pooled together\n"
                   "for a single correlation calculation, rather than averaging individual well correlations.\n"
                   "Well counts (2w, 3w, 4w) are from the common_correlations analysis.")
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
                # NEW: Use well count from common_correlations if available
                log_var = combined_corr_matrix.index[i]
                lab_var = combined_corr_matrix.columns[j]
                
                if (log_var, lab_var) in well_count_lookup:
                    n_wells = well_count_lookup[(log_var, lab_var)]
                else:
                    n_wells = len(wells_list)
                    
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
                  f'Values show: correlation (number of wells from common_correlations analysis)', 
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

#avg_r
# def create_comprehensive_correlation_scatter_plots_from_existing(df_all, correlations_by_well_count, min_correlation, 
#                                                  min_samples_per_well=8, max_plots_per_figure=20):
#     """
#     Create comprehensive scatter plots using pre-calculated correlations from common_correlations.
#     This ensures consistency with the well counts determined during analysis.
    
#     Parameters:
#     -----------
#     df_all : DataFrame
#         Combined dataframe with all wells' data
#     correlations_by_well_count : dict
#         Dictionary of correlations organized by number of wells (from common_correlations)
#     min_correlation : float
#         Minimum absolute correlation threshold (for display purposes only)
#     min_samples_per_well : int
#         Minimum number of samples required per well (for plotting)
#     max_plots_per_figure : int
#         Maximum number of plots to include in each summary figure
        
#     Returns:
#     --------
#     None
#         Saves plots to imgs/scatter_by_wells/
#     """
#     from pathlib import Path
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     # Define well colors
#     well_colors = {
#         'HRDH_697': '#1f77b4',
#         'HRDH_1119': '#d62728', 
#         'HRDH_1804': '#2ca02c',
#         'HRDH_1867': '#ff7f0e'
#     }
    
#     # Create directories for plots organized by well count
#     for n_wells in [2, 3, 4]:
#         Path(f'imgs/scatter_by_wells/{n_wells}_wells').mkdir(parents=True, exist_ok=True)
    
#     # Process each well count separately
#     for n_wells in [2, 3, 4]:
#         if n_wells not in correlations_by_well_count or not correlations_by_well_count[n_wells]:
#             continue
            
#         print(f"\nCreating scatter plots for {n_wells}-well correlations...")
        
#         # Get all correlations for this well count
#         correlations = correlations_by_well_count[n_wells]
        
#         # Separate positive and negative
#         positive_corrs = [(pair, wells_data, info) for pair, wells_data, info in correlations 
#                          if info['avg_corr'] > 0]
#         negative_corrs = [(pair, wells_data, info) for pair, wells_data, info in correlations 
#                          if info['avg_corr'] < 0]
        
#         # Create summary figures for positive correlations
#         if positive_corrs:
#             n_plots = min(len(positive_corrs), max_plots_per_figure)
#             n_cols = min(4, n_plots)
#             n_rows = (n_plots + n_cols - 1) // n_cols
            
#             fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#             if n_plots == 1:
#                 axes = [axes]
#             else:
#                 axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
#             fig.suptitle(f'{n_wells}-Well Positive Correlations (Top {n_plots} of {len(positive_corrs)} total)', 
#                         fontsize=16, fontweight='bold', color='darkgreen')
            
#             for idx, (pair, wells_data, info) in enumerate(positive_corrs[:n_plots]):
#                 ax = axes[idx]
#                 log_var, lab_var = pair
                
#                 # Plot ONLY the wells that have this correlation
#                 all_x = []
#                 all_y = []
                
#                 for well, well_r in wells_data:
#                     well_data = df_all[df_all['Well'] == well]
#                     mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
#                     x_data = well_data.loc[mask, log_var].values
#                     y_data = well_data.loc[mask, lab_var].values
                    
#                     if len(x_data) > 0:
#                         well_short = well.replace('HRDH_', '')
#                         ax.scatter(x_data, y_data, 
#                                  color=well_colors[well], 
#                                  alpha=0.6, s=30,
#                                  label=f'{well_short} (r={well_r:.2f})',
#                                  edgecolors='darkgreen',
#                                  linewidth=0.5)
                        
#                         all_x.extend(x_data)
#                         all_y.extend(y_data)
                
#                 # Add regression line using ONLY data from the wells with this correlation
#                 if len(all_x) > 3:
#                     all_x = np.array(all_x)
#                     all_y = np.array(all_y)
#                     z = np.polyfit(all_x, all_y, 1)
#                     p = np.poly1d(z)
#                     x_line = np.linspace(all_x.min(), all_x.max(), 100)
#                     ax.plot(x_line, p(x_line), 
#                            color='darkgreen', 
#                            linestyle='-', alpha=0.8, linewidth=2,
#                            label=f'Avg r={info["avg_corr"]:.2f}')
                
#                 # Styling
#                 ax.set_facecolor('#e8f5e9')
#                 ax.set_xlabel(log_var.replace('Log_', ''), fontsize=10)
#                 ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=10)
                
#                 # Title with ONLY the wells that have this correlation
#                 wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
#                 ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str})", 
#                            fontsize=9)
                
#                 ax.legend(fontsize=7, loc='best')
#                 ax.grid(True, alpha=0.3)
            
#             # Hide empty subplots
#             for idx in range(n_plots, len(axes)):
#                 axes[idx].set_visible(False)
            
#             plt.tight_layout()
#             plt.savefig(f'imgs/scatter_by_wells/{n_wells}_wells/positive_{n_wells}_well_correlations.png', 
#                        dpi=300, bbox_inches='tight')
#             plt.show()
        
#         # Create summary figures for negative correlations
#         if negative_corrs:
#             n_plots = min(len(negative_corrs), max_plots_per_figure)
#             n_cols = min(4, n_plots)
#             n_rows = (n_plots + n_cols - 1) // n_cols
            
#             fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#             if n_plots == 1:
#                 axes = [axes]
#             else:
#                 axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
#             fig.suptitle(f'{n_wells}-Well Negative Correlations (Top {n_plots} of {len(negative_corrs)} total)', 
#                         fontsize=16, fontweight='bold', color='darkred')
            
#             for idx, (pair, wells_data, info) in enumerate(negative_corrs[:n_plots]):
#                 ax = axes[idx]
#                 log_var, lab_var = pair
                
#                 # Plot ONLY the wells that have this correlation
#                 all_x = []
#                 all_y = []
                
#                 for well, well_r in wells_data:
#                     well_data = df_all[df_all['Well'] == well]
#                     mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
#                     x_data = well_data.loc[mask, log_var].values
#                     y_data = well_data.loc[mask, lab_var].values
                    
#                     if len(x_data) > 0:
#                         well_short = well.replace('HRDH_', '')
#                         ax.scatter(x_data, y_data, 
#                                  color=well_colors[well], 
#                                  alpha=0.6, s=30,
#                                  label=f'{well_short} (r={well_r:.2f})',
#                                  edgecolors='darkred',
#                                  linewidth=0.5)
                        
#                         all_x.extend(x_data)
#                         all_y.extend(y_data)
                
#                 # Add regression line using ONLY data from the wells with this correlation
#                 if len(all_x) > 3:
#                     all_x = np.array(all_x)
#                     all_y = np.array(all_y)
#                     z = np.polyfit(all_x, all_y, 1)
#                     p = np.poly1d(z)
#                     x_line = np.linspace(all_x.min(), all_x.max(), 100)
#                     ax.plot(x_line, p(x_line), 
#                            color='darkred', 
#                            linestyle='-', alpha=0.8, linewidth=2,
#                            label=f'Avg r={info["avg_corr"]:.2f}')
                
#                 # Styling
#                 ax.set_facecolor('#ffebee')
#                 ax.set_xlabel(log_var.replace('Log_', ''), fontsize=10)
#                 ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=10)
                
#                 # Title with ONLY the wells that have this correlation
#                 wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
#                 ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str})", 
#                            fontsize=9)
                
#                 ax.legend(fontsize=7, loc='best')
#                 ax.grid(True, alpha=0.3)
            
#             # Hide empty subplots
#             for idx in range(n_plots, len(axes)):
#                 axes[idx].set_visible(False)
            
#             plt.tight_layout()
#             plt.savefig(f'imgs/scatter_by_wells/{n_wells}_wells/negative_{n_wells}_well_correlations.png', 
#                        dpi=300, bbox_inches='tight')
#             plt.show()
    
#     # Print summary
#     print("\n" + "="*80)
#     print("SCATTER PLOT GENERATION SUMMARY (FROM COMMON_CORRELATIONS)")
#     print("="*80)
    
#     for n_wells in [2, 3, 4]:
#         if n_wells in correlations_by_well_count:
#             corrs = correlations_by_well_count[n_wells]
#             pos_count = sum(1 for _, _, info in corrs if info['avg_corr'] > 0)
#             neg_count = sum(1 for _, _, info in corrs if info['avg_corr'] < 0)
#             print(f"\n{n_wells}-well correlations:")
#             print(f"  Positive: {pos_count}")
#             print(f"  Negative: {neg_count}")
#             print(f"  Total: {len(corrs)}")


# shade + summary
# def create_comprehensive_correlation_scatter_plots_from_existing(df_all, correlations_by_well_count, min_correlation, 
#                                                  min_samples_per_well=8, max_plots_per_figure=20):
#     """
#     Create comprehensive scatter plots using pre-calculated correlations from common_correlations.
#     Includes confidence interval shading and summary plots.
    
#     Parameters:
#     -----------
#     df_all : DataFrame
#         Combined dataframe with all wells' data
#     correlations_by_well_count : dict
#         Dictionary of correlations organized by number of wells (from common_correlations)
#     min_correlation : float
#         Minimum absolute correlation threshold (for display purposes only)
#     min_samples_per_well : int
#         Minimum number of samples required per well (for plotting)
#     max_plots_per_figure : int
#         Maximum number of plots to include in each summary figure
        
#     Returns:
#     --------
#     None
#         Saves plots to imgs/scatter_by_wells/
#     """
#     from pathlib import Path
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from scipy import stats
#     import seaborn as sns
    
#     # Define well colors
#     well_colors = {
#         'HRDH_697': '#1f77b4',
#         'HRDH_1119': '#d62728', 
#         'HRDH_1804': '#2ca02c',
#         'HRDH_1867': '#ff7f0e'
#     }
    
#     # Create directories for plots organized by well count
#     for n_wells in [2, 3, 4]:
#         Path(f'imgs/scatter_by_wells/{n_wells}_wells').mkdir(parents=True, exist_ok=True)
#     Path('imgs/scatter_by_wells/summary').mkdir(parents=True, exist_ok=True)
    
#     # Collect all correlations for summary
#     all_correlations_for_summary = []
    
#     # Process each well count separately
#     for n_wells in [2, 3, 4]:
#         if n_wells not in correlations_by_well_count or not correlations_by_well_count[n_wells]:
#             continue
            
#         print(f"\nCreating scatter plots for {n_wells}-well correlations...")
        
#         # Get all correlations for this well count
#         correlations = correlations_by_well_count[n_wells]
        
#         # Calculate combined correlations for sorting
#         correlations_with_combined = []
#         for pair, wells_data, info in correlations:
#             log_var, lab_var = pair
            
#             # Get data only from the wells in wells_data
#             well_names = [w for w, _ in wells_data]
#             mask = df_all['Well'].isin(well_names)
#             combined_data = df_all[mask][[log_var, lab_var]].dropna()
            
#             if len(combined_data) > 5:
#                 combined_r, _ = stats.pearsonr(combined_data[log_var], combined_data[lab_var])
#                 n_points = len(combined_data)
#             else:
#                 combined_r = info['avg_corr']  # Fallback to average
#                 n_points = 0
                
#             correlations_with_combined.append((pair, wells_data, info, combined_r, n_points, n_wells))
#             all_correlations_for_summary.append((pair, wells_data, info, combined_r, n_points, n_wells))
        
#         # Sort by absolute combined correlation
#         correlations_with_combined.sort(key=lambda x: abs(x[3]), reverse=True)
        
#         # Separate positive and negative
#         positive_corrs = [(pair, wells_data, info) for pair, wells_data, info in correlations 
#                          if info['avg_corr'] > 0]
#         negative_corrs = [(pair, wells_data, info) for pair, wells_data, info in correlations 
#                          if info['avg_corr'] < 0]
        
#         # Create summary figures for positive correlations
#         if positive_corrs:
#             n_plots = min(len(positive_corrs), max_plots_per_figure)
#             n_cols = min(4, n_plots)
#             n_rows = (n_plots + n_cols - 1) // n_cols
            
#             fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#             if n_plots == 1:
#                 axes = [axes]
#             else:
#                 axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
#             fig.suptitle(f'{n_wells}-Well Positive Correlations (Top {n_plots} of {len(positive_corrs)} total)', 
#                         fontsize=16, fontweight='bold', color='darkgreen')
            
#             for idx, (pair, wells_data, info) in enumerate(positive_corrs[:n_plots]):
#                 ax = axes[idx]
#                 log_var, lab_var = pair
                
#                 # Plot ONLY the wells that have this correlation
#                 all_x = []
#                 all_y = []
                
#                 for well, well_r in wells_data:
#                     well_data = df_all[df_all['Well'] == well]
#                     mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
#                     x_data = well_data.loc[mask, log_var].values
#                     y_data = well_data.loc[mask, lab_var].values
                    
#                     if len(x_data) > 0:
#                         well_short = well.replace('HRDH_', '')
#                         ax.scatter(x_data, y_data, 
#                                  color=well_colors[well], 
#                                  alpha=0.6, s=30,
#                                  label=f'{well_short} (r={well_r:.2f})',
#                                  edgecolors='darkgreen',
#                                  linewidth=0.5)
                        
#                         all_x.extend(x_data)
#                         all_y.extend(y_data)
                
#                 # Add regression line using ONLY data from the wells with this correlation
#                 if len(all_x) > 3:
#                     all_x = np.array(all_x)
#                     all_y = np.array(all_y)
#                     z = np.polyfit(all_x, all_y, 1)
#                     p = np.poly1d(z)
#                     x_line = np.linspace(all_x.min(), all_x.max(), 100)
#                     ax.plot(x_line, p(x_line), 
#                            color='darkgreen', 
#                            linestyle='-', alpha=0.8, linewidth=2,
#                            label=f'Avg r={info["avg_corr"]:.2f}')
                
#                 # Styling
#                 ax.set_facecolor('#e8f5e9')
#                 ax.set_xlabel(log_var.replace('Log_', ''), fontsize=10)
#                 ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=10)
                
#                 # Title with ONLY the wells that have this correlation
#                 wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
#                 ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str})", 
#                            fontsize=9)
                
#                 ax.legend(fontsize=7, loc='best')
#                 ax.grid(True, alpha=0.3)
            
#             # Hide empty subplots
#             for idx in range(n_plots, len(axes)):
#                 axes[idx].set_visible(False)
            
#             plt.tight_layout()
#             plt.savefig(f'imgs/scatter_by_wells/{n_wells}_wells/positive_{n_wells}_well_correlations.png', 
#                        dpi=300, bbox_inches='tight')
#             plt.show()
        
#         # Create summary figures for negative correlations
#         if negative_corrs:
#             n_plots = min(len(negative_corrs), max_plots_per_figure)
#             n_cols = min(4, n_plots)
#             n_rows = (n_plots + n_cols - 1) // n_cols
            
#             fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
#             if n_plots == 1:
#                 axes = [axes]
#             else:
#                 axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
            
#             fig.suptitle(f'{n_wells}-Well Negative Correlations (Top {n_plots} of {len(negative_corrs)} total)', 
#                         fontsize=16, fontweight='bold', color='darkred')
            
#             for idx, (pair, wells_data, info) in enumerate(negative_corrs[:n_plots]):
#                 ax = axes[idx]
#                 log_var, lab_var = pair
                
#                 # Plot ONLY the wells that have this correlation
#                 all_x = []
#                 all_y = []
                
#                 for well, well_r in wells_data:
#                     well_data = df_all[df_all['Well'] == well]
#                     mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
#                     x_data = well_data.loc[mask, log_var].values
#                     y_data = well_data.loc[mask, lab_var].values
                    
#                     if len(x_data) > 0:
#                         well_short = well.replace('HRDH_', '')
#                         ax.scatter(x_data, y_data, 
#                                  color=well_colors[well], 
#                                  alpha=0.6, s=30,
#                                  label=f'{well_short} (r={well_r:.2f})',
#                                  edgecolors='darkred',
#                                  linewidth=0.5)
                        
#                         all_x.extend(x_data)
#                         all_y.extend(y_data)
                
#                 # Add regression line using ONLY data from the wells with this correlation
#                 if len(all_x) > 3:
#                     all_x = np.array(all_x)
#                     all_y = np.array(all_y)
#                     z = np.polyfit(all_x, all_y, 1)
#                     p = np.poly1d(z)
#                     x_line = np.linspace(all_x.min(), all_x.max(), 100)
#                     ax.plot(x_line, p(x_line), 
#                            color='darkred', 
#                            linestyle='-', alpha=0.8, linewidth=2,
#                            label=f'Avg r={info["avg_corr"]:.2f}')
                
#                 # Styling
#                 ax.set_facecolor('#ffebee')
#                 ax.set_xlabel(log_var.replace('Log_', ''), fontsize=10)
#                 ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=10)
                
#                 # Title with ONLY the wells that have this correlation
#                 wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
#                 ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str})", 
#                            fontsize=9)
                
#                 ax.legend(fontsize=7, loc='best')
#                 ax.grid(True, alpha=0.3)
            
#             # Hide empty subplots
#             for idx in range(n_plots, len(axes)):
#                 axes[idx].set_visible(False)
            
#             plt.tight_layout()
#             plt.savefig(f'imgs/scatter_by_wells/{n_wells}_wells/negative_{n_wells}_well_correlations.png', 
#                        dpi=300, bbox_inches='tight')
#             plt.show()
    
#     # Create summary bar chart
#     if all_correlations_for_summary:
#         # Sort all correlations by absolute combined correlation
#         all_correlations_for_summary.sort(key=lambda x: abs(x[3]), reverse=True)
        
#         # Create summary figure showing top correlations across all well counts
#         n_top = min(30, len(all_correlations_for_summary))  # Show top 30
        
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
#         # Prepare data for plotting
#         labels = []
#         colors = []
#         values = []
#         well_counts = []
        
#         for pair, wells_data, info, combined_r, n_points, n_wells in all_correlations_for_summary[:n_top]:
#             log_var = pair[0]
#             lab_var = pair[1]
#             wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
#             label = f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str})"
#             labels.append(label)
#             values.append(combined_r)
#             well_counts.append(n_wells)
            
#             # Color based on correlation direction and well count
#             if combined_r > 0:
#                 if n_wells == 4:
#                     colors.append('#006400')  # Dark green for 4-well positive
#                 elif n_wells == 3:
#                     colors.append('#228B22')  # Forest green for 3-well positive
#                 else:
#                     colors.append('#90EE90')  # Light green for 2-well positive
#             else:
#                 if n_wells == 4:
#                     colors.append('#8B0000')  # Dark red for 4-well negative
#                 elif n_wells == 3:
#                     colors.append('#DC143C')  # Crimson for 3-well negative
#                 else:
#                     colors.append('#FFA07A')  # Light salmon for 2-well negative
        
#         # Plot 1: Horizontal bar chart of all top correlations
#         y_pos = np.arange(len(labels))
#         bars = ax1.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
#         # Add value labels on bars
#         for i, (bar, value) in enumerate(zip(bars, values)):
#             x_pos = value + 0.01 if value > 0 else value - 0.01
#             ax1.text(x_pos, bar.get_y() + bar.get_height()/2, 
#                     f'{value:.3f}', 
#                     ha='left' if value > 0 else 'right', 
#                     va='center', fontsize=8)
        
#         ax1.set_yticks(y_pos)
#         ax1.set_yticklabels(labels, fontsize=8)
#         ax1.set_xlabel('Combined Correlation (r)', fontsize=12, fontweight='bold')
#         ax1.set_title(f'Top {n_top} Correlations Across All Wells\nSorted by Absolute Correlation Strength', 
#                      fontsize=14, fontweight='bold')
#         ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
#         ax1.grid(True, axis='x', alpha=0.3)
#         ax1.set_xlim(-1.1, 1.1)
        
#         # Add legend for well counts
#         from matplotlib.patches import Patch
#         legend_elements = [
#             Patch(facecolor='#006400', label='4-well positive'),
#             Patch(facecolor='#228B22', label='3-well positive'),
#             Patch(facecolor='#90EE90', label='2-well positive'),
#             Patch(facecolor='#8B0000', label='4-well negative'),
#             Patch(facecolor='#DC143C', label='3-well negative'),
#             Patch(facecolor='#FFA07A', label='2-well negative')
#         ]
#         ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)
        
#         # Plot 2: Summary statistics by well count
#         well_count_stats = {2: {'pos': 0, 'neg': 0, 'total': 0},
#                            3: {'pos': 0, 'neg': 0, 'total': 0},
#                            4: {'pos': 0, 'neg': 0, 'total': 0}}
        
#         for _, _, _, combined_r, _, n_wells in all_correlations_for_summary:
#             if n_wells in well_count_stats:
#                 well_count_stats[n_wells]['total'] += 1
#                 if combined_r > 0:
#                     well_count_stats[n_wells]['pos'] += 1
#                 else:
#                     well_count_stats[n_wells]['neg'] += 1
        
#         # Create grouped bar chart
#         x = np.arange(3)
#         width = 0.25
        
#         pos_counts = [well_count_stats[i]['pos'] for i in [2, 3, 4]]
#         neg_counts = [well_count_stats[i]['neg'] for i in [2, 3, 4]]
#         total_counts = [well_count_stats[i]['total'] for i in [2, 3, 4]]
        
#         bars1 = ax2.bar(x - width, pos_counts, width, label='Positive', color='green', alpha=0.8)
#         bars2 = ax2.bar(x, neg_counts, width, label='Negative', color='red', alpha=0.8)
#         bars3 = ax2.bar(x + width, total_counts, width, label='Total', color='blue', alpha=0.6)
        
#         # Add value labels on bars
#         for bars in [bars1, bars2, bars3]:
#             for bar in bars:
#                 height = bar.get_height()
#                 ax2.text(bar.get_x() + bar.get_width()/2., height,
#                         f'{int(height)}',
#                         ha='center', va='bottom', fontsize=10)
        
#         ax2.set_xlabel('Number of Wells', fontsize=12, fontweight='bold')
#         ax2.set_ylabel('Number of Correlations', fontsize=12, fontweight='bold')
#         ax2.set_title('Correlation Count by Number of Wells', fontsize=14, fontweight='bold')
#         ax2.set_xticks(x)
#         ax2.set_xticklabels(['2 wells', '3 wells', '4 wells'])
#         ax2.legend()
#         ax2.grid(True, axis='y', alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('imgs/scatter_by_wells/summary/all_correlations_summary.png', dpi=300, bbox_inches='tight')
#         plt.show()
    
#     # Print summary
#     print("\n" + "="*80)
#     print("SCATTER PLOT GENERATION SUMMARY (FROM COMMON_CORRELATIONS)")
#     print("="*80)
    
#     total_positive = 0
#     total_negative = 0
    
#     for n_wells in [2, 3, 4]:
#         if n_wells in correlations_by_well_count:
#             corrs = correlations_by_well_count[n_wells]
#             pos_count = sum(1 for _, _, info in corrs if info['avg_corr'] > 0)
#             neg_count = sum(1 for _, _, info in corrs if info['avg_corr'] < 0)
#             total_positive += pos_count
#             total_negative += neg_count
#             print(f"\n{n_wells}-well correlations:")
#             print(f"  Positive: {pos_count}")
#             print(f"  Negative: {neg_count}")
#             print(f"  Total: {len(corrs)}")
    
#     print(f"\nOverall totals:")
#     print(f"  Total positive correlations: {total_positive}")
#     print(f"  Total negative correlations: {total_negative}")
#     print(f"  Grand total: {total_positive + total_negative}")

# final
def create_comprehensive_correlation_scatter_plots_from_existing(
    df_all, correlations_by_well_count, min_correlation,
    min_samples_per_well=8, max_plots_per_figure=20
):
    """
    Create comprehensive scatter plots using pre-calculated correlations from common_correlations.
    Includes individual plots, summary plots, and overall summaries.

    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells' data
    correlations_by_well_count : dict
        Dictionary of correlations organized by number of wells (from common_correlations)
    min_correlation : float
        Minimum absolute correlation threshold (for display purposes only)
    min_samples_per_well : int
        Minimum number of samples required per well (for plotting)
    max_plots_per_figure : int
        Maximum number of plots to include in each summary figure

    Returns:
    --------
    None
        Saves plots to imgs/scatter_plots/
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import seaborn as sns
    import re

    # Define well colors
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#d62728',
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#ff7f0e'
    }

    # Create directories inside imgs
    Path('imgs').mkdir(parents=True, exist_ok=True)
    Path('imgs/scatter_plots/positive').mkdir(parents=True, exist_ok=True)
    Path('imgs/scatter_plots/negative').mkdir(parents=True, exist_ok=True)

    # For backwards compatibility with existing code
    Path('imgs/scatter_by_wells').mkdir(parents=True, exist_ok=True)
    Path('imgs/scatter_by_wells/summary').mkdir(parents=True, exist_ok=True)
    for n_wells in [2, 3, 4]:
        Path(f'imgs/scatter_by_wells/{n_wells}_wells').mkdir(parents=True, exist_ok=True)

    # Collect all correlations for sorting
    all_positive_correlations = []
    all_negative_correlations = []

    # First pass: collect and sort all correlations
    for n_wells in [4, 3, 2]:  # Process in order of well count priority
        if n_wells not in correlations_by_well_count:
            continue

        correlations = correlations_by_well_count[n_wells]

        for pair, wells_data, info in correlations:
            log_var, lab_var = pair

            # Get data only from the wells in wells_data
            well_names = [w for w, _ in wells_data]
            mask = df_all['Well'].isin(well_names)
            combined_data = df_all[mask][[log_var, lab_var]].dropna()

            if len(combined_data) > 5:
                combined_r, _ = stats.pearsonr(combined_data[log_var], combined_data[lab_var])
                n_points = len(combined_data)
            else:
                combined_r = info['avg_corr']
                n_points = 0

            correlation_data = (pair, wells_data, info, combined_r, n_points, n_wells)

            if combined_r > 0:
                all_positive_correlations.append(correlation_data)
            else:
                all_negative_correlations.append(correlation_data)

    # Sort by absolute correlation value
    all_positive_correlations.sort(key=lambda x: abs(x[3]), reverse=True)
    all_negative_correlations.sort(key=lambda x: abs(x[3]), reverse=True)

    # Function to sanitize filenames
    def sanitize_filename(name):
        """Remove or replace characters that are problematic for filenames"""
        # Replace forward slash with underscore
        name = name.replace('/', '_')
        # Replace backslash with underscore
        name = name.replace('\\', '_')
        # Replace colon with underscore
        name = name.replace(':', '_')
        # Replace spaces with underscore
        name = name.replace(' ', '_')
        # Remove any other special characters
        name = re.sub(r'[<>"|?*]', '', name)
        return name

    # Function to create individual scatter plot
    def create_individual_scatter_plot(pair, wells_data, info, combined_r, n_points, n_wells, plot_idx, is_positive):
        """Create and save individual scatter plot"""
        log_var, lab_var = pair

        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot data from each well
        all_x = []
        all_y = []

        for well, well_r in wells_data:
            well_data = df_all[df_all['Well'] == well]
            mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
            x_data = well_data.loc[mask, log_var].values
            y_data = well_data.loc[mask, lab_var].values

            if len(x_data) > 0:
                well_short = well.replace('HRDH_', '')
                ax.scatter(x_data, y_data,
                         color=well_colors.get(well, '#666666'),
                         alpha=0.6, s=40,
                         label=f'{well_short} (r={well_r:.2f})',
                         edgecolors='darkgreen' if is_positive else 'darkred',
                         linewidth=0.5)

                all_x.extend(x_data)
                all_y.extend(y_data)

        # Add regression line with confidence interval
        if len(all_x) > 3:
            all_x = np.array(all_x)
            all_y = np.array(all_y)

            # Regression line
            z = np.polyfit(all_x, all_y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(all_x.min(), all_x.max(), 100)
            y_line = p(x_line)

            ax.plot(x_line, y_line,
                   color='darkgreen' if is_positive else 'darkred',
                   linestyle='-', alpha=0.8, linewidth=2,
                   label=f'Combined r={combined_r:.3f}')

            # Add confidence interval
            sns.regplot(x=all_x, y=all_y, ax=ax,
                       scatter=False,
                       color='darkgreen' if is_positive else 'darkred',
                       line_kws={'linewidth': 0},
                       ci=95)

        # Styling
        ax.set_facecolor('#e8f5e9' if is_positive else '#ffebee')
        ax.set_xlabel(log_var.replace('Log_', ''), fontsize=12, fontweight='bold')
        ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=12, fontweight='bold')

        # Title
        wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
        ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n" +
                    f"Wells: {wells_str} | n={n_points} | {n_wells}-well correlation",
                    fontsize=14, fontweight='bold')

        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)

        # Save individual plot
        sign = "positive" if is_positive else "negative"

        # Sanitize variable names for filename
        log_var_clean = sanitize_filename(log_var.replace('Log_', ''))
        lab_var_clean = sanitize_filename(lab_var.replace('Lab_', ''))

        filename = f"{plot_idx:03d}_{log_var_clean}_vs_{lab_var_clean}_r{abs(combined_r):.2f}.png"
        filepath = f"imgs/scatter_plots/{sign}/{filename}"

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

    # Create individual plots for positive correlations
    print("\nCreating individual scatter plots for positive correlations...")
    positive_filenames = []
    for idx, (pair, wells_data, info, combined_r, n_points, n_wells) in enumerate(all_positive_correlations):
        filename = create_individual_scatter_plot(pair, wells_data, info, combined_r, n_points, n_wells, idx+1, True)
        positive_filenames.append(filename)
        if idx % 10 == 0:
            print(f"  Created {idx+1}/{len(all_positive_correlations)} positive plots...")

    # Create individual plots for negative correlations
    print("\nCreating individual scatter plots for negative correlations...")
    negative_filenames = []
    for idx, (pair, wells_data, info, combined_r, n_points, n_wells) in enumerate(all_negative_correlations):
        filename = create_individual_scatter_plot(pair, wells_data, info, combined_r, n_points, n_wells, idx+1, False)
        negative_filenames.append(filename)
        if idx % 10 == 0:
            print(f"  Created {idx+1}/{len(all_negative_correlations)} negative plots...")

    # Create comprehensive positive correlations summary
    # Create comprehensive positive correlations summary
    if all_positive_correlations:
        print("\nCreating comprehensive positive correlations summary...")
        n_plots = min(len(all_positive_correlations), max_plots_per_figure)
        n_cols = 5
        n_rows = (n_plots + n_cols - 1) // n_cols
    
        fig = plt.figure(figsize=(25, 5*n_rows))
        fig.suptitle(f'All Positive Correlations Summary (Top {n_plots} of {len(all_positive_correlations)} total)\n' +
                    'Sorted by Correlation Strength', fontsize=20, fontweight='bold', color='darkgreen')
    
        for idx, (pair, wells_data, info, combined_r, n_points, n_wells) in enumerate(all_positive_correlations[:n_plots]):
            ax = plt.subplot(n_rows, n_cols, idx+1)
            log_var, lab_var = pair
    
            # Plot data with well colors AND labels (matching individual plots)
            all_x = []
            all_y = []
    
            for well, well_r in wells_data:
                well_data = df_all[df_all['Well'] == well]
                mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                x_data = well_data.loc[mask, log_var].values
                y_data = well_data.loc[mask, lab_var].values
    
                if len(x_data) > 0:
                    well_short = well.replace('HRDH_', '')
                    ax.scatter(x_data, y_data,
                             color=well_colors.get(well, '#666666'),
                             alpha=0.6, s=20,
                             label=f'{well_short} (r={well_r:.2f})',  # Added correlation value
                             edgecolors='darkgreen',
                             linewidth=0.3)
                    all_x.extend(x_data)
                    all_y.extend(y_data)
    
            # Add regression line with confidence interval (matching individual plots)
            if len(all_x) > 3:
                all_x = np.array(all_x)
                all_y = np.array(all_y)
                
                # Regression line
                z = np.polyfit(all_x, all_y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(all_x.min(), all_x.max(), 100)
                y_line = p(x_line)
                
                ax.plot(x_line, y_line, 
                       color='darkgreen', 
                       linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Combined r={combined_r:.3f}')
                
                # Add confidence interval
                sns.regplot(x=all_x, y=all_y, ax=ax,
                           scatter=False,
                           color='darkgreen',
                           line_kws={'linewidth': 0},
                           ci=95)
    
            # Styling (matching individual plots)
            ax.set_facecolor('#e8f5e9')
            wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
            ax.set_title(f"#{idx+1}: {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n" +
                        f"Wells: {wells_str} | n={n_points} | {n_wells}-well correlation", 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel(log_var.replace('Log_', ''), fontsize=8, fontweight='bold')
            ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=8, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend to each subplot (but make it smaller)
            ax.legend(fontsize=6, loc='best')
    
        plt.tight_layout()
        plt.savefig('imgs/positive_correlations_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Create comprehensive negative correlations summary
    # Create comprehensive negative correlations summary
    if all_negative_correlations:
        print("\nCreating comprehensive negative correlations summary...")
        n_plots = min(len(all_negative_correlations), max_plots_per_figure)
        n_cols = 5
        n_rows = (n_plots + n_cols - 1) // n_cols
    
        fig = plt.figure(figsize=(25, 5*n_rows))
        fig.suptitle(f'All Negative Correlations Summary (Top {n_plots} of {len(all_negative_correlations)} total)\n' +
                    'Sorted by Correlation Strength', fontsize=20, fontweight='bold', color='darkred')
    
        for idx, (pair, wells_data, info, combined_r, n_points, n_wells) in enumerate(all_negative_correlations[:n_plots]):
            ax = plt.subplot(n_rows, n_cols, idx+1)
            log_var, lab_var = pair
    
            # Plot data with well colors AND labels (matching individual plots)
            all_x = []
            all_y = []
    
            for well, well_r in wells_data:
                well_data = df_all[df_all['Well'] == well]
                mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                x_data = well_data.loc[mask, log_var].values
                y_data = well_data.loc[mask, lab_var].values
    
                if len(x_data) > 0:
                    well_short = well.replace('HRDH_', '')
                    ax.scatter(x_data, y_data,
                             color=well_colors.get(well, '#666666'),
                             alpha=0.6, s=20,
                             label=f'{well_short} (r={well_r:.2f})',  # Added correlation value
                             edgecolors='darkred',
                             linewidth=0.3)
                    all_x.extend(x_data)
                    all_y.extend(y_data)
    
            # Add regression line with confidence interval (matching individual plots)
            if len(all_x) > 3:
                all_x = np.array(all_x)
                all_y = np.array(all_y)
                
                # Regression line
                z = np.polyfit(all_x, all_y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(all_x.min(), all_x.max(), 100)
                y_line = p(x_line)
                
                ax.plot(x_line, y_line, 
                       color='darkred', 
                       linestyle='-', alpha=0.8, linewidth=2,
                       label=f'Combined r={combined_r:.3f}')
                
                # Add confidence interval
                sns.regplot(x=all_x, y=all_y, ax=ax,
                           scatter=False,
                           color='darkred',
                           line_kws={'linewidth': 0},
                           ci=95)
    
            # Styling (matching individual plots)
            ax.set_facecolor('#ffebee')
            wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
            ax.set_title(f"#{idx+1}: {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n" +
                        f"Wells: {wells_str} | n={n_points} | {n_wells}-well correlation", 
                        fontsize=9, fontweight='bold')
            ax.set_xlabel(log_var.replace('Log_', ''), fontsize=8, fontweight='bold')
            ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=8, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add legend to each subplot (but make it smaller)
            ax.legend(fontsize=6, loc='best')
    
        plt.tight_layout()
        plt.savefig('imgs/negative_correlations_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Continue with existing grouped plots by well count (for backwards compatibility)
    for n_wells in [2, 3, 4]:
        if n_wells not in correlations_by_well_count or not correlations_by_well_count[n_wells]:
            continue

        print(f"\nCreating grouped scatter plots for {n_wells}-well correlations...")

        correlations = correlations_by_well_count[n_wells]

        # Calculate combined correlations for sorting
        correlations_with_combined = []
        for pair, wells_data, info in correlations:
            log_var, lab_var = pair

            # Get data only from the wells in wells_data
            well_names = [w for w, _ in wells_data]
            mask = df_all['Well'].isin(well_names)
            combined_data = df_all[mask][[log_var, lab_var]].dropna()

            if len(combined_data) > 5:
                combined_r, _ = stats.pearsonr(combined_data[log_var], combined_data[lab_var])
                n_points = len(combined_data)
            else:
                combined_r = info['avg_corr']
                n_points = 0

            correlations_with_combined.append((pair, wells_data, info, combined_r, n_points, n_wells))

        # Sort by absolute combined correlation
        correlations_with_combined.sort(key=lambda x: abs(x[3]), reverse=True)

        # Separate positive and negative
        positive_corrs = [(pair, wells_data, info, combined_r, n_points)
                         for pair, wells_data, info, combined_r, n_points, _ in correlations_with_combined
                         if combined_r > 0]
        negative_corrs = [(pair, wells_data, info, combined_r, n_points)
                         for pair, wells_data, info, combined_r, n_points, _ in correlations_with_combined
                         if combined_r < 0]

        # Create grouped plots for positive correlations
        if positive_corrs:
            n_plots = min(len(positive_corrs), max_plots_per_figure)
            n_cols = min(4, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

            fig.suptitle(f'{n_wells}-Well Positive Correlations (Top {n_plots} of {len(positive_corrs)} total)\nSorted by Combined Correlation Strength',
                        fontsize=16, fontweight='bold', color='darkgreen')

            for idx, (pair, wells_data, info, combined_r, n_points) in enumerate(positive_corrs[:n_plots]):
                ax = axes[idx]
                log_var, lab_var = pair

                # Plot ONLY the wells that have this correlation
                all_x = []
                all_y = []

                for well, well_r in wells_data:
                    well_data = df_all[df_all['Well'] == well]
                    mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                    x_data = well_data.loc[mask, log_var].values
                    y_data = well_data.loc[mask, lab_var].values

                    if len(x_data) > 0:
                        well_short = well.replace('HRDH_', '')
                        ax.scatter(x_data, y_data,
                                 color=well_colors[well],
                                 alpha=0.6, s=30,
                                 label=f'{well_short} (r={well_r:.2f})',
                                 edgecolors='darkgreen',
                                 linewidth=0.5)

                        all_x.extend(x_data)
                        all_y.extend(y_data)

                # Add regression line with confidence interval
                if len(all_x) > 3:
                    all_x = np.array(all_x)
                    all_y = np.array(all_y)

                    # Regression line
                    z = np.polyfit(all_x, all_y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(all_x.min(), all_x.max(), 100)
                    y_line = p(x_line)

                    ax.plot(x_line, y_line,
                           color='darkgreen',
                           linestyle='-', alpha=0.8, linewidth=2,
                           label=f'Combined r={combined_r:.2f}')

                    # Add confidence interval using seaborn
                    sns.regplot(x=all_x, y=all_y, ax=ax,
                               scatter=False, color='darkgreen',
                               line_kws={'linewidth': 0},
                               ci=95)

                # Styling
                ax.set_facecolor('#e8f5e9')
                ax.set_xlabel(log_var.replace('Log_', ''), fontsize=10)
                ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=10)

                # Title with ONLY the wells that have this correlation
                wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
                ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str}, n={n_points})",
                           fontsize=9)

                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for idx in range(n_plots, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(f'imgs/scatter_by_wells/{n_wells}_wells/positive_{n_wells}_well_correlations.png',
                       dpi=300, bbox_inches='tight')
            plt.show()

        # Create grouped plots for negative correlations
        if negative_corrs:
            n_plots = min(len(negative_corrs), max_plots_per_figure)
            n_cols = min(4, n_plots)
            n_rows = (n_plots + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

            fig.suptitle(f'{n_wells}-Well Negative Correlations (Top {n_plots} of {len(negative_corrs)} total)\nSorted by Combined Correlation Strength',
                        fontsize=16, fontweight='bold', color='darkred')

            for idx, (pair, wells_data, info, combined_r, n_points) in enumerate(negative_corrs[:n_plots]):
                ax = axes[idx]
                log_var, lab_var = pair

                # Plot ONLY the wells that have this correlation
                all_x = []
                all_y = []

                for well, well_r in wells_data:
                    well_data = df_all[df_all['Well'] == well]
                    mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                    x_data = well_data.loc[mask, log_var].values
                    y_data = well_data.loc[mask, lab_var].values

                    if len(x_data) > 0:
                        well_short = well.replace('HRDH_', '')
                        ax.scatter(x_data, y_data,
                                 color=well_colors[well],
                                 alpha=0.6, s=30,
                                 label=f'{well_short} (r={well_r:.2f})',
                                 edgecolors='darkred',
                                 linewidth=0.5)

                        all_x.extend(x_data)
                        all_y.extend(y_data)

                # Add regression line with confidence interval
                if len(all_x) > 3:
                    all_x = np.array(all_x)
                    all_y = np.array(all_y)

                    # Regression line
                    z = np.polyfit(all_x, all_y, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(all_x.min(), all_x.max(), 100)
                    y_line = p(x_line)

                    ax.plot(x_line, y_line,
                           color='darkred',
                           linestyle='-', alpha=0.8, linewidth=2,
                           label=f'Combined r={combined_r:.2f}')

                    # Add confidence interval using seaborn
                    sns.regplot(x=all_x, y=all_y, ax=ax,
                               scatter=False, color='darkred',
                               line_kws={'linewidth': 0},
                               ci=95)

                # Styling
                ax.set_facecolor('#ffebee')
                ax.set_xlabel(log_var.replace('Log_', ''), fontsize=10)
                ax.set_ylabel(lab_var.replace('Lab_', ''), fontsize=10)

                # Title with ONLY the wells that have this correlation
                wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
                ax.set_title(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str}, n={n_points})",
                           fontsize=9)

                ax.legend(fontsize=7, loc='best')
                ax.grid(True, alpha=0.3)

            # Hide empty subplots
            for idx in range(n_plots, len(axes)):
                axes[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(f'imgs/scatter_by_wells/{n_wells}_wells/negative_{n_wells}_well_correlations.png',
                       dpi=300, bbox_inches='tight')
            plt.show()

    # Create summary bar chart
    if all_positive_correlations or all_negative_correlations:
        all_correlations_for_summary = all_positive_correlations + all_negative_correlations
        all_correlations_for_summary.sort(key=lambda x: abs(x[3]), reverse=True)

        # Create summary figure showing top correlations across all well counts
        n_top = min(30, len(all_correlations_for_summary))  # Show top 30

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

        # Prepare data for plotting
        labels = []
        colors = []
        values = []
        well_counts = []

        for pair, wells_data, info, combined_r, n_points, n_wells in all_correlations_for_summary[:n_top]:
            log_var, lab_var = pair
            wells_str = ', '.join([w.replace('HRDH_', '') for w, _ in wells_data])
            label = f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}\n({wells_str})"
            labels.append(label)
            values.append(combined_r)
            well_counts.append(n_wells)

            # Color based on correlation direction and well count
            if combined_r > 0:
                if n_wells == 4:
                    colors.append('#006400')  # Dark green for 4-well positive
                elif n_wells == 3:
                    colors.append('#228B22')  # Forest green for 3-well positive
                else:
                    colors.append('#90EE90')  # Light green for 2-well positive
            else:
                if n_wells == 4:
                    colors.append('#8B0000')  # Dark red for 4-well negative
                elif n_wells == 3:
                    colors.append('#DC143C')  # Crimson for 3-well negative
                else:
                    colors.append('#FFA07A')  # Light salmon for 2-well negative

        # Plot 1: Horizontal bar chart of all top correlations
        y_pos = np.arange(len(labels))
        bars = ax1.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, values)):
            x_pos = value + 0.01 if value > 0 else value - 0.01
            ax1.text(x_pos, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}',
                    ha='left' if value > 0 else 'right',
                    va='center', fontsize=8)

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel('Combined Correlation (r)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Top {n_top} Correlations Across All Wells\nSorted by Absolute Correlation Strength',
                     fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.set_xlim(-1.1, 1.1)

        # Add legend for well counts
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#006400', label='4-well positive'),
            Patch(facecolor='#228B22', label='3-well positive'),
            Patch(facecolor='#90EE90', label='2-well positive'),
            Patch(facecolor='#8B0000', label='4-well negative'),
            Patch(facecolor='#DC143C', label='3-well negative'),
            Patch(facecolor='#FFA07A', label='2-well negative')
        ]
        ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

        # Plot 2: Summary statistics by well count
        well_count_stats = {2: {'pos': 0, 'neg': 0, 'total': 0},
                           3: {'pos': 0, 'neg': 0, 'total': 0},
                           4: {'pos': 0, 'neg': 0, 'total': 0}}

        for _, _, _, combined_r, _, n_wells in all_correlations_for_summary:
            if n_wells in well_count_stats:
                well_count_stats[n_wells]['total'] += 1
                if combined_r > 0:
                    well_count_stats[n_wells]['pos'] += 1
                else:
                    well_count_stats[n_wells]['neg'] += 1

        # Create grouped bar chart
        x = np.arange(3)
        width = 0.25

        pos_counts = [well_count_stats[i]['pos'] for i in [2, 3, 4]]
        neg_counts = [well_count_stats[i]['neg'] for i in [2, 3, 4]]
        total_counts = [well_count_stats[i]['total'] for i in [2, 3, 4]]

        bars1 = ax2.bar(x - width, pos_counts, width, label='Positive', color='green', alpha=0.8)
        bars2 = ax2.bar(x, neg_counts, width, label='Negative', color='red', alpha=0.8)
        bars3 = ax2.bar(x + width, total_counts, width, label='Total', color='blue', alpha=0.6)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}',
                        ha='center', va='bottom', fontsize=10)

        ax2.set_xlabel('Number of Wells', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Correlations', fontsize=12, fontweight='bold')
        ax2.set_title('Correlation Count by Number of Wells', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['2 wells', '3 wells', '4 wells'])
        ax2.legend()
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('imgs/scatter_by_wells/summary/all_correlations_summary.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Print summary
    print("\n" + "="*80)
    print("SCATTER PLOT GENERATION SUMMARY")
    print("="*80)

    print(f"\nIndividual scatter plots created:")
    print(f"  Positive correlations: {len(all_positive_correlations)} plots")
    print(f"  Negative correlations: {len(all_negative_correlations)} plots")
    print(f"  Total individual plots: {len(all_positive_correlations) + len(all_negative_correlations)}")

    print(f"\nSummary visualizations created:")
    print(f"  - imgs/positive_correlations_summary.png (top {min(max_plots_per_figure, len(all_positive_correlations))} positive)")
    print(f"  - imgs/negative_correlations_summary.png (top {min(max_plots_per_figure, len(all_negative_correlations))} negative)")

    print(f"\nIndividual plots saved to:")
    print(f"  - imgs/scatter_plots/positive/ ({len(positive_filenames)} files)")
    print(f"  - imgs/scatter_plots/negative/ ({len(negative_filenames)} files)")

    total_positive = len(all_positive_correlations)
    total_negative = len(all_negative_correlations)

    for n_wells in [2, 3, 4]:
        if n_wells in correlations_by_well_count:
            corrs = correlations_by_well_count[n_wells]
            pos_count = sum(1 for _, _, info in corrs if info['avg_corr'] > 0)
            neg_count = sum(1 for _, _, info in corrs if info['avg_corr'] < 0)
            print(f"\n{n_wells}-well correlations:")
            print(f"  Positive: {pos_count}")
            print(f"  Negative: {neg_count}")
            print(f"  Total: {len(corrs)}")

    print(f"\nOverall totals:")
    print(f"  Total positive correlations: {total_positive}")
    print(f"  Total negative correlations: {total_negative}")
    print(f"  Grand total: {total_positive + total_negative}")


# pair plot


def create_advanced_well_pairplots_by_group(df_all, variable_groups, common_correlations, 
                                           min_samples_per_well=5, sample_size=1000):
    """
    Create advanced pairplots for each variable group, integrated with common correlations.
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined dataframe with all wells' data
    variable_groups : dict
        Dictionary of variable groups, each with 'logs' and 'labs' lists
    common_correlations : DataFrame
        DataFrame from find_common_correlations containing analyzed correlations
    min_samples_per_well : int
        Minimum samples required for a well to be included in a correlation
    sample_size : int
        Maximum number of samples to use (for performance)
    """
    print("\nCreating advanced multi-well pairplots by variable group...")
    
    # First, populate the Top_Correlations group with actual top correlations
    if "Top_Correlations" in variable_groups and common_correlations is not None and not common_correlations.empty:
        # Get top 10 correlations by average absolute correlation
        top_corrs = common_correlations.nlargest(10, 'avg_abs_r')
        
        # Extract unique variables
        top_logs = list(set(top_corrs['log_var'].tolist()))
        top_labs = list(set(top_corrs['lab_var'].tolist()))
        
        # Update the group
        variable_groups["Top_Correlations"]["logs"] = top_logs[:5]  # Limit to 5 for readability
        variable_groups["Top_Correlations"]["labs"] = top_labs[:5]
        
        print(f"Populated Top_Correlations with {len(top_logs)} log vars and {len(top_labs)} lab vars")
    
    for group_name, group_info in variable_groups.items():
        print(f"\nProcessing group: {group_name}")
        
        # Skip if empty
        if not group_info['logs'] and not group_info['labs']:
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
        
        # Create the advanced pairplot with common correlations integration
        try:
            fig = create_advanced_well_pairplot_integrated(
                df_all, 
                available_vars,
                common_correlations,
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
            import traceback
            traceback.print_exc()


def create_advanced_well_pairplot_integrated(df_all, variables_to_plot, common_correlations,
                                           min_samples_per_well=5, sample_size=1000, 
                                           save_path='imgs/', filename=None):
    """
    Create an advanced pairplot that integrates with common_correlations analysis.
    Shows the actual correlation values from the multi-well analysis.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Create lookup for common correlations
    corr_lookup = {}
    if common_correlations is not None and not common_correlations.empty:
        for _, row in common_correlations.iterrows():
            key = (row['log_var'], row['lab_var'])
            corr_lookup[key] = {
                'avg_r': row['avg_r'],
                'num_wells': row['num_wells'],
                'wells': row['wells_found_in'],
                'consistent': row['consistent_direction']
            }
    
    # Create variable friendly names
    var_friendly_names = {var: var.replace('Log_', '').replace('Lab_', '') for var in variables_to_plot}
    
    # Sample data if needed
    wells = df_all['Well'].unique()
    if len(df_all) > sample_size:
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
    
    # Define well colors
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#d62728', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#ff7f0e'
    }
    
    # Create the grid
    n_vars = len(variables_to_plot)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(2.5*n_vars, 2.5*n_vars))
    
    if n_vars == 1:
        axes = np.array([[axes]])
    
    # Create histograms on diagonal
    for i, var in enumerate(variables_to_plot):
        ax = axes[i, i]
        for well in wells:
            well_data = df_plot[df_plot['Well'] == well]
            if len(well_data) > 0:
                sns.histplot(well_data[var].dropna(), 
                           ax=ax, 
                           color=well_colors[well], 
                           alpha=0.5,
                           label=well.replace('HRDH_', ''))
        
        ax.set_xlabel(var_friendly_names[var])
        ax.set_ylabel('Count')
        
        if i == 0:
            ax.legend(fontsize=8)
        
        ax.set_yticks([])
    
    # Create plots for off-diagonal
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                var1 = variables_to_plot[j]  # x-axis
                var2 = variables_to_plot[i]  # y-axis
                ax = axes[i, j]
                
                if i < j:  # Upper triangle - scatter plots
                    # Check if this correlation exists in common_correlations
                    corr_info = corr_lookup.get((var1, var2)) or corr_lookup.get((var2, var1))
                    
                    if corr_info:
                        # Only plot wells that have this correlation
                        wells_to_plot = corr_info['wells']
                        title_suffix = f" ({corr_info['num_wells']}w)"
                    else:
                        wells_to_plot = wells
                        title_suffix = ""
                    
                    # Plot each well
                    for well in wells_to_plot:
                        well_data = df_plot[df_plot['Well'] == well]
                        valid_data = well_data[[var1, var2]].dropna()
                        
                        if len(valid_data) > 0:
                            ax.scatter(valid_data[var1], valid_data[var2], 
                                     color=well_colors[well], 
                                     alpha=0.7, s=20, 
                                     label=well.replace('HRDH_', ''))
                    
                    # Add combined regression line if correlation exists
                    if corr_info:
                        # Get all data from wells with this correlation
                        mask = df_plot['Well'].isin(wells_to_plot)
                        combined_data = df_plot[mask][[var1, var2]].dropna()
                        
                        if len(combined_data) > 5:
                            x = combined_data[var1]
                            y = combined_data[var2]
                            
                            # Fit line
                            z = np.polyfit(x, y, 1)
                            p = np.poly1d(z)
                            
                            # Plot line
                            x_line = np.linspace(x.min(), x.max(), 100)
                            color = 'darkgreen' if corr_info['avg_r'] > 0 else 'darkred'
                            ax.plot(x_line, p(x_line), color=color, 
                                   linestyle='-', alpha=0.8, linewidth=2)
                    
                    ax.set_xlabel(var_friendly_names[var1])
                    ax.set_ylabel(var_friendly_names[var2])
                    ax.grid(True, linestyle=':', alpha=0.3)
                
                else:  # Lower triangle - correlation info
                    ax.clear()
                    
                    # Get correlation from common_correlations
                    corr_info = corr_lookup.get((var1, var2)) or corr_lookup.get((var2, var1))
                    
                    if corr_info:
                        r = corr_info['avg_r']
                        n_wells = corr_info['num_wells']
                        
                        # Color based on correlation
                        if r > 0:
                            color = plt.cm.Greens(0.5 + r/2)
                        else:
                            color = plt.cm.Reds(0.5 - r/2)
                        
                        # Fill with color
                        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                                 color=color, alpha=0.7))
                        
                        # Add text
                        consistency = "✓" if corr_info['consistent'] else "✗"
                        ax.text(0.5, 0.5, f'r = {r:.2f}', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=11, fontweight='bold',
                               color='black' if abs(r) < 0.7 else 'white')
                        
                        ax.text(0.5, 0.25, f'{n_wells} wells {consistency}', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=9,
                               color='black' if abs(r) < 0.7 else 'white')
                    else:
                        # No common correlation found
                        ax.text(0.5, 0.5, 'No common\ncorrelation', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=9, color='gray')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)
    
    # Add legend
    plt.figtext(0.5, 0.01, 
               "✓ = consistent direction across wells, ✗ = inconsistent direction", 
               ha='center', fontsize=10)
    
    # Title
    fig.suptitle("Multi-Well Correlation Analysis (Integrated with Common Correlations)", 
                fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)
    
    # Save
    if filename is None:
        var_names = [var_friendly_names[v] for v in variables_to_plot]
        filename = f"advanced_pairplot_integrated_{'_'.join(var_names[:3])}"
        if len(var_names) > 3:
            filename += "_etc"
    
    plt.savefig(f"{save_path}/{filename}.png", dpi=300, bbox_inches='tight')
    
    return fig

def create_advanced_well_pairplot_integrated_v2(df_all, variables_to_plot, common_correlations,
                                              min_samples_per_well=5, sample_size=1000, 
                                              save_path='imgs/', filename=None):
    """
    Enhanced version that shows all correlations, not just those in common_correlations.
    Calculates correlations on the fly for pairs not in common_correlations.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    
    # Create lookup for common correlations
    corr_lookup = {}
    if common_correlations is not None and not common_correlations.empty:
        for _, row in common_correlations.iterrows():
            key = (row['log_var'], row['lab_var'])
            corr_lookup[key] = {
                'avg_r': row['avg_r'],
                'num_wells': row['num_wells'],
                'wells': row['wells_found_in'],
                'consistent': row['consistent_direction']
            }
    
    # Create variable friendly names
    var_friendly_names = {var: var.replace('Log_', '').replace('Lab_', '') for var in variables_to_plot}
    
    # Sample data if needed
    wells = df_all['Well'].unique()
    if len(df_all) > sample_size:
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
    
    # Define well colors
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#d62728', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#ff7f0e'
    }
    
    # Create the grid
    n_vars = len(variables_to_plot)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(2.5*n_vars, 2.5*n_vars))
    
    if n_vars == 1:
        axes = np.array([[axes]])
    elif n_vars == 2:
        axes = np.array(axes).reshape(n_vars, n_vars)
    
    # Create plots
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            var1 = variables_to_plot[j]  # x-axis
            var2 = variables_to_plot[i]  # y-axis
            
            if i == j:  # Diagonal - histograms
                for well in wells:
                    well_data = df_plot[df_plot['Well'] == well]
                    if len(well_data) > 0:
                        well_vals = well_data[var1].dropna()
                        if len(well_vals) > 0:
                            sns.histplot(well_vals, 
                                       ax=ax, 
                                       color=well_colors[well], 
                                       alpha=0.5,
                                       label=well.replace('HRDH_', ''),
                                       bins=20)
                
                ax.set_xlabel(var_friendly_names[var1], fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                
                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc='upper right')
                
                ax.set_yticks([])
    
            elif i < j:  # Upper triangle - scatter plots
                # Plot all wells
                for well in wells:
                    well_data = df_plot[df_plot['Well'] == well]
                    valid_data = well_data[[var1, var2]].dropna()
                    
                    if len(valid_data) > 0:
                        ax.scatter(valid_data[var1], valid_data[var2], 
                                 color=well_colors[well], 
                                 alpha=0.7, s=20, 
                                 label=well.replace('HRDH_', ''))
                
                # Add regression line for all data
                all_data = df_plot[[var1, var2]].dropna()
                if len(all_data) > 5:
                    x = all_data[var1]
                    y = all_data[var2]
                    
                    # Calculate correlation
                    r, _ = stats.pearsonr(x, y)
                    
                    # Fit line
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    
                    # Plot line
                    x_line = np.linspace(x.min(), x.max(), 100)
                    color = 'darkgreen' if r > 0 else 'darkred'
                    ax.plot(x_line, p(x_line), color=color, 
                           linestyle='-', alpha=0.8, linewidth=2)
                
                # Set labels for scatter plots
                if i == 0:  # Top row
                    ax.set_title(var_friendly_names[var1], fontsize=10)
                if j == 0:  # Left column
                    ax.set_ylabel(var_friendly_names[var2], fontsize=10)
                if i == n_vars - 1:  # Bottom row of scatter plots
                    ax.set_xlabel(var_friendly_names[var1], fontsize=10)
                    
                ax.grid(True, linestyle=':', alpha=0.3)
                
            else:  # Lower triangle - correlation info
                ax.clear()
                
                # Get correlation from common_correlations
                corr_info = corr_lookup.get((var1, var2)) or corr_lookup.get((var2, var1))
                
                if corr_info:
                    # Use existing correlation info from common_correlations
                    r = corr_info['avg_r']
                    n_wells = corr_info['num_wells']
                    
                    # Color based on correlation
                    if r > 0:
                        color = plt.cm.Greens(0.5 + r/2)
                    else:
                        color = plt.cm.Reds(0.5 - r/2)
                    
                    # Fill with color
                    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                             color=color, alpha=0.7))
                    
                    # Add text
                    consistency = "✓" if corr_info['consistent'] else "✗"
                    ax.text(0.5, 0.5, f'r = {r:.2f}', 
                           transform=ax.transAxes,
                           ha='center', va='center', 
                           fontsize=11, fontweight='bold',
                           color='black' if abs(r) < 0.7 else 'white')
                    
                    ax.text(0.5, 0.25, f'{n_wells} wells {consistency}', 
                           transform=ax.transAxes,
                           ha='center', va='center', 
                           fontsize=9,
                           color='black' if abs(r) < 0.7 else 'white')
                else:
                    # Calculate correlation for all data since it's not in common_correlations
                    valid_data = df_plot[[var1, var2]].dropna()
                    
                    if len(valid_data) >= 10:
                        r, p_value = stats.pearsonr(valid_data[var1], valid_data[var2])
                        
                        # Color based on correlation strength
                        if abs(r) < 0.3:
                            color = 'lightgray'
                        elif r > 0:
                            color = plt.cm.Greens(0.3 + r/2)
                        else:
                            color = plt.cm.Reds(0.3 - r/2)
                        
                        # Fill with color
                        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, 
                                                 color=color, alpha=0.5))
                        
                        # Add text
                        ax.text(0.5, 0.5, f'r = {r:.2f}', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=11,
                               color='black' if abs(r) < 0.5 else 'white')
                        
                        # Check how many wells have sufficient data for this pair
                        wells_with_data = 0
                        for well in wells:
                            well_data = df_all[df_all['Well'] == well][[var1, var2]].dropna()
                            if len(well_data) >= min_samples_per_well:
                                wells_with_data += 1
                        
                        ax.text(0.5, 0.25, f'(all data)\nn={len(valid_data)}\n{wells_with_data} wells', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=8,
                               color='gray')
                    else:
                        # Not enough data
                        ax.text(0.5, 0.5, 'Insufficient\ndata', 
                               transform=ax.transAxes,
                               ha='center', va='center', 
                               fontsize=9, color='gray')
                
                # Set labels for correlation boxes
                if j == 0:  # Left column
                    ax.set_ylabel(var_friendly_names[var2], fontsize=10)
                if i == n_vars - 1:  # Bottom row
                    ax.set_xlabel(var_friendly_names[var1], fontsize=10)
                
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
    
    # Adjust spacing and add legend
    plt.tight_layout()
    
    # Add legend at the bottom with proper formatting
    legend_text = ("✓ = consistent direction across wells, ✗ = inconsistent direction\n" +
                   "Gray boxes show correlations calculated from all data (not in common_correlations)")
    fig.text(0.5, 0.01, legend_text, 
             ha='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
    
    # Title
    fig.suptitle("Multi-Well Correlation Analysis (All Correlations)", 
                fontsize=16, y=0.99)
    
    plt.subplots_adjust(top=0.96, bottom=0.08, hspace=0.15, wspace=0.15)
    
    # Save
    if filename is None:
        var_names = [var_friendly_names[v] for v in variables_to_plot]
        filename = f"advanced_pairplot_all_correlations_{'_'.join(var_names[:3])}"
        if len(var_names) > 3:
            filename += "_etc"
    
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    plt.savefig(save_path / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig