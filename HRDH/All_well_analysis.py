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
# All_well_analysis
def fix_duplicate_columns(df, well_name):
    """Fix duplicate columns by merging data from duplicates into the original column."""
    columns = df.columns.tolist()
    seen = {}
    duplicates = {}
    
    for i, col in enumerate(columns):
        if col in seen:
            if col not in duplicates:
                duplicates[col] = [seen[col]]
            duplicates[col].append(i)
        else:
            seen[col] = i
    
    if duplicates:
        print(f"\nFound duplicate columns in {well_name}:")
        for col, indices in duplicates.items():
            print(f"  - '{col}' appears at indices: {indices}")
        
        # Fix duplicates by merging data
        for col_name, indices in duplicates.items():
            dup_cols = [df.iloc[:, idx] for idx in indices]
            merged_data = dup_cols[0].copy()
            
            for dup_col in dup_cols[1:]:
                mask = merged_data.isna() & ~dup_col.isna()
                merged_data[mask] = dup_col[mask]
            
            df.iloc[:, indices[0]] = merged_data
            original_nulls = dup_cols[0].isna().sum()
            final_nulls = merged_data.isna().sum()
            recovered = original_nulls - final_nulls
            
            print(f"    Merged {len(indices)} columns named '{col_name}'")
            print(f"    Recovered {recovered} missing values")
        
        df = df.loc[:, ~df.columns.duplicated(keep='first')]
        print(f"  Removed duplicate columns")
    
    return df

def load_all_wells_csv_fixed():
    """Load all well CSV files with duplicate column fixing."""
    base_path = Path('.')
    all_data = []
    
    joined_files = list(base_path.glob('HRDH_*/HRDH_*_joined.csv'))
    
    if not joined_files:
        raise FileNotFoundError("No joined CSV files found")
    
    print(f"Found {len(joined_files)} joined CSV files:")
    for file in joined_files:
        print(f"  {file}")
    
    for file in joined_files:
        well_name = file.parent.name
        df = pd.read_csv(file)
        
        # Fix duplicate columns if present
        df = fix_duplicate_columns(df, well_name)
        
        # Add well identifier
        df['Well'] = well_name
        df.columns = df.columns.str.strip()
        
        print(f"Loaded {well_name}: {len(df)} samples, {len(df.columns)} columns")
        all_data.append(df)
    
    df_combined = pd.concat(all_data, ignore_index=True, sort=True)
    print(f"\nCombined dataset: {len(df_combined)} total samples from {len(all_data)} wells")
    print(f"Total columns after merge: {len(df_combined.columns)}")
    
    return df_combined

def load_all_wells_csv(base_path="."):
    

    """Load all joined CSV files and combine them into a single DataFrame"""
    
    # Find all joined CSV files
    csv_files = list(Path(base_path).glob('HRDH_*/HRDH_*_joined.csv'))
    
    print(f"Found {len(csv_files)} joined CSV files:")
    for file in csv_files:
        print(f"{file}")
    
    if len(csv_files) == 0:
        print("No joined CSV files found!")
        return pd.DataFrame()
    
    # Load each file and add well identifier
    all_wells = []
    
    for csv_file in csv_files:
        try:
            # Extract well name from folder name
            well_name = csv_file.parent.name
            
            # Load data
            df = pd.read_csv(csv_file)
            
            # Add well identifier as the first column for easy identification
            df.insert(0, 'Well', well_name)
            
            # Add to list
            all_wells.append(df)
            
            print(f"Loaded {well_name}: {df.shape[0]} samples, {df.shape[1]} columns")
            
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    if len(all_wells) == 0:
        print("No files were successfully loaded!")
        return pd.DataFrame()
    
    # Concatenate all dataframes
    df_all = pd.concat(all_wells, ignore_index=True)
    
    # Ensure Well column is properly typed
    df_all['Well'] = df_all['Well'].astype('category')
    
    print(f"\nCombined dataset: {df_all.shape[0]} total samples from {df_all['Well'].nunique()} wells")
    print(f"Wells: {', '.join(df_all['Well'].unique())}")
    print(f"Sample distribution by well:")
    print(df_all['Well'].value_counts().to_string())
    
    return df_all

    """Find ALL correlations and categorize them by how many wells they appear in."""
    correlation_tracker = {}
    all_wells = list(well_correlations.keys())
    total_wells = len(all_wells)
    
    # Find correlations in each well
    for well, corr_matrix in well_correlations.items():
        for log_var in corr_matrix.index:
            for lab_var in corr_matrix.columns:
                r = corr_matrix.loc[log_var, lab_var]
                
                # Check if correlation meets minimum threshold (using absolute value for threshold only)
                if pd.notna(r) and abs(r) >= min_correlation:
                    pair = (log_var, lab_var)
                    
                    if pair not in correlation_tracker:
                        correlation_tracker[pair] = []
                    
                    # Store actual r value (not absolute)
                    correlation_tracker[pair].append((well, r))
    
    # Categorize correlations by number of wells
    correlations_by_well_count = {2: [], 3: [], 4: []}
    
    for pair, wells_data in correlation_tracker.items():
        n_wells = len(wells_data)
        
        if n_wells >= 2:  # Only include correlations in 2+ wells
            # Calculate statistics using actual r values
            r_values = [r for _, r in wells_data]
            avg_r = np.mean(r_values)  # Average of actual r values
            avg_abs_r = np.mean([abs(r) for r in r_values])  # For sorting/display
            std_r = np.std(r_values)
            
            # Determine correlation type based on average r (not absolute)
            if avg_r > 0:
                correlation_type = "Positive"
            else:
                correlation_type = "Negative"
            
            # Check if all correlations have same sign
            all_positive = all(r > 0 for r in r_values)
            all_negative = all(r < 0 for r in r_values)
            consistent_direction = all_positive or all_negative
            
            # Find missing wells
            wells_present = [well for well, _ in wells_data]
            missing_wells = [well for well in all_wells if well not in wells_present]
            
            info = {
                'n_wells': n_wells,
                'avg_corr': avg_r,  # Changed from avg_abs_corr to avg_corr
                'avg_abs_corr': avg_abs_r,  # Keep for sorting
                'std_corr': std_r,
                'correlation_type': correlation_type,
                'consistent_direction': consistent_direction,
                'missing_wells': missing_wells
            }
            
            if n_wells in correlations_by_well_count:
                correlations_by_well_count[n_wells].append((pair, wells_data, info))
    
    # Sort each category by average absolute correlation strength (for display purposes)
    for n_wells in correlations_by_well_count:
        correlations_by_well_count[n_wells].sort(
            key=lambda x: x[2]['avg_abs_corr'], 
            reverse=True
        )
    
    return correlations_by_well_count

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

# finding correlations

def calculate_correlations_by_well(df_all, lab_columns, log_columns, min_samples=10):
    """
    Calculate correlation matrices between lab and log measurements for each well.
    Enhanced version with additional statistics and metadata.
    """
    well_correlations = {}
    well_correlation_stats = {}
    wells = sorted(df_all['Well'].unique())
    
    print("="*80)
    print("CALCULATING CORRELATIONS BY WELL")
    print("="*80)
    
    # First, let's diagnose data availability
    print("\nData Availability Check:")
    print("-" * 60)
    
    for well in wells:
        well_data = df_all[df_all['Well'] == well]
        print(f"\n{well}:")
        print(f"  Total samples: {len(well_data)}")
        
        # Check actual data availability for log and lab columns
        log_coverage = well_data[log_columns].notna().sum()
        lab_coverage = well_data[lab_columns].notna().sum()
        
        print(f"  Log data coverage: min={log_coverage.min()}, max={log_coverage.max()}, mean={log_coverage.mean():.1f}")
        print(f"  Lab data coverage: min={lab_coverage.min()}, max={lab_coverage.max()}, mean={lab_coverage.mean():.1f}")
        
        # Check how many valid pairs we can actually form
        valid_pair_count = 0
        for log_col in log_columns[:3]:  # Sample check
            for lab_col in lab_columns[:3]:
                valid_samples = well_data[[log_col, lab_col]].dropna()
                if len(valid_samples) >= min_samples:
                    valid_pair_count += 1
        
        estimated_valid_pairs = (valid_pair_count / 9) * (len(log_columns) * len(lab_columns))
        print(f"  Estimated valid pairs: {estimated_valid_pairs:.0f} ({estimated_valid_pairs/(len(log_columns)*len(lab_columns))*100:.1f}%)")
    
    print("\n" + "="*80)
    print("CORRELATION CALCULATIONS")
    print("="*80)
    
    for well in wells:
        well_data = df_all[df_all['Well'] == well]
        
        # Create correlation matrix and additional matrices for p-values and sample sizes
        corr_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=float)
        pvalue_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=float)
        sample_size_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=int)
        
        valid_pairs = 0
        total_pairs = len(log_columns) * len(lab_columns)
        
        for log_col in log_columns:
            for lab_col in lab_columns:
                # Get non-null data
                valid_data = well_data[[log_col, lab_col]].dropna()
                n_samples = len(valid_data)
                
                sample_size_matrix.loc[log_col, lab_col] = n_samples
                
                if n_samples >= min_samples:
                    try:
                        r, p = stats.pearsonr(valid_data[log_col], valid_data[lab_col])
                        corr_matrix.loc[log_col, lab_col] = r
                        pvalue_matrix.loc[log_col, lab_col] = p
                        valid_pairs += 1
                    except Exception as e:
                        # Handle any calculation errors
                        corr_matrix.loc[log_col, lab_col] = np.nan
                        pvalue_matrix.loc[log_col, lab_col] = np.nan
                else:
                    corr_matrix.loc[log_col, lab_col] = np.nan
                    pvalue_matrix.loc[log_col, lab_col] = np.nan
        
        well_correlations[well] = {
            'correlation': corr_matrix,
            'pvalue': pvalue_matrix,
            'sample_size': sample_size_matrix
        }
        
        # Calculate well-specific statistics
        valid_correlations = corr_matrix.values.flatten()
        valid_correlations = valid_correlations[~np.isnan(valid_correlations)]
        
        well_correlation_stats[well] = {
            'total_pairs': total_pairs,
            'valid_pairs': valid_pairs,
            'coverage': (valid_pairs / total_pairs) * 100,
            'mean_abs_correlation': np.mean(np.abs(valid_correlations)) if len(valid_correlations) > 0 else 0,
            'max_correlation': np.max(valid_correlations) if len(valid_correlations) > 0 else np.nan,
            'min_correlation': np.min(valid_correlations) if len(valid_correlations) > 0 else np.nan,
            'strong_correlations': np.sum(np.abs(valid_correlations) >= 0.7) if len(valid_correlations) > 0 else 0,
            'moderate_correlations': np.sum((np.abs(valid_correlations) >= 0.5) & (np.abs(valid_correlations) < 0.7)) if len(valid_correlations) > 0 else 0,
            'total_samples': len(well_data),
            'valid_correlations_count': len(valid_correlations)
        }
        
        # Print summary
        print(f"\n{well}:")
        print(f"  Total samples: {len(well_data):,}")
        print(f"  Valid correlations: {valid_pairs}/{total_pairs} ({well_correlation_stats[well]['coverage']:.1f}%)")
        if len(valid_correlations) > 0:
            print(f"  Mean |r|: {well_correlation_stats[well]['mean_abs_correlation']:.3f}")
            print(f"  Strong correlations (|r| ≥ 0.7): {well_correlation_stats[well]['strong_correlations']}")
            print(f"  Moderate correlations (0.5 ≤ |r| < 0.7): {well_correlation_stats[well]['moderate_correlations']}")
        else:
            print(f"  ⚠️ No valid correlations found (insufficient overlapping data)")
    
    # Add a warning summary
    print("\n" + "="*80)
    print("DATA QUALITY WARNINGS")
    print("="*80)
    
    wells_with_issues = [well for well, stats in well_correlation_stats.items() 
                        if stats['coverage'] < 10]
    
    if wells_with_issues:
        print(f"\n⚠️ Wells with very low coverage (<10%):")
        for well in wells_with_issues:
            print(f"  - {well}: {well_correlation_stats[well]['coverage']:.1f}% coverage, "
                  f"{well_correlation_stats[well]['valid_pairs']} valid pairs")
            
            # Diagnose why
            well_data = df_all[df_all['Well'] == well]
            sample_matrix = well_correlations[well]['sample_size']
            
            # Find which has more data - logs or labs
            log_data_counts = well_data[log_columns].notna().sum()
            lab_data_counts = well_data[lab_columns].notna().sum()
            
            if log_data_counts.mean() < 5:
                print(f"    → Issue: Very limited log data (avg {log_data_counts.mean():.1f} samples per log)")
            if lab_data_counts.mean() < 5:
                print(f"    → Issue: Very limited lab data (avg {lab_data_counts.mean():.1f} samples per lab)")
            
            # Check if it's a depth matching issue
            if len(well_data) < min_samples:
                print(f"    → Issue: Total samples ({len(well_data)}) below minimum required ({min_samples})")
    
    return well_correlations, well_correlation_stats

def find_all_correlations_by_well_count(well_correlations, min_correlation=0.5, significance_level=0.05):
    """
    Find ALL correlations and categorize them by how many wells they appear in.
    Enhanced version with comprehensive statistics for reporting.
    """
    correlation_tracker = {}
    all_wells = list(well_correlations.keys())
    total_wells = len(all_wells)
    
    # Track all correlation data for comprehensive analysis
    all_correlation_data = []
    
    # Find correlations in each well
    for well, matrices in well_correlations.items():
        corr_matrix = matrices['correlation']
        pvalue_matrix = matrices['pvalue']
        sample_size_matrix = matrices['sample_size']
        
        for log_var in corr_matrix.index:
            for lab_var in corr_matrix.columns:
                r = corr_matrix.loc[log_var, lab_var]
                p = pvalue_matrix.loc[log_var, lab_var]
                n = sample_size_matrix.loc[log_var, lab_var]
                
                # Store all correlations for comprehensive analysis
                if pd.notna(r) and n > 0:
                    all_correlation_data.append({
                        'well': well,
                        'log_var': log_var,
                        'lab_var': lab_var,
                        'correlation': r,
                        'pvalue': p,
                        'sample_size': n,
                        'significant': p < significance_level if pd.notna(p) else False
                    })
                
                # Check if correlation meets minimum threshold
                if pd.notna(r) and abs(r) >= min_correlation:
                    pair = (log_var, lab_var)
                    
                    if pair not in correlation_tracker:
                        correlation_tracker[pair] = []
                    
                    # Store comprehensive data
                    correlation_tracker[pair].append({
                        'well': well,
                        'correlation': r,
                        'pvalue': p,
                        'sample_size': n,
                        'significant': p < significance_level if pd.notna(p) else False
                    })
    
    # Categorize correlations by number of wells with enhanced statistics
    correlations_by_well_count = {1: [], 2: [], 3: [], 4: []}
    
    for pair, wells_data in correlation_tracker.items():
        n_wells = len(wells_data)
        
        if n_wells in correlations_by_well_count:
            # Extract values for statistics
            r_values = [d['correlation'] for d in wells_data]
            p_values = [d['pvalue'] for d in wells_data if pd.notna(d['pvalue'])]
            n_samples = [d['sample_size'] for d in wells_data]
            significant_count = sum(d['significant'] for d in wells_data)
            
            # Calculate comprehensive statistics
            avg_r = np.mean(r_values)
            avg_abs_r = np.mean([abs(r) for r in r_values])
            std_r = np.std(r_values)
            min_r = np.min(r_values)
            max_r = np.max(r_values)
            range_r = max_r - min_r
            
            # Determine correlation type and consistency
            if avg_r > 0:
                correlation_type = "Positive"
            else:
                correlation_type = "Negative"
            
            all_positive = all(r > 0 for r in r_values)
            all_negative = all(r < 0 for r in r_values)
            consistent_direction = all_positive or all_negative
            
            # Find missing wells
            wells_present = [d['well'] for d in wells_data]
            missing_wells = [well for well in all_wells if well not in wells_present]
            
            # Calculate confidence metrics
            avg_sample_size = np.mean(n_samples)
            total_samples = np.sum(n_samples)
            
            # Calculate correlation strength category
            if avg_abs_r >= 0.7:
                strength_category = "Strong"
            elif avg_abs_r >= 0.5:
                strength_category = "Moderate"
            elif avg_abs_r >= 0.3:
                strength_category = "Weak"
            else:
                strength_category = "Very Weak"
            
            info = {
                'n_wells': n_wells,
                'avg_corr': avg_r,
                'avg_abs_corr': avg_abs_r,
                'std_corr': std_r,
                'min_corr': min_r,
                'max_corr': max_r,
                'range_corr': range_r,
                'correlation_type': correlation_type,
                'consistent_direction': consistent_direction,
                'missing_wells': missing_wells,
                'avg_sample_size': avg_sample_size,
                'total_samples': total_samples,
                'min_sample_size': min(n_samples),
                'max_sample_size': max(n_samples),
                'significant_count': significant_count,
                'significant_ratio': significant_count / n_wells,
                'avg_pvalue': np.mean(p_values) if p_values else np.nan,
                'strength_category': strength_category,
                'wells_present': wells_present
            }
            
            # Convert wells_data to simpler format for backward compatibility
            simple_wells_data = [(d['well'], d['correlation']) for d in wells_data]
            
            correlations_by_well_count[n_wells].append((pair, simple_wells_data, info))
    
    # Sort each category by average absolute correlation strength
    for n_wells in correlations_by_well_count:
        correlations_by_well_count[n_wells].sort(
            key=lambda x: x[2]['avg_abs_corr'], 
            reverse=True
        )
    
    # Create comprehensive report data
    report_data = {
        'correlations_by_well_count': correlations_by_well_count,
        'all_correlation_data': pd.DataFrame(all_correlation_data),
        'summary_stats': {
            'total_unique_pairs': len(correlation_tracker),
            'total_correlations': len(all_correlation_data),
            'pairs_by_well_count': {n: len(corrs) for n, corrs in correlations_by_well_count.items()},
            'threshold_used': min_correlation,
            'significance_level': significance_level
        }
    }
    
    return correlations_by_well_count, report_data

def print_categorized_correlation_summary(correlations_by_well_count, report_data=None, top_n=10, export_path=None):
    """
    Print a detailed summary of correlations categorized by well count.
    Enhanced version with export capability and comprehensive statistics.
    """
    print("\n" + "="*100)
    print("COMPREHENSIVE CORRELATION ANALYSIS REPORT")
    print("="*100)
    
    # If report_data is a tuple (backward compatibility), extract the first element
    if isinstance(correlations_by_well_count, tuple):
        correlations_by_well_count = correlations_by_well_count[0]
    
    total_correlations = sum(len(corrs) for corrs in correlations_by_well_count.values())
    
    # Overall summary
    print(f"\n 📊 OVERALL SUMMARY")
    print("-"*80)
    print(f"Total unique correlation pairs found: {total_correlations}")
    
    if report_data and 'summary_stats' in report_data:
        print(f"Minimum correlation threshold: |r| ≥ {report_data['summary_stats']['threshold_used']}")
        print(f"Significance level: α = {report_data['summary_stats']['significance_level']}")
    
    # Add new summary section for well distribution
    print("\n📈 CORRELATION DISTRIBUTION BY NUMBER OF WELLS:")
    print("-"*80)
    print(f"{'Wells':<10} {'Count':<10} {'Percentage':<12} {'Cumulative %':<15} {'Avg |r|':<10} {'Strong':<10} {'Moderate':<10}")
    print("-"*80)
    
    cumulative_percentage = 0
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        correlations = correlations_by_well_count[n_wells]
        if correlations:
            count = len(correlations)
            percentage = (count / total_correlations) * 100
            cumulative_percentage += percentage
            
            # Calculate strength distribution
            strong_count = sum(1 for _, _, info in correlations if info['avg_abs_corr'] >= 0.7)
            moderate_count = sum(1 for _, _, info in correlations if 0.5 <= info['avg_abs_corr'] < 0.7)
            avg_abs_corr = np.mean([info['avg_abs_corr'] for _, _, info in correlations])
            
            print(f"{n_wells:<10} {count:<10} {percentage:<12.1f}% {cumulative_percentage:<15.1f}% "
                  f"{avg_abs_corr:<10.3f} {strong_count:<10} {moderate_count:<10}")
    
    # Add new section: Common correlations summary
    print("\n🔍 COMMON CORRELATIONS SUMMARY:")
    print("-"*80)
    
    # Correlations in all wells (if any)
    all_wells = list(next(iter(correlations_by_well_count.values()))[0][1][0][0].split('_')[0] + '_' 
                     for _ in range(max(correlations_by_well_count.keys())))
    
    if 4 in correlations_by_well_count and correlations_by_well_count[4]:
        print(f"Correlations found in ALL 4 wells: {len(correlations_by_well_count[4])}")
        print("  Examples:")
        for i, (pair, _, info) in enumerate(correlations_by_well_count[4][:3]):
            log_var, lab_var = pair
            print(f"    • {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: "
                  f"|r̄| = {info['avg_abs_corr']:.3f}")
    
    if 3 in correlations_by_well_count and correlations_by_well_count[3]:
        print(f"\nCorrelations found in 3 wells: {len(correlations_by_well_count[3])}")
        print("  Examples:")
        for i, (pair, _, info) in enumerate(correlations_by_well_count[3][:3]):
            log_var, lab_var = pair
            print(f"    • {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: "
                  f"|r̄| = {info['avg_abs_corr']:.3f}")
    
    if 2 in correlations_by_well_count and correlations_by_well_count[2]:
        print(f"\nCorrelations found in 2 wells: {len(correlations_by_well_count[2])}")
        print("  Top 5 by strength:")
        for i, (pair, _, info) in enumerate(correlations_by_well_count[2][:5]):
            log_var, lab_var = pair
            print(f"    • {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: "
                  f"|r̄| = {info['avg_abs_corr']:.3f}")
    
    # Calculate how many correlations appear in multiple wells
    multi_well_count = sum(len(corrs) for n_wells, corrs in correlations_by_well_count.items() if n_wells >= 2)
    multi_well_percentage = (multi_well_count / total_correlations * 100) if total_correlations > 0 else 0
    
    print(f"\n📌 MULTI-WELL CORRELATION STATISTICS:")
    print("-"*80)
    print(f"Correlations appearing in 2+ wells: {multi_well_count} ({multi_well_percentage:.1f}%)")
    print(f"Correlations appearing in 3+ wells: "
          f"{sum(len(corrs) for n_wells, corrs in correlations_by_well_count.items() if n_wells >= 3)} "
          f"({sum(len(corrs) for n_wells, corrs in correlations_by_well_count.items() if n_wells >= 3) / total_correlations * 100:.1f}%)")
    
    # Detailed analysis for each category
    all_summary_data = []
    
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        correlations = correlations_by_well_count[n_wells]
        
        if not correlations:
            continue
            
        print(f"\n{'='*100}")
        print(f"CORRELATIONS FOUND IN {n_wells} WELLS (Top {min(top_n, len(correlations))})")
        print(f"{'='*100}")
        
        # Category statistics
        positive_count = sum(1 for _, _, info in correlations if info['correlation_type'] == 'Positive')
        negative_count = sum(1 for _, _, info in correlations if info['correlation_type'] == 'Negative')
        consistent_count = sum(1 for _, _, info in correlations if info['consistent_direction'])
        significant_count = sum(1 for _, _, info in correlations if info.get('significant_ratio', 0) >= 0.5)
        
        print(f"\n📊 Category Statistics:")
        print(f"  Total pairs: {len(correlations)}")
        print(f"  Positive correlations: {positive_count} ({positive_count/len(correlations)*100:.1f}%)")
        print(f"  Negative correlations: {negative_count} ({negative_count/len(correlations)*100:.1f}%)")
        print(f"  Consistent direction: {consistent_count} ({consistent_count/len(correlations)*100:.1f}%)")
        print(f"  Mostly significant (>50% wells): {significant_count}")
        
        # Detailed table
        print(f"\n{'Rank':<5} {'Log Variable':<20} {'Lab Variable':<30} {'Avg r':<10} {'|Avg r|':<10} {'Std':<8} {'Range':<8} {'n̄':<8} {'Type':<10}")
        print("-" * 120)
        
        for i, (pair, wells_data, info) in enumerate(correlations[:top_n]):
            log_var, lab_var = pair
            log_name = log_var.replace('Log_', '')
            lab_name = lab_var.replace('Lab_', '')
            
            # Add to summary data for export
            summary_row = {
                'n_wells': n_wells,
                'rank': i + 1,
                'log_variable': log_name,
                'lab_variable': lab_name,
                'avg_correlation': info['avg_corr'],
                'avg_abs_correlation': info['avg_abs_corr'],
                'std_correlation': info['std_corr'],
                'min_correlation': info['min_corr'],
                'max_correlation': info['max_corr'],
                'range_correlation': info['range_corr'],
                'correlation_type': info['correlation_type'],
                'consistent_direction': info['consistent_direction'],
                'avg_sample_size': info['avg_sample_size'],
                'total_samples': info['total_samples'],
                'significant_ratio': info['significant_ratio'],
                'strength_category': info['strength_category']
            }
            
            # Add individual well correlations
            for well, r in wells_data:
                summary_row[f'{well}_r'] = r
            
            all_summary_data.append(summary_row)
            
            print(f"{i+1:<5} {log_name:<20} {lab_name:<30} "
                  f"{info['avg_corr']:>9.3f} {info['avg_abs_corr']:>9.3f} "
                  f"{info['std_corr']:>7.3f} {info['range_corr']:>7.3f} "
                  f"{info['avg_sample_size']:>7.0f} {info['correlation_type']:<10}")
            
            # Show individual well values with additional info
            well_details = []
            for well, r in wells_data:
                well_name = well.split('_')[-1]
                well_details.append(f"{well_name}:{r:.3f}")
            print(f"      Wells: {', '.join(well_details)}")
            
            if not info['consistent_direction']:
                print(f"      ⚠️  Mixed correlation signs across wells")
            
            if info['significant_ratio'] < 0.5:
                print(f"      ⚠️  Low significance ratio: {info['significant_ratio']:.2f}")
    
    # Export comprehensive report if path provided
    if export_path:
        summary_df = pd.DataFrame(all_summary_data)
        summary_df.to_csv(export_path, index=False)
        print(f"\n✅ Detailed summary exported to: {export_path}")
    
    # Print top correlations across all categories
    print(f"\n{'='*100}")
    print("TOP CORRELATIONS ACROSS ALL CATEGORIES")
    print(f"{'='*100}")
    
    all_correlations = []
    for n_wells, corrs in correlations_by_well_count.items():
        for pair, wells_data, info in corrs:
            all_correlations.append((pair, wells_data, info, n_wells))
    
    # Sort by absolute correlation
    all_correlations.sort(key=lambda x: x[2]['avg_abs_corr'], reverse=True)
    
    print(f"\n🏆 Top 15 Strongest Correlations Overall:")
    print("-" * 100)
    for i, (pair, wells_data, info, n_wells) in enumerate(all_correlations[:15]):
        log_var, lab_var = pair
        print(f"{i+1:2}. {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: "
              f"|r̄| = {info['avg_abs_corr']:.3f} ({n_wells} wells, {info['strength_category']})")
    
    # Add new visualization summary
    print(f"\n📊 VISUALIZATION SUMMARY:")
    print("-" * 80)
    print("Common correlations (≥2 wells) are visualized in:")
    print("  • Correlation coverage heatmap: imgs/correlation_coverage_heatmap.png")
    print("  • Scatter plots: imgs/scatter_plots/positive/ and imgs/scatter_plots/negative/")
    print("  • Consistency analysis: imgs/correlation_consistency_analysis.png")
    
    return pd.DataFrame(all_summary_data) if all_summary_data else None

def find_common_correlations(well_correlations, min_correlation=0.5, min_wells=2):
    """
    Find correlations that appear in multiple wells.
    Enhanced version that uses the new comprehensive functions.
    """
    # Use the enhanced function to get all correlations
    correlations_by_well_count, report_data = find_all_correlations_by_well_count(
        well_correlations, min_correlation=min_correlation
    )
    
    # Combine correlations from different well counts based on min_wells
    common_correlations = []
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        if n_wells >= min_wells:
            common_correlations.extend(correlations_by_well_count[n_wells])
    
    # Add report data as attribute for access if needed
    
    
    return common_correlations


def check_data_distribution(df, log_var, lab_var):
    """Check data distribution and identify potential outliers"""
    
    data = df[[log_var, lab_var, 'Well']].dropna()
    
    print(f"\nData distribution for {log_var} vs {lab_var}:")
    print(f"Total samples: {len(data)}")
    
    # Basic statistics
    print(f"\n{log_var} statistics:")
    print(data[log_var].describe())
    
    print(f"\n{lab_var} statistics:")
    print(data[lab_var].describe())
    
    # Check for potential outliers
    for var in [log_var, lab_var]:
        q1, q3 = data[var].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = data[(data[var] < q1 - 1.5*iqr) | (data[var] > q3 + 1.5*iqr)]
        
        if len(outliers) > 0:
            print(f"\nPotential outliers in {var}: {len(outliers)} samples")
            print(f"Outlier range: [{outliers[var].min():.2f}, {outliers[var].max():.2f}]")
            print(f"Normal range: [{q1 - 1.5*iqr:.2f}, {q3 + 1.5*iqr:.2f}]")

#  visuals

def create_comprehensive_correlation_scatter_plots(df_all, correlations_by_well_count, min_correlation, max_plots_per_figure=20):
    """
    Create two comprehensive scatter plot figures: one for positive and one for negative correlations.
    Only shows correlations that appear in 2+ wells.
    Plots are sorted by correlation strength and include combined statistics.
    Also saves individual plots in organized folders.
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
        if len(valid_data) >= 10:  # Minimum samples for correlation
            r, p = pearsonr(valid_data[log_var], valid_data[lab_var])
            return r
        return np.nan
    
    # NEW: Get all possible log-lab combinations to match heatmap
    log_columns = [col for col in df_all.columns if col.startswith('Log_') and 
                   col not in ['Log_Depth', 'Log_FRAMENO']]
    lab_columns = [col for col in df_all.columns if col.startswith('Lab_') and 
                   col not in ['Lab_Depth', 'Lab_Sample_ID']]
    
    # Calculate combined correlations for ALL pairs (like the heatmap does)
    all_combined_correlations = []
    min_samples_per_well = 10
    
    for log_var in log_columns:
        for lab_var in lab_columns:
            # Check which wells have sufficient data
            wells_with_data = []
            for well in df_all['Well'].unique():
                well_data = df_all[df_all['Well'] == well]
                valid_data = well_data[[log_var, lab_var]].dropna()
                if len(valid_data) >= min_samples_per_well:
                    r, _ = pearsonr(valid_data[log_var], valid_data[lab_var])
                    wells_with_data.append((well, r))
            
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
                        'consistent_direction': all(r > 0 for _, r in wells_with_data) or all(r < 0 for _, r in wells_with_data)
                    }
                    
                    all_combined_correlations.append(((log_var, lab_var), wells_with_data, info, combined_r))
    
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
    
    # Check for mixed-sign correlations
    mixed_count = 0
    for pair, wells_data, info in all_positive_corrs + all_negative_corrs:
        correlations = [r for _, r in wells_data]
        if len(set(np.sign(correlations))) > 1:  # Mixed signs
            mixed_count += 1
    
    print(f"Correlations with mixed signs across wells: {mixed_count}")
    
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
    
    # Create figure for positive correlations
    if all_positive_corrs:
        n_positive = min(len(all_positive_corrs), max_plots_per_figure)
        n_cols = 4
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
        
        for idx, (pair, wells_data, info) in enumerate(all_positive_corrs):
            # Get the combined r value for this pair
            combined_r = positive_correlations[idx][3]
            
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
                    combined_r = positive_correlations[idx][3]  # Use pre-calculated value
                    
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
    
    # Create figure for negative correlations
    if all_negative_corrs:
        n_negative = min(len(all_negative_corrs), max_plots_per_figure)
        n_cols = 4
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
        
        for idx, (pair, wells_data, info) in enumerate(all_negative_corrs):
            # Get the combined r value for this pair
            combined_r = negative_correlations[idx][3]
            
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
                    combined_r = negative_correlations[idx][3]  # Use pre-calculated value
                    
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
        
        print(f"✅ Saved {len(all_negative_corrs)} negative correlation plots to: {negative_dir}")
    
    # Create summary statistics
    print("\n" + "="*80)
    print("SCATTER PLOT GENERATION SUMMARY (2+ WELLS ONLY)")
    print("="*80)
    
    print("\nINDIVIDUAL PLOTS SAVED:")
    print("-" * 60)
    print(f"Positive correlations: imgs/scatter_plots/positive/")
    print(f"   - {len(all_positive_corrs)} plots saved")
    print(f"Negative correlations: imgs/scatter_plots/negative/")
    print(f"   - {len(all_negative_corrs)} plots saved")
    
    print("\nFILE NAMING CONVENTION:")
    print("Format: ###_LogVar_vs_LabVar_rValue.png")
    print("### = Ranking by correlation strength")
    print("LogVar = Log measurement name")
    print("LabVar = Lab measurement name")
    print("rValue = Combined correlation value")
    
    print("\nCOMBINED PLOTS:")
    print(f"Total positive correlations (2+ wells): {len(all_positive_corrs)}")
    print(f"Total negative correlations (2+ wells): {len(all_negative_corrs)}")
    print(f"Total correlations visualized in combined plots: {min(len(all_positive_corrs), max_plots_per_figure) + min(len(all_negative_corrs), max_plots_per_figure)}")
    
    # Distribution by well count (2+ wells only)
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        if n_wells >= 2:  # Only show 2+ wells
            corrs = correlations_by_well_count[n_wells]
            if corrs:
                # Count correlations that meet the threshold
                corrs_meeting_threshold = []
                for pair, wells_data, info in corrs:
                    log_var, lab_var = pair
                    combined_r = calculate_combined_correlation(df_all, log_var, lab_var)
                    if not np.isnan(combined_r) and abs(combined_r) >= min_correlation:
                        corrs_meeting_threshold.append((pair, wells_data, info, combined_r))
                
                pos_count = sum(1 for _, _, _, r in corrs_meeting_threshold if r > 0)
                neg_count = sum(1 for _, _, _, r in corrs_meeting_threshold if r < 0)
                print(f"\n{n_wells} wells: {len(corrs_meeting_threshold)} total meeting |r| ≥ {min_correlation} ({pos_count} positive, {neg_count} negative)")
    
    # Top correlations summary
    print("\nTop 10 Positive Correlations (by combined r, 2+ wells):")
    for i, (pair, wells_data, info, combined_r) in enumerate(positive_correlations[:10]):
        log_var, lab_var = pair
        wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
        print(f"{i+1}. {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: " +
              f"Combined r = {combined_r:.3f} (Wells: {', '.join(wells_involved)})")
    
    print("\nTop 10 Negative Correlations (by combined r, 2+ wells):")
    for i, (pair, wells_data, info, combined_r) in enumerate(negative_correlations[:10]):
        log_var, lab_var = pair
        wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
        print(f"{i+1}. {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: " +
              f"Combined r = {combined_r:.3f} (Wells: {', '.join(wells_involved)})")

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

def create_combined_correlation_heatmap(df_all, lab_columns, log_columns, well_correlations):
    """
    Create heatmaps showing correlations calculated from combined data across all wells.
    This differs from the average correlation by pooling all data points together.
    """
    
    # Calculate combined correlations (pooling all data)
    combined_corr_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=float)
    sample_size_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=int)
    
    # Track which wells have sufficient data for each pair (matching the correlation calculation criteria)
    wells_with_data_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=object)
    
    # Define minimum samples required for correlation (same as in calculate_correlations_by_well)
    min_samples = 10
    
    for log_var in log_columns:
        for lab_var in lab_columns:
            # Get combined data
            combined_data = df_all[[log_var, lab_var, 'Well']].dropna()
            
            if len(combined_data) > 5:
                r, p = stats.pearsonr(combined_data[log_var], combined_data[lab_var])
                combined_corr_matrix.loc[log_var, lab_var] = r
                sample_size_matrix.loc[log_var, lab_var] = len(combined_data)
                
                # Count wells with sufficient data (matching the criteria from calculate_correlations_by_well)
                wells_with_sufficient_data = []
                for well in sorted(df_all['Well'].unique()):
                    well_data = df_all[df_all['Well'] == well][[log_var, lab_var]].dropna()
                    if len(well_data) >= min_samples:  # Use same threshold as correlation calculation
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
    
    # --- Figure 1: Combined Correlations with Clean Annotations ---
    fig1, ax1 = plt.subplots(figsize=(18, 14))  # Increased height for legend
    
    # Create custom annotations - simplified format with just correlation and number of wells
    annot_text = np.empty_like(combined_corr_matrix, dtype=object)
    for i in range(len(combined_corr_matrix.index)):
        for j in range(len(combined_corr_matrix.columns)):
            r = combined_corr_matrix.iloc[i, j]
            n = sample_size_matrix.iloc[i, j]
            wells_list = wells_with_data_matrix.iloc[i, j]
            
            if pd.notna(r) and n > 0:
                # Only count wells with sufficient data (≥ min_samples)
                n_wells = len(wells_list)
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
        cbar_kws={'label': 'Combined Correlation (r)', 'shrink': 0.8, 'pad': 0.02},
        linewidths=0.8,
        linecolor='white',
        square=True,
        mask=mask,
        ax=ax1
    )
    
    # Improve title and labels
    ax1.set_title('Combined Correlations: All Wells Pooled Together\n' + 
                  f'Pearson correlation coefficient with number of wells (≥{min_samples} samples per well)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Laboratory Measurements', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Geophysical Log Measurements', fontsize=13, fontweight='bold')
    
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
    plt.subplots_adjust(top=0.85)  # Make room for legend at top
    plt.savefig('imgs/heatmap_combined_correlations_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # --- Figure 3: Strong correlations only with emphasis ---
    fig3, ax3 = plt.subplots(figsize=(18, 14))  # Increased height for legend
    
    # Create mask for weak correlations
    strong_threshold = 0.5
    mask_weak = np.abs(combined_corr_matrix) < strong_threshold
    
    # Count strong correlations
    n_strong = (~mask_weak & ~pd.isna(combined_corr_matrix)).sum().sum()
    n_very_strong = (np.abs(combined_corr_matrix) >= 0.7).sum().sum()
    
    # Create annotations only for strong correlations - simplified format
    strong_annot_text = np.empty_like(combined_corr_matrix, dtype=object)
    for i in range(len(combined_corr_matrix.index)):
        for j in range(len(combined_corr_matrix.columns)):
            r = combined_corr_matrix.iloc[i, j]
            n = sample_size_matrix.iloc[i, j]
            wells_list = wells_with_data_matrix.iloc[i, j]
            
            if pd.notna(r) and abs(r) >= strong_threshold and n > 0:
                # Only count wells with sufficient data
                n_wells = len(wells_list)
                strong_annot_text[i, j] = f'{r:.2f}\n({n_wells}w)'
            else:
                strong_annot_text[i, j] = ''
    
    sns.heatmap(
        combined_corr_matrix.astype(float),
        annot=strong_annot_text,
        fmt='',
        annot_kws={'size': 8, 'va': 'center', 'ha': 'center', 'weight': 'bold'},
        cmap='RdYlGn',
        center=0, vmin=-1, vmax=1,
        cbar_kws={'label': 'Strong Correlations (|r| ≥ 0.5)', 'shrink': 0.8, 'pad': 0.02},
        linewidths=0.8,
        linecolor='white',
        square=True,
        mask=mask_weak,
        ax=ax3
    )
    
    # Improve title and labels
    ax3.set_title(f'Strong Combined Correlations Only (|r| ≥ 0.5)\n' +
                  f'Pearson correlation coefficient with number of wells (≥{min_samples} samples per well)', 
                  fontsize=16, fontweight='bold', pad=20)
    ax3.set_xlabel('Laboratory Measurements', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Geophysical Log Measurements', fontsize=13, fontweight='bold')
    
    # Clean tick labels
    ax3.set_xticklabels([col.replace('Lab_', '') for col in combined_corr_matrix.columns], 
                        rotation=45, ha='right', fontsize=10)
    ax3.set_yticklabels([row.replace('Log_', '') for row in combined_corr_matrix.index], 
                        rotation=0, fontsize=10)
    
    # Add grid for better readability
    ax3.set_facecolor('#f8f8f8')
    
    # Add LOG_DESCRIPTIONS legend in top left for strong correlations plot too
    legend_text = "Log Descriptions:\n" + "-"*50 + "\n"
    for log_var in log_vars_in_plot:
        if log_var in LOG_DESCRIPTIONS:
            legend_text += f"{log_var:<6}: {LOG_DESCRIPTIONS[log_var]}\n"
    
    # Add text box with descriptions in top left corner
    plt.figtext(0.02, 0.98, legend_text, fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Make room for legend at top
    plt.savefig('imgs/heatmap_combined_strong_only_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print improved summary statistics
    print("\n" + "="*80)
    print("COMBINED CORRELATION ANALYSIS - IMPROVED SUMMARY")
    print("="*80)
    
    print("\n📊 CORRELATION STRENGTH DISTRIBUTION:")
    print("-" * 60)
    total_valid = (~pd.isna(combined_corr_matrix)).sum().sum()
    for threshold, label in [(0.7, "Very Strong"), (0.5, "Strong"), (0.3, "Moderate")]:
        count = (np.abs(combined_corr_matrix) >= threshold).sum().sum()
        percentage = (count / total_valid) * 100 if total_valid > 0 else 0
        print(f"{label} (|r| ≥ {threshold}): {count} ({percentage:.1f}%)")
    
    print("\n📈 SAMPLE SIZE DISTRIBUTION:")
    print("-" * 60)
    valid_sizes = sample_size_matrix[sample_size_matrix > 0].values.flatten()
    print(f"Range: {valid_sizes.min()} - {valid_sizes.max()} samples")
    print(f"Mean: {valid_sizes.mean():.0f} samples")
    print(f"Median: {np.median(valid_sizes):.0f} samples")
    
    # Find correlations with highest confidence (large n and strong r)
    confidence_scores = np.abs(combined_corr_matrix) * np.sqrt(sample_size_matrix)
    top_confidence_idx = np.unravel_index(np.nanargmax(confidence_scores), confidence_scores.shape)
    
    print("\n🎯 HIGHEST CONFIDENCE CORRELATION:")
    print("-" * 60)
    log_var = combined_corr_matrix.index[top_confidence_idx[0]]
    lab_var = combined_corr_matrix.columns[top_confidence_idx[1]]
    wells_for_top = wells_with_data_matrix.iloc[top_confidence_idx]
    print(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}")
    print(f"r = {combined_corr_matrix.iloc[top_confidence_idx]:.3f}, n = {sample_size_matrix.iloc[top_confidence_idx]}")
    print(f"Wells with sufficient data (≥{min_samples} samples): {len(wells_for_top)} wells")
    
    # Print log descriptions in summary
    print("\n📖 LOG MEASUREMENT DESCRIPTIONS:")
    print("-" * 60)
    for log_var in log_vars_in_plot:
        if log_var in LOG_DESCRIPTIONS:
            print(f"{log_var:<6}: {LOG_DESCRIPTIONS[log_var]}")
    
    return combined_corr_matrix, sample_size_matrix

def create_seaborn_pairplot_with_correlations(df_all, variables_to_plot=None, sample_size=2000, save_path='imgs/'):
    """
    Create a seaborn-style pairplot with correlation values in lower triangle.
    
    Parameters:
    -----------
    df_all : DataFrame
        Combined data from all wells
    variables_to_plot : list, optional
        List of variables to include. If None, will select top correlated pairs
    sample_size : int
        Number of samples to use for plotting
    save_path : str
        Path to save the plots
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import pearsonr
    
    # If no variables specified, select based on strongest correlations
    if variables_to_plot is None:
        # Get log and lab columns
        log_cols = [col for col in df_all.columns if col.startswith('Log_') and 
                   col not in ['Log_Depth', 'Log_FRAMENO']]
        lab_cols = [col for col in df_all.columns if col.startswith('Lab_') and 
                   col not in ['Lab_Depth', 'Lab_Sample_ID']]
        
        # Find strongest correlations to select variables
        corr_pairs = []
        for log_col in log_cols[:10]:
            for lab_col in lab_cols[:10]:
                valid_data = df_all[[log_col, lab_col]].dropna()
                if len(valid_data) > 10:
                    r, _ = pearsonr(valid_data[log_col], valid_data[lab_col])
                    if abs(r) > 0.3:
                        corr_pairs.append((abs(r), log_col, lab_col))
        
        # Get unique variables from top correlations
        corr_pairs.sort(reverse=True)
        variables_to_plot = []
        for _, log_col, lab_col in corr_pairs[:4]:  # Top 4 pairs
            if log_col not in variables_to_plot:
                variables_to_plot.append(log_col)
            if lab_col not in variables_to_plot:
                variables_to_plot.append(lab_col)
        
        # Limit to 8 variables for readability
        variables_to_plot = variables_to_plot[:8]
    
    if len(variables_to_plot) < 2:
        print("Not enough variables to create pair plot")
        return
    
    print(f"Creating pairplot for variables: {[v.replace('Log_', '').replace('Lab_', '') for v in variables_to_plot]}")
    
    # Sample data if too large
    if len(df_all) > sample_size:
        df_plot = df_all[variables_to_plot + ['Well']].sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} points from {len(df_all)} total")
    else:
        df_plot = df_all[variables_to_plot + ['Well']].copy()
    
    # Define color palette for wells
    well_palette = {
        'HRDH_697': "#3c00ff",
        'HRDH_1119':  '#d62728',
        'HRDH_1804': "#00ff00",
        'HRDH_1867': "#ff7700"
    }
    
    # Create the pairplot
    g = sns.pairplot(df_plot, 
                     hue='Well',
                     palette=well_palette,
                     diag_kind='hist',
                     plot_kws={'alpha': 0.6, 's': 20},
                     diag_kws={'alpha': 0.7})
    
    # Rename the variables on axes
    for ax in g.axes.flatten():
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace('Log_', '').replace('Lab_', ''))
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel().replace('Log_', '').replace('Lab_', ''))
    
    # Calculate and add correlation values to lower triangle
    n_vars = len(variables_to_plot)
    for i in range(n_vars):
        for j in range(i):  # Lower triangle only
            ax = g.axes[i, j]
            
            # Calculate correlation using all data
            var_x = variables_to_plot[j]
            var_y = variables_to_plot[i]
            valid_data = df_all[[var_x, var_y]].dropna()
            
            if len(valid_data) > 3:
                r, p = pearsonr(valid_data[var_x], valid_data[var_y])
                
                # Clear the scatter plot and show correlation value
                ax.clear()
                
                # Set background color based on correlation strength
                if abs(r) > 0.7:
                    bg_color = '#2ecc71' if r > 0 else '#e74c3c'  # Strong: green/red
                elif abs(r) > 0.5:
                    bg_color = '#a9dfbf' if r > 0 else '#f5b7b1'  # Medium: light green/red
                elif abs(r) > 0.3:
                    bg_color = '#d5f4e6' if r > 0 else '#fadbd8'  # Weak: very light green/red
                else:
                    bg_color = '#f8f9f9'  # Very weak: almost white
                
                ax.set_facecolor(bg_color)
                
                # Add correlation text
                ax.text(0.5, 0.5, f'{r:.2f}', 
                       transform=ax.transAxes,
                       ha='center', va='center',
                       fontsize=14, fontweight='bold')
                
                # Add p-value if significant
                if p < 0.001:
                    ax.text(0.5, 0.3, '***', 
                           transform=ax.transAxes,
                           ha='center', va='center',
                           fontsize=10)
                elif p < 0.01:
                    ax.text(0.5, 0.3, '**', 
                           transform=ax.transAxes,
                           ha='center', va='center',
                           fontsize=10)
                elif p < 0.05:
                    ax.text(0.5, 0.3, '*', 
                           transform=ax.transAxes,
                           ha='center', va='center',
                           fontsize=10)
                
                # Remove ticks
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Keep labels for edge plots
                if j == 0:
                    ax.set_ylabel(var_y.replace('Log_', '').replace('Lab_', ''))
                if i == n_vars - 1:
                    ax.set_xlabel(var_x.replace('Log_', '').replace('Lab_', ''))
    
    # Add title
    g.fig.suptitle('Pairplot with Correlations - All Wells', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure using the figure object BEFORE showing
    g.fig.savefig(f'{save_path}seaborn_pairplot_with_correlations.png', dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return g

def create_correlation_heatmap_matrix(df_all, variables_to_plot=None, save_path='imgs/'):
    """
    Create a correlation heatmap matrix similar to the reference image.
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    
    if variables_to_plot is None:
        # Select top log and lab variables based on data availability
        log_cols = [col for col in df_all.columns if col.startswith('Log_') and 
                   col not in ['Log_Depth', 'Log_FRAMENO']]
        lab_cols = [col for col in df_all.columns if col.startswith('Lab_') and 
                   col not in ['Lab_Depth', 'Lab_Sample_ID']]
        
        # Select variables with most data
        data_coverage = df_all[log_cols + lab_cols].notna().sum()
        top_vars = data_coverage.nlargest(8).index.tolist()
        variables_to_plot = top_vars
    
    # Calculate correlation matrix
    corr_matrix = df_all[variables_to_plot].corr()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Create custom colormap
    # cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    # Draw the heatmap
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='RdYlGn',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": .8},
                annot=True,
                fmt='.2f',
                annot_kws={'fontsize': 10})
    
    # Customize labels
    labels = [label.replace('Log_', '').replace('Lab_', '') for label in variables_to_plot]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)
    
    plt.title('Correlation Matrix - All Wells Combined', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save figure using the figure object BEFORE showing
    fig.savefig(f'{save_path}correlation_heatmap_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return corr_matrix

# Function to create pairplot for each group
def create_grouped_pairplots(df_all, variable_groups, sample_size=1500):
    """Create separate pairplots for each variable group."""
    
    for group_name, group_info in variable_groups.items():
        
        if group_name == "Top_Correlations":
            # Dynamically select top correlated variables
            # Get top 4 log and top 4 lab variables from your correlation analysis
            if 'consistency_df' in globals() and not consistency_df.empty:
                top_pairs = consistency_df.head(8)
                logs_set = set()
                labs_set = set()
                
                for _, row in top_pairs.iterrows():
                    logs_set.add(row['Log_Variable'])
                    labs_set.add(row['Lab_Variable'])
                
                group_info['logs'] = list(logs_set)[:4]
                group_info['labs'] = list(labs_set)[:4]
        
        # Combine logs and labs for this group
        variables = group_info['logs'] + group_info['labs']
        
        # Skip if not enough variables
        if len(variables) < 2:
            print(f"Skipping {group_name}: Not enough variables")
            continue
        
        # Check data availability
        available_vars = [v for v in variables if v in df_all.columns]
        if len(available_vars) < 2:
            print(f"Skipping {group_name}: Variables not found in dataset")
            continue
        
        print(f"\nCreating pairplot for {group_name}...")
        print(f"Description: {group_info['description']}")
        print(f"Variables ({len(available_vars)}): {', '.join([v.replace('Log_', '').replace('Lab_', '') for v in available_vars])}")
        
        # Create the pairplot
        try:
            g = create_seaborn_pairplot_with_correlations(
                df_all, 
                variables_to_plot=available_vars,
                sample_size=sample_size
            )
            
            # Update title
            if g:
                g.fig.suptitle(f'{group_name}: {group_info["description"]}', 
                             fontsize=16, y=1.02)
                
                # Save with descriptive filename using the figure object
                g.fig.savefig(f'imgs/pairplot_{group_name.lower()}.png', 
                           dpi=300, bbox_inches='tight')
                
                # Show the plot in the notebook
                plt.show()
                
                # Close after showing to free memory
                plt.close(g.fig)
                
        except Exception as e:
            print(f"   ⚠️ Error creating pairplot for {group_name}: {e}")

