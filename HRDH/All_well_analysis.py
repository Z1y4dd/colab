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
                
                if n_samples >= min_samples:  # Configurable minimum sample threshold
                    r, p = pearsonr(valid_data[log_col], valid_data[lab_col])
                    corr_matrix.loc[log_col, lab_col] = r
                    pvalue_matrix.loc[log_col, lab_col] = p
                    sample_size_matrix.loc[log_col, lab_col] = n_samples
                    valid_pairs += 1
                else:
                    sample_size_matrix.loc[log_col, lab_col] = n_samples
        
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
            'total_samples': len(well_data)
        }
        
        # Print summary
        print(f"\n{well}:")
        print(f"  Total samples: {len(well_data):,}")
        print(f"  Valid correlations: {valid_pairs}/{total_pairs} ({well_correlation_stats[well]['coverage']:.1f}%)")
        print(f"  Mean |r|: {well_correlation_stats[well]['mean_abs_correlation']:.3f}")
        print(f"  Strong correlations (|r| ≥ 0.7): {well_correlation_stats[well]['strong_correlations']}")
        print(f"  Moderate correlations (0.5 ≤ |r| < 0.7): {well_correlation_stats[well]['moderate_correlations']}")
    
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

# new

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


def analyze_correlation_consistency(common_correlations, well_correlations, export_path=None):
    """
    Analyze how consistent correlations are across wells.
    Enhanced version with comprehensive statistical analysis and visualization.
    """
    consistency_data = []
    
    # Handle both old and new formats
    correlations_to_analyze = common_correlations[:50]  # Analyze top 50 for better coverage
    
    for item in correlations_to_analyze:
        if len(item) == 3:
            pair, wells_data, info = item
        else:
            pair, wells_data = item
            info = {}
            
        log_var, lab_var = pair
        
        # Collect all correlation data across wells
        all_well_data = []
        for well, matrices in well_correlations.items():
            corr_matrix = matrices['correlation'] if isinstance(matrices, dict) else matrices
            pvalue_matrix = matrices.get('pvalue', None) if isinstance(matrices, dict) else None
            sample_size_matrix = matrices.get('sample_size', None) if isinstance(matrices, dict) else None
            
            if log_var in corr_matrix.index and lab_var in corr_matrix.columns:
                r = corr_matrix.loc[log_var, lab_var]
                if not pd.isna(r):
                    well_data = {
                        'well': well,
                        'correlation': r,
                        'pvalue': pvalue_matrix.loc[log_var, lab_var] if pvalue_matrix is not None else np.nan,
                        'sample_size': sample_size_matrix.loc[log_var, lab_var] if sample_size_matrix is not None else np.nan
                    }
                    all_well_data.append(well_data)
        
        if len(all_well_data) >= 2:
            r_values = [d['correlation'] for d in all_well_data]
            p_values = [d['pvalue'] for d in all_well_data if not pd.isna(d['pvalue'])]
            n_samples = [d['sample_size'] for d in all_well_data if not pd.isna(d['sample_size'])]
            
            # Calculate comprehensive consistency metrics
            consistency_metrics = {
                'Variable_Pair': f'{log_var.replace("Log_", "")} vs {lab_var.replace("Lab_", "")}',
                'Log_Variable': log_var,
                'Lab_Variable': lab_var,
                'N_Wells': len(all_well_data),
                'Mean_r': np.mean(r_values),
                'Median_r': np.median(r_values),
                'Std_r': np.std(r_values),
                'Min_r': np.min(r_values),
                'Max_r': np.max(r_values),
                'Range_r': np.max(r_values) - np.min(r_values),
                'CV_r': np.std(r_values) / np.mean(np.abs(r_values)) if np.mean(np.abs(r_values)) > 0 else np.nan,
                'All_Same_Sign': all(r > 0 for r in r_values) or all(r < 0 for r in r_values),
                'Correlation_Type': 'Positive' if np.mean(r_values) > 0 else 'Negative',
                'Avg_Sample_Size': np.mean(n_samples) if n_samples else np.nan,
                'Total_Samples': np.sum(n_samples) if n_samples else np.nan,
                'Avg_P_Value': np.mean(p_values) if p_values else np.nan,
                'All_Significant': all(p < 0.05 for p in p_values) if p_values else False
            }
            
            # Add individual well correlations
            for well_data in all_well_data:
                well_name = well_data['well']
                consistency_metrics[f'{well_name}_r'] = well_data['correlation']
                consistency_metrics[f'{well_name}_p'] = well_data['pvalue']
                consistency_metrics[f'{well_name}_n'] = well_data['sample_size']
            
            # Calculate consistency score (lower is better)
            consistency_metrics['Consistency_Score'] = consistency_metrics['Std_r'] / abs(consistency_metrics['Mean_r'])
            
            consistency_data.append(consistency_metrics)
    
    # Create DataFrame and sort by consistency
    consistency_df = pd.DataFrame(consistency_data)
    consistency_df = consistency_df.sort_values('Std_r')
    
    # Add rankings
    consistency_df['Consistency_Rank'] = range(1, len(consistency_df) + 1)
    consistency_df['Strength_Rank'] = consistency_df['Mean_r'].abs().rank(ascending=False, method='min').astype(int)
    
    # Export if path provided
    if export_path:
        consistency_df.to_csv(export_path, index=False)
        print(f"\n✅ Consistency analysis exported to: {export_path}")
    
    # Create enhanced visualization
    if len(consistency_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Consistency vs Strength scatter with well count color coding
        ax1 = axes[0, 0]
        scatter = ax1.scatter(consistency_df['Mean_r'].abs(), 
                             consistency_df['Std_r'],
                             c=consistency_df['N_Wells'],
                             s=consistency_df['Total_Samples']/10,
                             alpha=0.6,
                             cmap='viridis')
        ax1.set_xlabel('Average |r|')
        ax1.set_ylabel('Standard Deviation')
        ax1.set_title('Correlation Consistency vs Strength\n(Size = Sample Count)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Number of Wells')
        
        # Add annotations for top 5 most consistent
        for i, row in consistency_df.head(5).iterrows():
            ax1.annotate(row['Variable_Pair'], 
                        (abs(row['Mean_r']), row['Std_r']),
                        fontsize=8, alpha=0.7)
        
        # 2. Top consistent correlations with error bars
        ax2 = axes[0, 1]
        top_consistent = consistency_df.head(10)
        y_pos = np.arange(len(top_consistent))
        ax2.barh(y_pos, top_consistent['Mean_r'].abs())
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_consistent['Variable_Pair'], fontsize=8)
        ax2.set_xlabel('Average |r|')
        ax2.set_title('Top 10 Most Consistent Correlations')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # Add error bars
        ax2.errorbar(top_consistent['Mean_r'].abs(), y_pos, 
                    xerr=top_consistent['Std_r'], 
                    fmt='none', color='red', alpha=0.5)
        
        # 3. Correlation range plot
        ax3 = axes[1, 0]
        top_15 = consistency_df.head(15)
        for i, row in enumerate(top_15.iterrows()):
            _, data = row
            ax3.plot([data['Min_r'], data['Max_r']], [i, i], 'b-', linewidth=2)
            ax3.plot(data['Mean_r'], i, 'ro', markersize=8)
        ax3.set_yticks(range(len(top_15)))
        ax3.set_yticklabels(top_15['Variable_Pair'], fontsize=8)
        ax3.set_xlabel('Correlation Range')
        ax3.set_title('Correlation Ranges Across Wells (Top 15)')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        
        # 4. Well distribution histogram
        ax4 = axes[1, 1]
        well_counts = consistency_df['N_Wells'].value_counts().sort_index()
        ax4.bar(well_counts.index, well_counts.values)
        ax4.set_xlabel('Number of Wells')
        ax4.set_ylabel('Number of Correlations')
        ax4.set_title('Distribution of Correlations by Well Count')
        ax4.set_xticks(well_counts.index)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (wells, count) in enumerate(well_counts.items()):
            ax4.text(wells, count + 0.5, str(count), ha='center')
        
        plt.suptitle('Correlation Consistency Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('imgs/correlation_consistency_analysis_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Print enhanced summary
    print("\n" + "="*80)
    print("CORRELATION CONSISTENCY ANALYSIS")
    print("="*80)
    
    print(f"\nAnalyzed {len(consistency_df)} correlation pairs")
    
    if len(consistency_df) > 0:
        # Summary by number of wells
        print("\n📊 DISTRIBUTION BY NUMBER OF WELLS:")
        print("-" * 60)
        for n_wells in sorted(consistency_df['N_Wells'].unique()):
            subset = consistency_df[consistency_df['N_Wells'] == n_wells]
            print(f"{n_wells} wells: {len(subset)} correlations "
                  f"(avg σ = {subset['Std_r'].mean():.3f}, avg |r| = {subset['Mean_r'].abs().mean():.3f})")
        
        print("\n🏆 MOST CONSISTENT CORRELATIONS (lowest std):")
        print("-" * 80)
        for i, row in consistency_df.head(10).iterrows():
            print(f"{i+1:2}. {row['Variable_Pair']:<40} σ = {row['Std_r']:.3f}, "
                  f"|r̄| = {abs(row['Mean_r']):.3f}, n_wells = {row['N_Wells']}")
            
        print("\n💪 STRONGEST AVERAGE CORRELATIONS:")
        print("-" * 80)
        strongest = consistency_df.nlargest(10, 'Mean_r', keep='all')
        for i, (_, row) in enumerate(strongest.iterrows()):
            print(f"{i+1:2}. {row['Variable_Pair']:<40} |r̄| = {abs(row['Mean_r']):.3f}, "
                  f"σ = {row['Std_r']:.3f}, n_wells = {row['N_Wells']}")
        
        # Find problematic correlations
        print("\n⚠️  INCONSISTENT CORRELATIONS (high std or mixed signs):")
        print("-" * 80)
        problematic = consistency_df[
            (consistency_df['Std_r'] > 0.2) | (~consistency_df['All_Same_Sign'])
        ].head(10)
        
        if len(problematic) > 0:
            for i, row in problematic.iterrows():
                sign_issue = " [Mixed signs]" if not row['All_Same_Sign'] else ""
                print(f"• {row['Variable_Pair']:<40} σ = {row['Std_r']:.3f}, "
                      f"range = {row['Range_r']:.3f}{sign_issue}")
        else:
            print("No highly inconsistent correlations found!")
    
    return consistency_df


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

    
def analyze_correlation_distribution(common_correlations):
    """
    Analyze and visualize the distribution of correlation strengths.
    """
    import matplotlib.pyplot as plt
    
    # Extract all correlation values
    all_correlations = []
    for _, wells_data, _ in common_correlations:
        for _, r in wells_data:
            all_correlations.append(r)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram of all correlations
    ax1 = axes[0, 0]
    ax1.hist(all_correlations, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Correlation (r)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of All Correlations')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of absolute correlations
    ax2 = axes[0, 1]
    abs_correlations = [abs(r) for r in all_correlations]
    ax2.hist(abs_correlations, bins=20, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax2.axvline(x=0.7, color='green', linestyle='--', linewidth=2, label='Strong (0.7)')
    ax2.set_xlabel('Absolute Correlation |r|')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Absolute Correlations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot by number of wells
    ax3 = axes[1, 0]
    data_by_wells = {}
    for _, wells_data, info in common_correlations:
        n_wells = info['n_wells']
        if n_wells not in data_by_wells:
            data_by_wells[n_wells] = []
        data_by_wells[n_wells].extend([abs(r) for _, r in wells_data])
    
    box_data = [data_by_wells[n] for n in sorted(data_by_wells.keys())]
    box_labels = [f'{n} wells' for n in sorted(data_by_wells.keys())]
    
    ax3.boxplot(box_data, labels=box_labels)
    ax3.set_ylabel('Absolute Correlation |r|')
    ax3.set_title('Correlation Strength by Well Coverage')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter plot: Average correlation vs Standard deviation
    ax4 = axes[1, 1]
    avg_corrs = [info['avg_abs_corr'] for _, _, info in common_correlations]
    std_corrs = [info['std_corr'] for _, _, info in common_correlations]
    n_wells = [info['n_wells'] for _, _, info in common_correlations]
    
    scatter = ax4.scatter(avg_corrs, std_corrs, c=n_wells, s=50, alpha=0.6, cmap='viridis')
    ax4.set_xlabel('Average |r|')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Correlation Consistency')
    ax4.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Number of Wells')
    
    plt.tight_layout()
    plt.savefig('imgs/correlation_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print("\nCorrelation Distribution Statistics:")
    print(f"Total correlations: {len(all_correlations)}")
    print(f"Mean |r|: {np.mean(abs_correlations):.3f}")
    print(f"Median |r|: {np.median(abs_correlations):.3f}")
    print(f"Std |r|: {np.std(abs_correlations):.3f}")
    print(f"Range: [{min(all_correlations):.3f}, {max(all_correlations):.3f}]")

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

# new visuals

def create_correlation_coverage_heatmap(common_correlations, well_correlations, top_n=20):
    """
    Create a heatmap showing which correlations are present in which wells.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get top correlations
    top_correlations = common_correlations[:top_n]
    
    # Create matrix
    all_wells = sorted(well_correlations.keys())
    matrix_data = []
    pair_labels = []
    
    for pair, _, info in top_correlations:
        log_var, lab_var = pair
        pair_label = f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}"
        pair_labels.append(pair_label)
        
        row = []
        for well in all_wells:
            # Get correlation value for this well
            # Handle both old format (direct matrix) and new format (dict with matrices)
            if isinstance(well_correlations[well], dict):
                corr_matrix = well_correlations[well]['correlation']
            else:
                corr_matrix = well_correlations[well]
            
            r = corr_matrix.loc[log_var, lab_var]
            
            if pd.isna(r):
                row.append(np.nan)
            elif abs(r) >= 0.5:  # Using the threshold
                row.append(r)
            else:
                row.append(0)  # Below threshold
        
        matrix_data.append(row)
    
    # Create DataFrame
    coverage_df = pd.DataFrame(matrix_data, columns=all_wells, index=pair_labels)
    
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(12, max(8, len(pair_labels)*0.4)))
    
    # Heatmap 1: Correlation values
    sns.heatmap(coverage_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                vmin=-1, vmax=1, cbar_kws={'label': 'Correlation (r)'},
                ax=ax1, square=True)
    ax1.set_title('Correlation Values by Well', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Wells')
    ax1.set_ylabel('Variable Pairs')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('imgs/correlation_coverage_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_comprehensive_correlation_scatter_plots(df_all, correlations_by_well_count, max_plots_per_figure=20):
    """
    Create two comprehensive scatter plot figures: one for positive and one for negative correlations.
    Only shows correlations that appear in 2+ wells.
    Plots are sorted by correlation strength and include combined statistics.
    Also saves individual plots in organized folders.
    """
    
    # Define colors for each well
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#d62728' , 
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
        """Replace invalid filename characters with underscores."""
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']
        sanitized = name
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        return sanitized
    
    # Function to calculate combined correlation
    def calculate_combined_correlation(df_all, log_var, lab_var):
        """Calculate correlation from all wells' data combined."""
        valid_data = df_all[[log_var, lab_var]].dropna()
        if len(valid_data) < 3:
            return np.nan
        r, _ = stats.pearsonr(valid_data[log_var], valid_data[lab_var])
        return r
    
    # Collect correlations from 2+ wells only
    all_correlations_with_combined = []
    
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        if n_wells >= 2:  # Only include correlations in 2+ wells
            for pair, wells_data, info in correlations_by_well_count[n_wells]:
                log_var, lab_var = pair
                # Calculate the actual combined correlation
                combined_r = calculate_combined_correlation(df_all, log_var, lab_var)
                if not np.isnan(combined_r):
                    all_correlations_with_combined.append((pair, wells_data, info, combined_r))
    
    # Separate based on combined correlation sign
    positive_correlations = [(pair, wells_data, info, combined_r) 
                            for pair, wells_data, info, combined_r in all_correlations_with_combined 
                            if combined_r > 0]
    negative_correlations = [(pair, wells_data, info, combined_r) 
                            for pair, wells_data, info, combined_r in all_correlations_with_combined 
                            if combined_r < 0]
    
    # Sort by absolute combined correlation value (strongest first)
    positive_correlations.sort(key=lambda x: abs(x[3]), reverse=True)
    negative_correlations.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Remove combined_r from tuples for compatibility with rest of code
    all_positive_corrs = [(pair, wells_data, info) for pair, wells_data, info, _ in positive_correlations]
    all_negative_corrs = [(pair, wells_data, info) for pair, wells_data, info, _ in negative_correlations]
    
    print(f"\nCorrelation Classification Summary (2+ wells only):")
    print(f"Positive correlations (combined r > 0): {len(all_positive_corrs)}")
    print(f"Negative correlations (combined r < 0): {len(all_negative_corrs)}")
    
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
                    z_well = np.polyfit(x_data, y_data, 1)
                    p_well = np.poly1d(z_well)
                    x_line_well = np.linspace(x_data.min(), x_data.max(), 50)
                    ax.plot(x_line_well, p_well(x_line_well), 
                           color=well_colors[well], 
                           linestyle='--', 
                           alpha=0.5, 
                           linewidth=1.5)
                
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
                    
                    # Get valid data points
                    mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                    x_data = well_data.loc[mask, log_var].values
                    y_data = well_data.loc[mask, lab_var].values
                    
                    if len(x_data) > 0:
                        # Shorter well names in legend
                        well_short = well.replace('HRDH_', '')
                        ax.scatter(x_data, y_data, 
                                 color=well_colors[well], 
                                 alpha=0.6, 
                                 s=30,
                                 label=f'{well_short} (r={r:.2f}, n={len(x_data)})',
                                 edgecolors='darkgreen',
                                 linewidth=0.5)
                        
                        # Add individual well regression line (thinner, dashed)
                        if len(x_data) > 1:
                            z_well = np.polyfit(x_data, y_data, 1)
                            p_well = np.poly1d(z_well)
                            x_line_well = np.linspace(x_data.min(), x_data.max(), 50)
                            ax.plot(x_line_well, p_well(x_line_well), 
                                   color=well_colors[well], 
                                   linestyle='--', 
                                   alpha=0.5, 
                                   linewidth=1)
                        
                        all_x.extend(x_data)
                        all_y.extend(y_data)
                
                # Calculate combined statistics from ALL data points
                if len(all_x) > 3:
                    all_x = np.array(all_x)
                    all_y = np.array(all_y)
                    
                    # Use the pre-calculated combined_r
                    combined_p = stats.pearsonr(all_x, all_y)[1]
                    
                    # Calculate R-squared
                    z = np.polyfit(all_x, all_y, 1)
                    p = np.poly1d(z)
                    y_pred = p(all_x)
                    ss_res = np.sum((all_y - y_pred)**2)
                    ss_tot = np.sum((all_y - np.mean(all_y))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    # Calculate 95% confidence interval for the regression line
                    n = len(all_x)
                    t_val = stats.t.ppf(0.975, n-2)  # 95% CI
                    s_yx = np.sqrt(ss_res / (n-2))  # Standard error of estimate
                    
                    x_line = np.linspace(all_x.min(), all_x.max(), 100)
                    y_line = p(x_line)
                    
                    # Standard error for each predicted value
                    x_mean = np.mean(all_x)
                    se_line = s_yx * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((all_x - x_mean)**2))
                    ci_upper = y_line + t_val * se_line
                    ci_lower = y_line - t_val * se_line
                    
                    # Plot confidence interval
                    ax.fill_between(x_line, ci_lower, ci_upper, 
                                   color='darkgreen', alpha=0.1, 
                                   label='95% CI')
                    
                    # Plot combined regression line
                    ax.plot(x_line, y_line, color='darkgreen', 
                           linestyle='-', alpha=0.8, linewidth=2.5,
                           label=f'Combined: r={combined_r:.2f}, R²={r_squared:.2f}')
                    
                    # Add statistics box in corner
                    stats_text = f'p={combined_p:.2e}\nn={len(all_x)}'
                    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           verticalalignment='bottom', horizontalalignment='right',
                           fontsize=8, color='darkgreen')
                
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
                    
                    # Get valid data points
                    mask = (~well_data[log_var].isna()) & (~well_data[lab_var].isna())
                    x_data = well_data.loc[mask, log_var].values
                    y_data = well_data.loc[mask, lab_var].values
                    
                    if len(x_data) > 0:
                        # Shorter well names in legend
                        well_short = well.replace('HRDH_', '')
                        ax.scatter(x_data, y_data, 
                                 color=well_colors[well], 
                                 alpha=0.6, 
                                 s=30,
                                 label=f'{well_short} (r={r:.2f}, n={len(x_data)})',
                                 edgecolors='darkred',
                                 linewidth=0.5)
                        
                        # Add individual well regression line (thinner, dashed)
                        if len(x_data) > 1:
                            z_well = np.polyfit(x_data, y_data, 1)
                            p_well = np.poly1d(z_well)
                            x_line_well = np.linspace(x_data.min(), x_data.max(), 50)
                            ax.plot(x_line_well, p_well(x_line_well), 
                                   color=well_colors[well], 
                                   linestyle='--', 
                                   alpha=0.5, 
                                   linewidth=1)
                        
                        all_x.extend(x_data)
                        all_y.extend(y_data)
                
                # Calculate combined statistics from ALL data points
                if len(all_x) > 3:
                    all_x = np.array(all_x)
                    all_y = np.array(all_y)
                    
                    # Use the pre-calculated combined_r
                    combined_p = stats.pearsonr(all_x, all_y)[1]
                    
                    # Calculate R-squared
                    z = np.polyfit(all_x, all_y, 1)
                    p = np.poly1d(z)
                    y_pred = p(all_x)
                    ss_res = np.sum((all_y - y_pred)**2)
                    ss_tot = np.sum((all_y - np.mean(all_y))**2)
                    r_squared = 1 - (ss_res / ss_tot)
                    
                    # Calculate 95% confidence interval for the regression line
                    n = len(all_x)
                    t_val = stats.t.ppf(0.975, n-2)  # 95% CI
                    s_yx = np.sqrt(ss_res / (n-2))  # Standard error of estimate
                    
                    x_line = np.linspace(all_x.min(), all_x.max(), 100)
                    y_line = p(x_line)
                    
                    # Standard error for each predicted value
                    x_mean = np.mean(all_x)
                    se_line = s_yx * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((all_x - x_mean)**2))
                    ci_upper = y_line + t_val * se_line
                    ci_lower = y_line - t_val * se_line
                    
                    # Plot confidence interval
                    ax.fill_between(x_line, ci_lower, ci_upper, 
                                   color='darkred', alpha=0.1, 
                                   label='95% CI')
                    
                    # Plot combined regression line
                    ax.plot(x_line, y_line, color='darkred', 
                           linestyle='-', alpha=0.8, linewidth=2.5,
                           label=f'Combined: r={combined_r:.2f}, R²={r_squared:.2f}')
                    
                    # Add statistics box in corner
                    stats_text = f'p={combined_p:.2e}\nn={len(all_x)}'
                    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                           verticalalignment='bottom', horizontalalignment='right',
                           fontsize=8, color='darkred')
                
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
                pos_count = sum(1 for _, _, info in corrs if info['correlation_type'] == 'Positive')
                neg_count = sum(1 for _, _, info in corrs if info['correlation_type'] == 'Negative')
                print(f"\n{n_wells} wells: {len(corrs)} total ({pos_count} positive, {neg_count} negative)")
    
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

def create_3d_correlation_explorer(df_all, top_correlations, save_path='imgs/'):
    """
    Create interactive 3D scatter plots showing relationships between 3 variables at once.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Get top 3 correlations
    top_3 = []
    for n_wells in [4, 3, 2]:
        if n_wells in correlations_by_well_count:
            for pair, wells_data, info in correlations_by_well_count[n_wells][:2]:
                top_3.append((pair, info['avg_abs_corr']))
    
    # Find variables that appear most frequently
    var_counts = {}
    for (log_var, lab_var), _ in top_3[:6]:
        var_counts[log_var] = var_counts.get(log_var, 0) + 1
        var_counts[lab_var] = var_counts.get(lab_var, 0) + 1
    
    # Get top 3 most connected variables
    top_vars = sorted(var_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    if len(top_vars) >= 3:
        var1, var2, var3 = [v[0] for v in top_vars]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        colors = {'HRDH_697': 'blue', 'HRDH_1119': 'orange', 
                  'HRDH_1804': 'green', 'HRDH_1867': 'red'}
        
        for well in df_all['Well'].unique():
            well_data = df_all[df_all['Well'] == well]
            valid_data = well_data[[var1, var2, var3]].dropna()
            
            fig.add_trace(go.Scatter3d(
                x=valid_data[var1],
                y=valid_data[var2],
                z=valid_data[var3],
                mode='markers',
                name=well,
                marker=dict(
                    size=4,
                    color=colors.get(well, 'gray'),
                    opacity=0.6
                ),
                text=well_data.index,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              f'{var1}: %{{x:.2f}}<br>' +
                              f'{var2}: %{{y:.2f}}<br>' +
                              f'{var3}: %{{z:.2f}}<br>' +
                              'Index: %{text}'
            ))
        
        fig.update_layout(
            title=f'3D Correlation Explorer: {var1.replace("Log_", "").replace("Lab_", "")} vs ' +
                  f'{var2.replace("Log_", "").replace("Lab_", "")} vs {var3.replace("Log_", "").replace("Lab_", "")}',
            scene=dict(
                xaxis_title=var1.replace('Log_', '').replace('Lab_', ''),
                yaxis_title=var2.replace('Log_', '').replace('Lab_', ''),
                zaxis_title=var3.replace('Log_', '').replace('Lab_', ''),
            ),
            height=800
        )
        
        fig.write_html(f'{save_path}3d_correlation_explorer.html')
        fig.show()

def create_correlation_network(df_all, correlations_by_well_count, min_correlation=0.6, save_path='imgs/'):
    """
    Create an interactive network graph showing relationships between variables.
    """
    import networkx as nx
    import plotly.graph_objects as go
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes and edges
    for n_wells in [4, 3, 2]:
        if n_wells in correlations_by_well_count:
            for pair, wells_data, info in correlations_by_well_count[n_wells]:
                if info['avg_abs_corr'] >= min_correlation:
                    log_var, lab_var = pair
                    
                    # Add nodes
                    G.add_node(log_var, node_type='log')
                    G.add_node(lab_var, node_type='lab')
                    
                    # Add edge with weight based on correlation strength
                    G.add_edge(log_var, lab_var, 
                              weight=info['avg_abs_corr'],
                              n_wells=n_wells,
                              correlation_type=info['correlation_type'])
    
    # Create layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create Plotly figure
    edge_trace = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        # Color based on correlation type
        color = 'green' if edge[2]['correlation_type'] == 'Positive' else 'red'
        width = edge[2]['weight'] * 5
        
        edge_trace.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=width, color=color),
            hoverinfo='text',
            text=f"{edge[0].replace('Log_', '')} - {edge[1].replace('Lab_', '')}<br>" +
                 f"Avg |r|: {edge[2]['weight']:.3f}<br>" +
                 f"Wells: {edge[2]['n_wells']}",
            showlegend=False
        ))
    
    # Node trace
    node_trace = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode='markers+text',
        text=[node.replace('Log_', '').replace('Lab_', '') for node in G.nodes()],
        textposition="top center",
        marker=dict(
            size=15,
            color=['lightblue' if G.nodes[node]['node_type'] == 'log' else 'lightcoral' 
                   for node in G.nodes()],
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=[f"{node}<br>Type: {G.nodes[node]['node_type']}<br>" +
                   f"Connections: {G.degree(node)}" for node in G.nodes()]
    )
    
    fig = go.Figure(data=edge_trace + [node_trace])
    
    fig.update_layout(
        title='Correlation Network Graph (|r| ≥ 0.6)',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=800
    )
    
    fig.write_html(f'{save_path}correlation_network.html')
    fig.show()

def create_parallel_coordinates(df_all, selected_vars, save_path='imgs/'):
    """
    Create parallel coordinates plot to visualize multi-dimensional patterns.
    """
    from sklearn.preprocessing import StandardScaler
    import plotly.graph_objects as go
    
    # Prepare data
    plot_data = df_all[selected_vars + ['Well']].dropna()
    
    # Standardize numerical data
    scaler = StandardScaler()
    scaled_data = plot_data.copy()
    scaled_data[selected_vars] = scaler.fit_transform(plot_data[selected_vars])
    
    # Create color mapping for wells
    well_colors = {'HRDH_697': 0, 'HRDH_1119': 1, 'HRDH_1804': 2, 'HRDH_1867': 3}
    scaled_data['Well_Color'] = scaled_data['Well'].map(well_colors)
    
    # Create dimensions
    dimensions = []
    for var in selected_vars:
        dimensions.append(
            dict(
                label=var.replace('Log_', '').replace('Lab_', ''),
                values=scaled_data[var],
                range=[scaled_data[var].min(), scaled_data[var].max()]
            )
        )
    
    # Create parallel coordinates plot
    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=scaled_data['Well_Color'],
                colorscale=[[0, 'blue'], [0.33, 'orange'], [0.66, 'green'], [1, 'red']],
                showscale=True,
                colorbar=dict(
                    title='Well',
                    tickvals=[0, 1, 2, 3],
                    ticktext=['697', '1119', '1804', '1867']
                )
            ),
            dimensions=dimensions
        )
    )
    
    fig.update_layout(
        title='Parallel Coordinates: Multi-Variable Patterns Across Wells',
        height=600
    )
    
    fig.write_html(f'{save_path}parallel_coordinates.html')
    fig.show()

def create_hierarchical_clustering_heatmap(df_all, variables, save_path='imgs/'):
    """
    Create a clustered heatmap showing how samples group together.
    """
    import seaborn as sns
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    
    # Prepare data
    data_for_clustering = df_all[variables].dropna()
    well_labels = df_all.loc[data_for_clustering.index, 'Well']
    
    # Standardize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_for_clustering)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), 
                                   gridspec_kw={'width_ratios': [1, 4]})
    
    # Create color map for wells
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#ff7f0e', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#d62728'
    }
    
    # Create well color array
    colors = [well_colors[well] for well in well_labels]
    
    # Plot well indicator
    well_matrix = [[1] for _ in colors]
    # cmap_color = sns.color_palette(list(well_colors.values()))
    ax1.imshow(well_matrix, aspect='auto', cmap='RdYlGn')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('Well', fontsize=12)
    
    # Create clustered heatmap
    g = sns.clustermap(scaled_data, 
                       col_cluster=True,
                       row_cluster=True,
                       cmap='RdYlBu_r',
                       xticklabels=[v.replace('Log_', '').replace('Lab_', '') for v in variables],
                       yticklabels=False,
                       figsize=(15, 10),
                       ax=ax2)
    
    plt.suptitle('Hierarchical Clustering of Samples', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_path}hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()

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

