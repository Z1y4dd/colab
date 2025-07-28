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
# Create combined scatter plots for top correlations across all wells
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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

def find_all_correlations_by_well_count(well_correlations, min_correlation=0.5):
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


def calculate_correlations_by_well(df_all, lab_columns, log_columns):
    """Calculate correlation matrices between lab and log measurements for each well."""
    well_correlations = {}
    wells = sorted(df_all['Well'].unique())
    
    for well in wells:
        well_data = df_all[df_all['Well'] == well]
        
        # Create correlation matrix
        corr_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=float)
        
        for log_col in log_columns:
            for lab_col in lab_columns:
                # Get non-null data
                valid_data = well_data[[log_col, lab_col]].dropna()
                
                if len(valid_data) >= 3:  # Need at least 3 points for correlation
                    r, _ = pearsonr(valid_data[log_col], valid_data[lab_col])
                    corr_matrix.loc[log_col, lab_col] = r
        
        well_correlations[well] = corr_matrix
        
        # Print summary
        print(f"\n{well} correlation matrix created:")
        print(f"  Shape: {corr_matrix.shape}")
        print(f"  Non-null correlations: {corr_matrix.notna().sum().sum()}")
    
    return well_correlations



def print_categorized_correlation_summary(correlations_by_well_count, top_n=10):
    """Print a detailed summary of correlations categorized by well count."""
    print("\n" + "="*100)
    print("CORRELATION ANALYSIS - CATEGORIZED BY WELL COUNT")
    print("="*100)
    
    total_correlations = sum(len(corrs) for corrs in correlations_by_well_count.values())
    print(f"\nTotal correlation pairs found: {total_correlations}")
    
    print("\nDistribution by well count:")
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        count = len(correlations_by_well_count[n_wells])
        print(f"  {n_wells} wells: {count} correlations")
    
    # Detailed analysis for each category
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        correlations = correlations_by_well_count[n_wells]
        
        if correlations:
            print(f"\n{'='*80}")
            print(f"CORRELATIONS FOUND IN {n_wells} WELLS (Top {min(top_n, len(correlations))})")
            print(f"{'='*80}")
            
            # Count correlation types
            positive_count = sum(1 for _, _, info in correlations if info['correlation_type'] == 'Positive')
            negative_count = sum(1 for _, _, info in correlations if info['correlation_type'] == 'Negative')
            consistent_count = sum(1 for _, _, info in correlations if info['consistent_direction'])
            
            print(f"\nCorrelation types:")
            print(f"  Positive: {positive_count}")
            print(f"  Negative: {negative_count}")
            print(f"  Consistent direction: {consistent_count}")
            
            print(f"\nTop {min(top_n, len(correlations))} correlations:")
            print("-" * 100)
            print(f"{'Rank':<5} {'Log Variable':<20} {'Lab Variable':<30} {'Avg r':<10} {'|Avg r|':<10} {'Std':<8} {'Type':<10}")
            print("-" * 100)
            
            for i, (pair, wells_data, info) in enumerate(correlations[:top_n]):
                log_var, lab_var = pair
                log_name = log_var.replace('Log_', '')
                lab_name = lab_var.replace('Lab_', '')
                
                # Use avg_corr for display (actual average, not absolute)
                print(f"{i+1:<5} {log_name:<20} {lab_name:<30} "
                      f"{info['avg_corr']:>9.3f} {info['avg_abs_corr']:>9.3f} "
                      f"{info['std_corr']:>7.3f} {info['correlation_type']:<10}")
                
                # Show individual well values
                well_details = []
                for well, r in wells_data:
                    well_name = well.split('_')[-1]
                    well_details.append(f"{well_name}:{r:.3f}")
                print(f"      Wells: {', '.join(well_details)}")
                
                if not info['consistent_direction']:
                    print(f"      ⚠️  Mixed correlation signs across wells")

def find_common_correlations(well_correlations, min_correlation=0.5, min_wells=2):
    """Find correlations that appear in multiple wells (for backward compatibility)."""
    correlations_by_well_count = find_all_correlations_by_well_count(
        well_correlations, min_correlation=min_correlation
    )
    
    common_correlations = []
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        if n_wells >= min_wells:
            common_correlations.extend(correlations_by_well_count[n_wells])
    
    return common_correlations


def analyze_correlation_consistency(common_correlations, well_correlations):
    """Analyze how consistent correlations are across wells."""
    consistency_data = []
    
    for item in common_correlations[:20]:  # Top 20 pairs
        if len(item) == 3:
            pair, wells_data, info = item
        else:
            pair, wells_data = item
            
        log_var, lab_var = pair
        
        all_well_corrs = []
        for well, corr_matrix in well_correlations.items():
            if log_var in corr_matrix.index and lab_var in corr_matrix.columns:
                r = corr_matrix.loc[log_var, lab_var]
                if not pd.isna(r):
                    all_well_corrs.append(r)
        
        if len(all_well_corrs) >= 2:
            consistency_data.append({
                'Variable_Pair': f'{log_var.replace("Log_", "")} vs {lab_var.replace("Lab_", "")}',
                'Mean_r': np.mean(all_well_corrs),
                'Std_r': np.std(all_well_corrs),
                'Min_r': np.min(all_well_corrs),
                'Max_r': np.max(all_well_corrs),
                'Range_r': np.max(all_well_corrs) - np.min(all_well_corrs),
                'N_Wells': len(all_well_corrs)
            })
    
    consistency_df = pd.DataFrame(consistency_data)
    consistency_df = consistency_df.sort_values('Std_r')
    
    return consistency_df


def create_comprehensive_correlation_scatter_plots(df_all, correlations_by_well_count, max_plots_per_figure=20):
    """
    Create two comprehensive scatter plot figures: one for positive and one for negative correlations.
    Plots are sorted by correlation strength and include combined statistics.
    Also saves individual plots in organized folders.
    """
    
    # Define colors for each well
    well_colors = {
        'HRDH_697': '#1f77b4',
        'HRDH_1119': '#ff7f0e', 
        'HRDH_1804': '#2ca02c',
        'HRDH_1867': '#d62728'
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
    
    # Collect all correlations and calculate their combined correlation
    all_correlations_with_combined = []
    
    for n_wells in correlations_by_well_count.keys():
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
    
    print(f"\nCorrelation Classification Summary:")
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
        
        fig.suptitle(f'Positive Correlations (Top {n_positive} of {len(all_positive_corrs)} total)', 
                    fontsize=20, fontweight='bold', color='darkgreen', y=0.995)
        
        # Save individual positive plots
        print(f"\nSaving {len(all_positive_corrs)} individual positive correlation plots...")
        
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
        plt.savefig('imgs/scatter_all_positive_correlations.png', dpi=300, bbox_inches='tight')
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
        
        fig.suptitle(f'Negative Correlations (Top {n_negative} of {len(all_negative_corrs)} total)', 
                    fontsize=20, fontweight='bold', color='darkred', y=0.995)
        
        # Save individual negative plots
        print(f"\nSaving {len(all_negative_corrs)} individual negative correlation plots...")
        
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
        plt.savefig('imgs/scatter_all_negative_correlations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Saved {len(all_negative_corrs)} negative correlation plots to: {negative_dir}")
    
    # Create summary statistics
    print("\n" + "="*80)
    print("SCATTER PLOT GENERATION SUMMARY")
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
    print("rValue = Average absolute correlation value")
    
    print("\nCOMBINED PLOTS:")
    print(f"Total positive correlations: {len(all_positive_corrs)}")
    print(f"Total negative correlations: {len(all_negative_corrs)}")
    print(f"Total correlations visualized in combined plots: {min(len(all_positive_corrs), max_plots_per_figure) + min(len(all_negative_corrs), max_plots_per_figure)}")
    
    # Distribution by well count
    for n_wells in sorted(correlations_by_well_count.keys(), reverse=True):
        corrs = correlations_by_well_count[n_wells]
        if corrs:
            pos_count = sum(1 for _, _, info in corrs if info['correlation_type'] == 'positive')
            neg_count = sum(1 for _, _, info in corrs if info['correlation_type'] == 'negative')
            print(f"\n{n_wells} wells: {len(corrs)} total ({pos_count} positive, {neg_count} negative)")
    
    # Top correlations summary
    print("\nTop 10 Positive Correlations (by combined r):")
    for i, (pair, wells_data, info, combined_r) in enumerate(positive_correlations[:10]):
        log_var, lab_var = pair
        wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
        print(f"{i+1}. {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: " +
              f"Combined r = {combined_r:.3f} (Wells: {', '.join(wells_involved)})")
    
    print("\nTop 10 Negative Correlations (by combined r):")
    for i, (pair, wells_data, info, combined_r) in enumerate(negative_correlations[:10]):
        log_var, lab_var = pair
        wells_involved = [well.replace('HRDH_', '') for well, _ in wells_data]
        print(f"{i+1}. {log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}: " +
              f"Combined r = {combined_r:.3f} (Wells: {', '.join(wells_involved)})")

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
    
    # Also track which wells have data for each pair
    wells_with_data_matrix = pd.DataFrame(index=log_columns, columns=lab_columns, dtype=object)
    
    for log_var in log_columns:
        for lab_var in lab_columns:
            # Get combined data
            combined_data = df_all[[log_var, lab_var, 'Well']].dropna()
            
            if len(combined_data) > 5:
                r, p = pearsonr(combined_data[log_var], combined_data[lab_var])
                combined_corr_matrix.loc[log_var, lab_var] = r
                sample_size_matrix.loc[log_var, lab_var] = len(combined_data)
                
                # Get list of wells that have data for this pair
                wells_present = sorted(combined_data['Well'].unique())
                wells_with_data_matrix.loc[log_var, lab_var] = wells_present
    
    # Drop empty rows and columns
    combined_corr_matrix = combined_corr_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)
    sample_size_matrix = sample_size_matrix.loc[combined_corr_matrix.index, combined_corr_matrix.columns]
    wells_with_data_matrix = wells_with_data_matrix.loc[combined_corr_matrix.index, combined_corr_matrix.columns]
    
    # --- Figure 1: Combined Correlations with Clean Annotations ---
    fig1, ax1 = plt.subplots(figsize=(18, 14))  # Increased height for legend
    
    # Create custom annotations - more compact format with wells info
    annot_text = np.empty_like(combined_corr_matrix, dtype=object)
    for i in range(len(combined_corr_matrix.index)):
        for j in range(len(combined_corr_matrix.columns)):
            r = combined_corr_matrix.iloc[i, j]
            n = sample_size_matrix.iloc[i, j]
            wells_list = wells_with_data_matrix.iloc[i, j]
            
            if pd.notna(r) and n > 0:
                # Format well names - just the numbers
                well_numbers = [w.split('_')[1] for w in wells_list]
                n_wells = len(wells_list)
                
                # Create multi-line annotation based on number of wells
                if n_wells == 4:
                    # Don't mention well names when all 4 wells are present
                    annot_text[i, j] = f'{r:.2f}\nn={n}\n(all wells)'
                elif n_wells == 3:
                    # For 3 wells, use abbreviated format
                    annot_text[i, j] = f'{r:.2f}\nn={n}\n(3 wells)'
                elif n_wells == 2:
                    # For 2 wells, show the well numbers
                    annot_text[i, j] = f'{r:.2f}\nn={n}\n({",".join(well_numbers)})'
                else:
                    # For 1 well
                    annot_text[i, j] = f'{r:.2f}\nn={n}\n({well_numbers[0]})'
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
                  'Pearson correlation coefficient (r) with sample size (n)', 
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
    
    # Create annotations only for strong correlations - cleaner format with wells info
    strong_annot_text = np.empty_like(combined_corr_matrix, dtype=object)
    for i in range(len(combined_corr_matrix.index)):
        for j in range(len(combined_corr_matrix.columns)):
            r = combined_corr_matrix.iloc[i, j]
            n = sample_size_matrix.iloc[i, j]
            wells_list = wells_with_data_matrix.iloc[i, j]
            
            if pd.notna(r) and abs(r) >= strong_threshold and n > 0:
                # Format well names - just the numbers
                well_numbers = [w.split('_')[1] for w in wells_list]
                n_wells = len(wells_list)
                
                # Create multi-line annotation based on number of wells
                if n_wells == 4:
                    # Don't mention well names when all 4 wells are present
                    strong_annot_text[i, j] = f'{r:.2f}\nn={n}\n(all wells)'
                elif n_wells == 3:
                    # For 3 wells, use abbreviated format
                    strong_annot_text[i, j] = f'{r:.2f}\nn={n}\n(3 wells)'
                elif n_wells == 2:
                    # For 2 wells, show the well numbers
                    strong_annot_text[i, j] = f'{r:.2f}\nn={n}\n({",".join(well_numbers)})'
                else:
                    # For 1 well
                    strong_annot_text[i, j] = f'{r:.2f}\nn={n}\n({well_numbers[0]})'
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
                  f'Pearson correlation coefficient (r) with sample size (n)', 
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
    print(f"{log_var.replace('Log_', '')} vs {lab_var.replace('Lab_', '')}")
    print(f"r = {combined_corr_matrix.iloc[top_confidence_idx]:.3f}, n = {sample_size_matrix.iloc[top_confidence_idx]}")
    
    # Print log descriptions in summary
    print("\n📖 LOG MEASUREMENT DESCRIPTIONS:")
    print("-" * 60)
    for log_var in log_vars_in_plot:
        if log_var in LOG_DESCRIPTIONS:
            print(f"{log_var:<6}: {LOG_DESCRIPTIONS[log_var]}")
    
    return combined_corr_matrix, sample_size_matrix