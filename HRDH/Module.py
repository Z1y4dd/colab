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

# loading dlis files
def dlis_to_df(path, needed=None, frame_index=0, verbose=True):
    """
    Load DLIS file and return a DataFrame with well log data.
    
    Parameters:
    -----------
    path : str or Path
        Path to the DLIS file
    needed : list, optional
        List of specific channel names to extract
    frame_index : int, default 0
        Index of frame to extract (if multiple frames exist)
    verbose : bool, default True
        Whether to print detailed information
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with log data, depth as index
    """
    
    def vprint(msg):
        """Verbose print function"""
        if verbose:
            print(msg)
    
    # 1. FILE VALIDATION
    vprint(" STEP 1: FILE VALIDATION")
    vprint("=" * 40)
    
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DLIS file not found: {path}")
        
        if not path.suffix.lower() in ['.dlis', '.dls']:
            vprint(f" Warning: File extension '{path.suffix}' is not typical for DLIS files")
            
        file_size_mb = path.stat().st_size / (1024 * 1024)
        vprint(f"✅ File found: {path.name}")
        vprint(f"Size: {file_size_mb:.1f} MB")
        vprint(f"Path: {path}")
        
    except Exception as e:
        vprint(f" File validation failed: {e}")
        return pd.DataFrame()
    
    # 2. DLIS FILE LOADING
    vprint(f"\n STEP 2: DLIS FILE LOADING")
    vprint("=" * 40)
    
    try:
        # Load DLIS file
        files = dlis.load(str(path))
        vprint(f"✅ DLIS file loaded successfully")
        vprint(f"Number of logical files: {len(files)}")
        
        if len(files) == 0:
            raise ValueError("No logical files found in DLIS file")
            
        # Use first logical file
        logical_file = files[0]
        vprint(f"Using logical file: {logical_file.name if hasattr(logical_file, 'name') else 'unnamed'}")
        
    except Exception as e:
        vprint(f" DLIS loading failed: {e}")
        traceback.print_exc() if verbose else None
        return pd.DataFrame()
    
    # 3. FRAME VALIDATION
    vprint(f"\n STEP 3: FRAME VALIDATION")
    vprint("=" * 40)
    
    try:
        frames = logical_file.frames
        vprint(f"✅ Found {len(frames)} frame(s)")
        
        if len(frames) == 0:
            raise ValueError("No frames found in logical file")
            
        # List all available frames
        for i, frame in enumerate(frames):
            frame_name = frame.name if hasattr(frame, 'name') else f"Frame_{i}"
            vprint(f"Frame {i}: {frame_name}")
            
        # Validate frame index
        if frame_index >= len(frames):
            vprint(f" Frame index {frame_index} out of range, using frame 0")
            frame_index = 0
            
        selected_frame = frames[frame_index]
        frame_name = selected_frame.name if hasattr(selected_frame, 'name') else f"Frame_{frame_index}"
        vprint(f"✅ Selected frame {frame_index}: {frame_name}")
        
    except Exception as e:
        vprint(f" Frame validation failed: {e}")
        return pd.DataFrame()
    
    # 4. CHANNEL ANALYSIS
    vprint(f"\n STEP 4: CHANNEL ANALYSIS")
    vprint("=" * 40)
    
    try:
        channels = selected_frame.channels
        channel_names = [ch.name for ch in channels]
        
        # Add dictionary to store channel descriptions
        channel_descriptions = {}
        
        vprint(f"✅ Found {len(channel_names)} channels:")
        
        # Show channel details
        for i, (ch, name) in enumerate(zip(channels, channel_names)):
            dimension = getattr(ch, 'dimension', 'Unknown')
            units = getattr(ch, 'units', 'Unknown')
            
            # Extract long_name for better descriptions
            long_name = getattr(ch, 'long_name', '')
            if long_name:
                channel_descriptions[name] = long_name
                vprint(f"   {i+1:2d}. {name:<15} | {long_name:<30} | Dim: {dimension} | Units: {units}")
            else:
                vprint(f"   {i+1:2d}. {name:<15} | Dim: {dimension} | Units: {units}")
            
        # Check for depth channel
        depth_candidates = ['TDEP', 'DEPTH', 'DEPT', 'Depth', 'depth', 'MD', 'MEASURED_DEPTH']
        found_depth = None
        for candidate in depth_candidates:
            if candidate in channel_names:
                found_depth = candidate
                break
                
        if found_depth:
            vprint(f"✅ Depth channel found: {found_depth}")
        else:
            vprint(f" No standard depth channel found. Available: {channel_names[:5]}...")
        
        # Store channel descriptions in metadata to return later
        metadata = {
            'channel_descriptions': channel_descriptions,
            'channels': channels,
            'channel_names': channel_names,
            'depth_channel': found_depth
        }
            
    except Exception as e:
        vprint(f" Channel analysis failed: {e}")
        return pd.DataFrame()
    
    # 5. DATA EXTRACTION
    vprint(f"\n STEP 5: DATA EXTRACTION")
    vprint("=" * 40)
    
    try:
        # Get curves data
        curves_data = selected_frame.curves()
        vprint(f"✅ Curves data extracted")
        vprint(f"Data type: {type(curves_data)}")
        vprint(f"Shape: {curves_data.shape}")
        vprint(f"Dtype: {curves_data.dtype}")
        
        # Check if structured array
        if not (hasattr(curves_data.dtype, 'names') and curves_data.dtype.names):
            vprint(f" Data is not a structured array with named fields")
            vprint(f"Cannot extract channel data automatically")
            return pd.DataFrame()
            
        field_names = curves_data.dtype.names
        vprint(f"✅ Found {len(field_names)} data fields")
        
    except Exception as e:
        vprint(f" Data extraction failed: {e}")
        traceback.print_exc() if verbose else None
        return pd.DataFrame()
    
    # 6. FIELD TO CHANNEL MAPPING
    vprint(f"\n STEP 6: FIELD MAPPING")
    vprint("=" * 40)
    
    field_to_channel = {}
    data_dict = {}
    skipped_fields = []
    
    for field in field_names:
        try:
            # Handle complex field names
            simple_name = None
            
            # Case 1: Tuple field names like (('T.CHANNEL-I.GR-O.1-C.0', 'GR'), '<f4')
            if isinstance(field, tuple) and len(field) > 1:
                if isinstance(field[0], tuple) and len(field[0]) > 1:
                    simple_name = field[0][1]  # Extract 'GR'
                elif isinstance(field[0], str):
                    simple_name = field[0]
                    
            # Case 2: Direct string field names
            elif isinstance(field, str):
                simple_name = field
                
            # Case 3: Search in channel names
            if not simple_name:
                for ch_name in channel_names:
                    if ch_name in str(field):
                        simple_name = ch_name
                        break
                        
            if not simple_name:
                skipped_fields.append(str(field))
                continue
                
            # Extract array data
            array = curves_data[field]
            
            # Check dimensionality
            if array.ndim > 1:
                vprint(f"    Skipping multi-dimensional: {simple_name} {array.shape}")
                skipped_fields.append(f"{simple_name} (multi-dim)")
                continue
                
            # Store valid data
            field_to_channel[field] = simple_name
            data_dict[simple_name] = array
            vprint(f"   ✅ {simple_name}: {len(array)} samples")
            
        except Exception as e:
            vprint(f"    Error processing {field}: {e}")
            skipped_fields.append(f"{field} (error)")
            
    vprint(f"\n MAPPING SUMMARY:")
    vprint(f"Successfully mapped: {len(data_dict)} fields")
    vprint(f"Skipped: {len(skipped_fields)} fields")
    if skipped_fields and verbose:
        vprint(f"Skipped list: {skipped_fields[:5]}{'...' if len(skipped_fields) > 5 else ''}")
    
    # 7. DATAFRAME CREATION
    vprint(f"\n STEP 7: DATAFRAME CREATION")
    vprint("=" * 40)
    
    if not data_dict:
        vprint(f" No valid data extracted")
        return pd.DataFrame()
        
    try:
        # Check array lengths
        lengths = {name: len(arr) for name, arr in data_dict.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            vprint(f" Arrays have different lengths: {dict(list(lengths.items())[:5])}")
            min_len = min(lengths.values())
            vprint(f"Truncating all arrays to {min_len} samples")
            data_dict = {name: arr[:min_len] for name, arr in data_dict.items()}
            
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        vprint(f"✅ DataFrame created: {df.shape}")
        
        # Set depth as index
        depth_channel = found_depth if found_depth and found_depth in df.columns else None
        if depth_channel:
            df = df.set_index(depth_channel)
            vprint(f"✅ Set {depth_channel} as index")
            vprint(f"Depth range: {df.index.min():.2f} - {df.index.max():.2f}")
        else:
            vprint(f" No depth channel found in extracted data")
            
        # Filter requested columns
        if needed is not None:
            available_cols = [col for col in needed if col in df.columns]
            missing_cols = [col for col in needed if col not in df.columns]
            
            if available_cols:
                df = df[available_cols]
                vprint(f"✅ Filtered to {len(available_cols)} requested columns")
            else:
                vprint(f" None of requested columns found: {needed}")
                
            if missing_cols:
                vprint(f" Missing requested columns: {missing_cols}")
                
        vprint(f"✅ Final DataFrame: {df.shape}")
        return df
        
    except Exception as e:
        vprint(f" DataFrame creation failed: {e}")
        traceback.print_exc() if verbose else None
        return pd.DataFrame()

    """Wrapper function with additional validation"""
    try:
        df = dlis_to_df(path, **kwargs)
        
        if df.empty:
            print(" Failed to load DLIS data")
            return df
            
        print(f"\n✅ FINAL VALIDATION:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Index: {df.index.name}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f" Load and validation failed: {e}")
        return pd.DataFrame()

def load_dlis_files_from_list(
    file_paths: list[str],
    channels: list[str] | None = None,
    frame_idx: int = 0,
    concatenate: bool = True,
    remove_duplicates: bool = True,
    verbose: bool = True
):
    """
    Load multiple DLIS files from a provided list of file paths.
    
    Args:
        file_paths: List of absolute paths to DLIS files to load.
        channels: List of curve mnemonics to extract (None => all).
        frame_idx: Frame index to use for each DLIS file.
        concatenate: If True, concatenate all DataFrames into one. If False, return list of DataFrames.
        remove_duplicates: If True, remove duplicate depth indices when concatenating.
        verbose: Whether to print detailed loading information.
        
    Returns:
        If concatenate=True: (combined_df, metadata)
        If concatenate=False: (list_of_dfs, metadata)
        
    Example:
        # Load specific files
        paths = [
            "/path/to/file1.dlis",
            "/path/to/file2.dlis", 
            "/path/to/file3.dlis"
        ]
        df, meta = load_dlis_files_from_list(paths, channels=['GR', 'NPHI', 'RHOB'])
    """
    
    def vprint(msg):
        """Verbose print function"""
        if verbose:
            print(msg)
    
    vprint(f" LOADING {len(file_paths)} DLIS FILES FROM LIST")
    vprint("=" * 50)
    
    # 1. Validate input file paths
    vprint(f"\n STEP 1: VALIDATING FILE PATHS")
    vprint("-" * 30)
    
    valid_paths = []
    invalid_paths = []
    
    for i, path in enumerate(file_paths):
        try:
            path_obj = Path(path)
            if path_obj.exists():
                if path_obj.suffix.lower() in ['.dlis', '.dls']:
                    valid_paths.append(str(path_obj))
                    size_mb = path_obj.stat().st_size / (1024 * 1024)
                    vprint(f"✅ {i+1:2d}. {path_obj.name} ({size_mb:.1f} MB)")
                else:
                    invalid_paths.append((path, f"Invalid extension: {path_obj.suffix}"))
                    vprint(f" {i+1:2d}. {path_obj.name} - Invalid extension")
            else:
                invalid_paths.append((path, "File not found"))
                vprint(f" {i+1:2d}. {Path(path).name} - File not found")
        except Exception as e:
            invalid_paths.append((path, str(e)))
            vprint(f" {i+1:2d}. {Path(path).name} - Error: {e}")
    
    vprint(f"\n VALIDATION SUMMARY:")
    vprint(f"Valid files: {len(valid_paths)}")
    vprint(f"Invalid files: {len(invalid_paths)}")
    
    if len(valid_paths) == 0:
        vprint(" No valid DLIS files found!")
        empty_metadata = {
            "error": "No valid files",
            "invalid_paths": invalid_paths,
            "summary": {"files_processed": 0, "files_loaded": 0, "files_failed": len(file_paths)}
        }
        if concatenate:
            return pd.DataFrame(), empty_metadata
        else:
            return [], empty_metadata
    
    # 2. Load each valid file
    vprint(f"\n🔧 STEP 2: LOADING DLIS FILES")
    vprint("-" * 30)
    
    dataframes = []
    load_metadata = []
    failed_loads = []
    
    for i, file_path in enumerate(valid_paths):
        try:
            vprint(f"\n📁 Loading file {i+1}/{len(valid_paths)}: {Path(file_path).name}")
            
            # Load using existing dlis_to_df function
            df = dlis_to_df(
                path=file_path,
                needed=channels,
                frame_index=frame_idx,
                verbose=verbose
            )
            
            if not df.empty:
                dataframes.append(df)
                
                # Create metadata for this file
                file_meta = {
                    'path': file_path,
                    'filename': Path(file_path).name,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'depth_range': (df.index.min(), df.index.max()) if len(df) > 0 else (None, None),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2
                }
                load_metadata.append(file_meta)
                
                vprint(f"✅ Success: {df.shape[0]} samples × {df.shape[1]} channels")
                vprint(f"   Depth range: {df.index.min():.1f} - {df.index.max():.1f}")
                vprint(f"   Memory: {file_meta['memory_mb']:.1f} MB")
            else:
                failed_loads.append((file_path, "Empty DataFrame returned"))
                vprint(f" Empty DataFrame returned")
                
        except Exception as e:
            failed_loads.append((file_path, str(e)))
            vprint(f" Loading failed: {e}")
            
            # Print detailed error for debugging
            if verbose:
                import traceback
                vprint(f"   Full error: {traceback.format_exc()}")
    
    vprint(f"\n LOADING SUMMARY:")
    vprint(f"Files processed: {len(valid_paths)}")
    vprint(f"Successfully loaded: {len(dataframes)}")
    vprint(f"Failed to load: {len(failed_loads)}")
    
    if len(dataframes) == 0:
        vprint(" No files were successfully loaded!")
        empty_metadata = {
            "error": "No files loaded successfully",
            "failed_loads": failed_loads,
            "invalid_paths": invalid_paths,
            "summary": {"files_processed": len(valid_paths), "files_loaded": 0, "files_failed": len(failed_loads)}
        }
        if concatenate:
            return pd.DataFrame(), empty_metadata
        else:
            return [], empty_metadata
    
    # 3. Concatenate if requested
    if concatenate:
        vprint(f"\n🔗 STEP 3: CONCATENATING DATAFRAMES")
        vprint("-" * 30)
        
        try:
            # Show column information before concatenation
            all_columns = set()
            for df in dataframes:
                all_columns.update(df.columns)
            vprint(f"Unique columns across all files: {len(all_columns)}")
            
            # Concatenate all DataFrames
            combined_df = pd.concat(dataframes, sort=True).sort_index()
            vprint(f"Combined shape before deduplication: {combined_df.shape}")
            
            # Remove duplicates if requested
            if remove_duplicates:
                duplicates_before = combined_df.index.duplicated().sum()
                if duplicates_before > 0:
                    vprint(f"Found {duplicates_before} duplicate depths")
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    vprint(f"Removed duplicates, new shape: {combined_df.shape}")
                else:
                    vprint(f"No duplicate depths found")
            
            # Final statistics
            total_memory = combined_df.memory_usage(deep=True).sum() / 1024**2
            vprint(f"\n✅ CONCATENATION COMPLETE:")
            vprint(f"Final shape: {combined_df.shape}")
            vprint(f"Depth range: {combined_df.index.min():.1f} - {combined_df.index.max():.1f}")
            vprint(f"Total memory: {total_memory:.1f} MB")
            vprint(f"Columns: {list(combined_df.columns)}")
            
            result_df = combined_df
            
        except Exception as e:
            vprint(f" Concatenation failed: {e}")
            result_df = pd.DataFrame()
    else:
        vprint(f"\n📦 Returning {len(dataframes)} separate DataFrames (concatenate=False)")
        result_df = dataframes
    
    # 4. Compile comprehensive metadata
    metadata = {
        "input_paths": file_paths,
        "valid_paths": valid_paths,
        "invalid_paths": invalid_paths,
        "failed_loads": failed_loads,
        "load_metadata": load_metadata,
        "concatenated": concatenate,
        "duplicates_removed": remove_duplicates if concatenate else None,
        "summary": {
            "total_input_files": len(file_paths),
            "valid_files": len(valid_paths),
            "invalid_files": len(invalid_paths),
            "files_loaded": len(dataframes),
            "files_failed": len(failed_loads),
            "final_shape": result_df.shape if concatenate and isinstance(result_df, pd.DataFrame) and not result_df.empty else None,
            "total_memory_mb": sum(meta['memory_mb'] for meta in load_metadata)
        }
    }
    
    # Add channel information if available
    if load_metadata:
        all_channels = set()
        for meta in load_metadata:
            all_channels.update(meta['columns'])
        metadata['all_channels_found'] = sorted(list(all_channels))
    
    return result_df, metadata
# joining depth
def match_lab_to_log(log_df, lab_df, tol=0.1):
    """
    For each lab depth, find the nearest log depth within `tol`.
    Returns at most len(lab_df) matched pairs.
    Adds columns for Distance and Match_Type.
    """
    print(f"\nFUNCTION START - INPUT VALIDATION:")
    print(f"Log DataFrame shape: {log_df.shape}")
    print(f"Lab DataFrame shape: {lab_df.shape}")
    
    # Ensure unique indices
    log = log_df[~log_df.index.duplicated(keep='first')]
    lab = lab_df[~lab_df.index.duplicated(keep='first')]
    
    print(f"After deduplication - Log: {len(log)}, Lab: {len(lab)}")
    
    if len(log) == 0 or len(lab) == 0:
        print(" ERROR: Empty datasets after deduplication!")
        return pd.DataFrame()
    
    # Convert to float64 arrays
    log_depths = np.array(log.index.values, dtype=np.float64).reshape(-1, 1)
    lab_depths = np.array(lab.index.values, dtype=np.float64).reshape(-1, 1)
    
    # print(f"Log depths sample: {log_depths[:3].flatten()}")
    # print(f"Lab depths sample: {lab_depths[:3].flatten()}")
    
    # Build KD-Tree on log depths
    tree = cKDTree(log_depths)
    
    # Query each lab depth
    dists, idxs = tree.query(lab_depths, distance_upper_bound=tol)
    
    # FIXED: Check for finite distances (successful matches)
    mask = np.isfinite(dists)
    
    print(f"Lab depths range: {lab_depths.min():.2f} - {lab_depths.max():.2f}")
    print(f"Log depths range: {log_depths.min():.2f} - {log_depths.max():.2f}")
    print(f"Tolerance: {tol} ft")
    print(f"Valid matches found: {mask.sum()}/{len(mask)}")
    
    if mask.sum() > 0:
        print(f"Min distance: {dists[mask].min():.2f} ft")
        print(f"Max distance: {dists[mask].max():.2f} ft")
        
        # DETAILED MATCH VERIFICATION
        print(f"\nMATCH VERIFICATION:")
        for i in range(min(24, mask.sum())):
            match_idx = np.where(mask)[0][i]
            lab_depth = lab_depths[match_idx][0]
            log_idx = idxs[match_idx]
            log_depth = log_depths[log_idx][0]
            distance = dists[match_idx]
            print(f"   Match {i+1}: Lab {lab_depth:.2f} → Log {log_depth:.2f} (Δ{distance:.2f} ft)")
    
    if mask.sum() == 0:
        print(" No matches found within tolerance!")
        # Try with larger tolerance for diagnosis
        dists_large, idxs_large = tree.query(lab_depths, distance_upper_bound=1.0)
        mask_large = np.isfinite(dists_large)
        
        if mask_large.sum() > 0:
            print(f"\nCLOSEST MATCHES (within 1.0 ft):")
            for i in range(min(5, mask_large.sum())):
                match_idx = np.where(mask_large)[0][i]
                lab_depth = lab_depths[match_idx][0]
                log_idx = idxs_large[match_idx]
                log_depth = log_depths[log_idx][0]
                distance = dists_large[match_idx]
                print(f"   Lab {lab_depth:.2f} → Log {log_depth:.2f} (Δ{distance:.2f} ft)")
        
        return pd.DataFrame()
    
    # Get matched samples - FIXED: Only include valid matches
    matched_lab_indices = np.where(mask)[0]
    matched_log_indices = idxs[mask]
    
    matched_lab = lab.iloc[matched_lab_indices].reset_index()
    matched_log = log.iloc[matched_log_indices].reset_index()
    
    print(f"Matched samples extracted: {len(matched_lab)} pairs")
    
    # Rename depth columns and add prefixes
    matched_lab = matched_lab.rename(columns={matched_lab.columns[0]: 'Lab_Depth'})
    matched_log = matched_log.rename(columns={matched_log.columns[0]: 'Log_Depth'})
    
    # Add prefixes to avoid column name conflicts
    lab_cols = ['Lab_Depth'] + [f'Lab_{col}' for col in matched_lab.columns[1:]]
    log_cols = ['Log_Depth'] + [f'Log_{col}' for col in matched_log.columns[1:]]
    
    matched_lab.columns = lab_cols
    matched_log.columns = log_cols

    # Concatenate side-by-side
    joined_df = pd.concat([matched_lab, matched_log], axis=1)
    
    # Add distance column - CALCULATE ACTUAL DISTANCES
    joined_df['Distance'] = np.abs(joined_df['Lab_Depth'] - joined_df['Log_Depth'])
    
    # Add match type column
    joined_df['Match_Type'] = np.where(joined_df['Distance'] == 0, 'Exact', 'Near')
    
    # VERIFICATION: Show actual matches
    # print(f"\n✅ FINAL VERIFICATION - First 5 matches:")
    # for i in range(min(5, len(joined_df))):
    #     row = joined_df.iloc[i]
    #     print(f"   Lab: {row['Lab_Depth']:.2f} → Log: {row['Log_Depth']:.2f} (Δ{row['Distance']:.2f} ft)")
    
    return joined_df

# summary
def create_log_summary(log_df):
    """Generate a comprehensive summary of well log data"""
    import numpy as np
    import pandas as pd
    
    # Basic statistics
    total_samples = len(log_df)
    total_curves = len(log_df.columns)
    depth_min = log_df.index.min()
    depth_max = log_df.index.max()
    depth_span = depth_max - depth_min
    
    # Calculate missing data percentages
    completeness = (1 - log_df.isna().mean()) * 100
    avg_completeness = completeness.mean()
    min_completeness = completeness.min()
    min_complete_curve = log_df.columns[completeness.argmin()]
    max_completeness = completeness.max()
    max_complete_curve = log_df.columns[completeness.argmax()]
    
    # Analyze depth sampling
    depth_intervals = np.diff(log_df.index)
    avg_interval = depth_intervals.mean()
    min_interval = depth_intervals.min()
    max_interval = depth_intervals.max()
    
    
    # Print summary report
    print("\n COMPREHENSIVE LOG DATA SUMMARY")
    print("=" * 50)
    
    print("\n DEPTH COVERAGE:")
    print(f"Range: {depth_min:.1f} - {depth_max:.1f} ft")
    print(f"Total span: {depth_span:.1f} ft")
    
    print("\n DATA STATISTICS:")
    print(f"Total samples: {total_samples:,} depth points")
    print(f"Total curves: {total_curves} measurement channels")
    print(f"Data density: {total_samples/depth_span:.1f} samples/ft")
    

    
    print("\n✅ DATA QUALITY:")
    print(f"Overall completeness: {avg_completeness:.1f}%")
    print(f"Best curve: {max_complete_curve} ({max_completeness:.1f}% complete)")
    print(f"Worst curve: {min_complete_curve} ({min_completeness:.1f}% complete)")
    
    print("\n DEPTH SAMPLING:")
    print(f"Average interval: {avg_interval:.3f} ft")
    print(f"Min interval: {min_interval:.3f} ft")
    print(f"Max interval: {max_interval:.3f} ft")
    
    return {
        "depth_range": (depth_min, depth_max),
        "total_samples": total_samples,
        "total_curves": total_curves,
        "avg_completeness": avg_completeness,
        "depth_sampling": avg_interval
    }
# visuals

def visualize_match_quality(joined, well_name):
        """
        Visualize the quality of matches between lab and log data.
        Plots a histogram of match distances and a pie chart of match types.
        """
        plt.figure(figsize=(12, 6))

        # Left plot: Distance histogram
        plt.subplot(1, 2, 1)
        sns.histplot(joined['Distance'], bins=20, kde=True)
        plt.xlabel('Match Distance (ft)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Match Distances')
        plt.grid(True)

        # Right plot: Match types
        plt.subplot(1, 2, 2)
        match_counts = joined['Match_Type'].value_counts()
        plt.pie(match_counts, labels=match_counts.index, autopct='%1.1f%%', 
                        colors=['#66b3ff', '#99ff99'], startangle=90)
        plt.axis('equal')
        plt.title('Match Type Distribution')

        plt.tight_layout()
        plt.savefig(f'imgs/{well_name}_match_quality.png', dpi=300, bbox_inches='tight')
        plt.show()

def create_mineral_composition_bar(data, mineral_columns, well_name):
    """Create stacked bar chart of mineral composition by depth"""
    
    # Sort by depth for proper stratigraphic order (deepest at bottom)
    plot_data = data.sort_values('Lab_Depth', ascending=False)
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 14))
    
    # Filter to only include specified mineral columns and ensure they sum to 100%
    minerals_data = plot_data[mineral_columns].copy()
    
    # Normalize each row to sum to 100%
    row_sums = minerals_data.sum(axis=1)
    normalized_data = minerals_data.div(row_sums, axis=0) * 100
    
    # Create stacked horizontal bar chart
    normalized_data.plot(kind='barh', stacked=True, ax=ax, width=0.8,
                         colormap='tab20')
    
    # Add depth labels on y-axis
    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels([f"{d:.1f}" for d in plot_data['Lab_Depth']])
    
    # Customize plot
    ax.set_xlabel('Mineral Composition (%)')
    ax.set_ylabel('Depth (ft)')
    ax.set_title('Mineral Composition vs Depth', fontsize=16)
    plt.legend(title='Minerals', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    # plt.savefig('mineral_composition_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'imgs/{well_name}_mineral_composition_vs_depth.png', dpi=300, bbox_inches='tight')

    plt.show()
    
    return fig

    """Create a grid of scatterplots comparing log measurements to mineral content"""
    
    # Find all combinations with correlation coefficients
    correlations = []
    for log_var in log_vars:
        for mineral_var in mineral_vars:
            if log_var in data.columns and mineral_var in data.columns:
                valid_data = data[[log_var, mineral_var]].dropna()
                if len(valid_data) >= 5:  # Need at least 5 points for meaningful correlation
                    corr = valid_data.corr().iloc[0, 1]
                    correlations.append((log_var, mineral_var, corr, len(valid_data)))
    
    # Sort by absolute correlation (strongest first)
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Select top correlations (up to 9 plots)
    top_correlations = correlations[:9]
    
    if not top_correlations:
        print(" No valid correlations found with sufficient data points")
        return None
    
    # Determine grid size
    n_plots = len(top_correlations)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # Create scatterplots
    for i, (log_var, mineral_var, corr, n) in enumerate(top_correlations):
        if i < len(axes):
            # Clean and prepare data
            plot_data = data[[log_var, mineral_var, 'Distance']].dropna()
            
            # Color by match distance quality
            scatter = axes[i].scatter(plot_data[log_var], plot_data[mineral_var], 
                          c=plot_data['Distance'], cmap='viridis_r',
                          alpha=0.8, s=50, edgecolor='k', linewidth=0.5)
            
            # Add trend line
            sns.regplot(x=log_var, y=mineral_var, data=plot_data, 
                        scatter=False, ax=axes[i], color='red', line_kws={'linewidth': 2})
            
            # Add correlation text
            axes[i].text(0.05, 0.95, f'r = {corr:.2f}\nn = {n}', 
                      transform=axes[i].transAxes, fontsize=12,
                      verticalalignment='top', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Customize plot
            axes[i].set_title(f'{log_var.replace("Log_", "")} vs {mineral_var.replace("Lab_", "")}')
            axes[i].set_xlabel(log_var)
            axes[i].set_ylabel(mineral_var)
            axes[i].grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    # Add colorbar for match distance
    if n_plots > 0:
        cbar = fig.colorbar(scatter, ax=axes, pad=0.01)
        cbar.set_label('Match Distance (ft)')
    
    plt.tight_layout()
    plt.savefig('imgs/log_vs_mineral_scatterplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_depth_trend_plots(data, variables, well_name, n_cols=3):
    """Create depth trend plots for selected variables"""
    
    n_vars = len(variables)
    n_rows = (n_vars + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(1, n_cols, figsize=(16, 8), sharey=True)
    if n_cols == 1:
        axes = [axes]
    
    # Sort by depth for proper stratigraphic order (deepest at bottom)
    plot_data = data.sort_values('Lab_Depth')
    
    # Common y-axis (depth)
    depth = plot_data['Lab_Depth']
    
    # Color palette
    colors = plt.cm.tab10.colors
    
    for i, var in enumerate(variables[:n_cols]):
        if var in plot_data.columns:
            # Get clean data
            var_data = plot_data[var].values
            
            # Create line plot
            axes[i].plot(var_data, depth, 'o-', color=colors[i % len(colors)], 
                        markersize=6, linewidth=2, alpha=0.7)
            
            # Add horizontal grid lines
            axes[i].grid(True, axis='y', linestyle='--', alpha=0.7)
            
            # Add labels
            axes[i].set_title(var.replace('Lab_', '').replace('Log_', ''), fontsize=12)
            axes[i].set_xlabel(var, fontsize=10)
            
            # Invert y-axis for depth (top at top)
            axes[i].invert_yaxis()
    
    # Set common y label only on leftmost plot
    axes[0].set_ylabel('Depth (ft)', fontsize=12)
    
    # Add main title
    fig.suptitle('Depth Trends of Key Variables', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'imgs/{well_name}_depth_trend_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_composite_log_plot(data, log_vars, well_name, label_cols=None):
    """Create a composite log plot with multiple tracks"""
    
    # Sort by depth
    plot_data = data.sort_values('Lab_Depth')
    
    # Set up tracks (columns of plots)
    n_tracks = min(4, len(log_vars) // 2 + 1)  # At most 4 tracks
    
    # Create figure
    fig, axes = plt.subplots(1, n_tracks, figsize=(3*n_tracks, 10), sharey=True)
    if n_tracks == 1:
        axes = [axes]
    
    # Depth data
    depth = plot_data['Lab_Depth']
    
    # Customize each track
    # Track 1: Basic measurements (GR, density)
    basic_logs = [var for var in log_vars if any(x in var for x in ['GR', 'ZDEN', 'PE'])]
    if basic_logs and len(basic_logs) > 0:
        ax = axes[0]
        
        # Multiple curves on same track with different colors
        for i, log in enumerate(basic_logs[:3]):  # Limit to 3 curves per track
            if log in plot_data.columns:
                color = ['green', 'red', 'blue'][i % 3]
                ax.plot(plot_data[log], depth, label=log.replace('Log_', ''), 
                       color=color, linewidth=2)
        
        ax.set_title('Basic Logs', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_yaxis()  # Depth increases downward
    
    # Track 2: Spectral data
    spectral_logs = [var for var in log_vars if any(x in var for x in ['K', 'U', 'TH'])]
    if len(spectral_logs) > 0 and len(axes) > 1:
        ax = axes[1]
        
        for i, log in enumerate(spectral_logs[:3]):
            if log in plot_data.columns:
                color = ['purple', 'brown', 'orange'][i % 3]
                ax.plot(plot_data[log], depth, label=log.replace('Log_', ''), 
                       color=color, linewidth=2)
        
        ax.set_title('Spectral Data', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Track 3: Lab data if available
    mineral_vars = [var for var in data.columns if 'XRD' in var][:3]
    if len(mineral_vars) > 0 and len(axes) > 2:
        ax = axes[2]
        
        for i, mineral in enumerate(mineral_vars):
            if mineral in plot_data.columns:
                color = ['darkgreen', 'navy', 'maroon'][i % 3]
                ax.plot(plot_data[mineral], depth, label=mineral.replace('Lab_', ''), 
                       color=color, linewidth=2)
        
        ax.set_title('Minerals', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Additional track if needed
    other_logs = [var for var in log_vars if not any(x in var for x in ['GR', 'ZDEN', 'PE', 'K', 'U', 'TH'])][:3]
    if len(other_logs) > 0 and len(axes) > 3:
        ax = axes[3]
        
        for i, log in enumerate(other_logs):
            if log in plot_data.columns:
                color = ['teal', 'magenta', 'gold'][i % 3]
                ax.plot(plot_data[log], depth, label=log.replace('Log_', ''), 
                       color=color, linewidth=2)
        
        ax.set_title('Other Logs', fontsize=12)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
    
    # Set common y label only on leftmost plot
    axes[0].set_ylabel('Depth (ft)', fontsize=14)
    
    # Add labels if provided
    if label_cols:
        # Find label positions at regular intervals
        label_positions = np.linspace(depth.min(), depth.max(), 10)
        
        for label_col in label_cols:
            if label_col in plot_data.columns:
                for pos in label_positions:
                    # Find nearest sample
                    idx = (plot_data['Lab_Depth'] - pos).abs().idxmin()
                    label_text = f"{plot_data.loc[idx, label_col]:.1f}"
                    
                    # Add label to all tracks
                    for ax in axes:
                        ax.text(0.95, pos, label_text, transform=ax.get_yaxis_transform(),
                               verticalalignment='center', horizontalalignment='right',
                               fontsize=8, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # Add title
    fig.suptitle('Composite Log', fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig(f'imgs/{well_name}_composite_log_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig
# z-score
def calculate_zscore(data, columns=None):
    """Calculate z-scores for specified columns
    
    Args:
        data (DataFrame): Input data
        columns (list, optional): Columns to calculate z-scores for. If None, use all numeric columns.
    
    Returns:
        DataFrame: DataFrame with z-scores
    """
    # Select columns to process
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns
    
    # Create new DataFrame for z-scores
    z_data = pd.DataFrame(index=data.index)
    
    # Calculate z-score for each column
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        
        # Avoid division by zero
        if std > 0:
            z_data[f"{col}_z"] = (data[col] - mean) / std
        else:
            z_data[f"{col}_z"] = np.zeros(len(data))
            
    return z_data

    """Create heatmap of z-scores to identify patterns of outliers"""
    
    # Create a mask for extreme values
    z_subset = z_data[[f"{col}_z" for col in vars_list]]
    # Clip extreme values for better visualization
    z_subset_clipped = z_subset.clip(lower=-5, upper=5)
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create heatmap
    sns.heatmap(z_subset_clipped, 
                cmap='RdYlGn',
                center=0, 
                vmin=-threshold, 
                vmax=threshold,
                yticklabels=z_subset.index)
    
    # Add labels and title
    plt.title('Z-Score Heatmap (Red = Positive Outliers, Blue = Negative Outliers)', fontsize=16)
    plt.xlabel('Variables', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    
    # Better formatting for x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('imgs/zscore_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def enhanced_zscore_correlation_analysis(data, log_vars, lab_vars, significance_level=0.05):
    """Enhanced correlation with multiple statistics and significance testing for z-scores"""
    
    # Initialize results dictionary
    results = {
        'pearson_r': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'pearson_p': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'spearman_r': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'spearman_p': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'n_samples': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=int)
    }
    
    print("Z-SCORE ENHANCED CORRELATION ANALYSIS:")
    print("=" * 40)
    
    significant_correlations = []
    
    for log_var in log_vars:
        for lab_var in lab_vars:
            # Get clean data pairs
            clean_data = data[[log_var, lab_var]].dropna()
            n = len(clean_data)
            
            if n < 3:  # Need minimum 3 points for correlation
                continue
                
            try:
                # Calculate Pearson correlation
                pearson_r, pearson_p = pearsonr(clean_data[log_var], clean_data[lab_var])
                
                # Calculate Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(clean_data[log_var], clean_data[lab_var])
                
                # Store results
                results['pearson_r'].loc[log_var, lab_var] = pearson_r
                results['pearson_p'].loc[log_var, lab_var] = pearson_p
                results['spearman_r'].loc[log_var, lab_var] = spearman_r
                results['spearman_p'].loc[log_var, lab_var] = spearman_p
                results['n_samples'].loc[log_var, lab_var] = n
                
                # Check for statistical significance and strong correlation
                if pearson_p <= significance_level and abs(pearson_r) >= 0.6:
                    significant_correlations.append({
                        'log_var': log_var,
                        'lab_var': lab_var,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'n_samples': n
                    })
                    
            except Exception as e:
                print(f"Error calculating correlation for {log_var} vs {lab_var}: {e}")
                continue
    
    # Display significant correlations
    print(f"\nSIGNIFICANT STRONG Z-SCORE CORRELATIONS (|r| ≥ 0.6, p ≤ {significance_level}):")
    if significant_correlations:
        for corr in sorted(significant_correlations, key=lambda x: abs(x['pearson_r']), reverse=True):
            print(f"{corr['log_var']} ↔ {corr['lab_var']}: "
                  f"r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.3f}, n={corr['n_samples']})")
    else:
        print("No significant strong correlations found")
    
    # Summary statistics
    all_r = results['pearson_r'].values.flatten()
    all_r = all_r[~pd.isna(all_r)]
    
    print(f"\nZ-SCORE CORRELATION SUMMARY:")
    print(f"Total correlations calculated: {len(all_r)}")
    print(f"Mean |r|: {np.abs(all_r).mean():.3f}")
    print(f"Strong correlations (|r| ≥ 0.6): {(np.abs(all_r) >= 0.6).sum()}")
    print(f"Moderate correlations (0.3 ≤ |r| < 0.6): {((np.abs(all_r) >= 0.3) & (np.abs(all_r) < 0.6)).sum()}")
    
    return results

def create_zscore_enhanced_heatmap(correlation_results, well_name, significance_level=0.05):
    """Create correlation heatmap with significance masking for z-scores"""
    
    # Get correlation matrix and p-values
    corr_matrix = correlation_results['pearson_r']
    p_matrix = correlation_results['pearson_p']
    n_matrix = correlation_results['n_samples']
    
    # Create significance mask
    significance_mask = (p_matrix > significance_level) | pd.isna(p_matrix)
    
    # Create figure with subplots
    fig, ax1 = plt.subplots(figsize=(24,24))
    
    # Add more space on the left side of the plot for y-labels
    plt.subplots_adjust(left=0.2)
    
    # Plot: Only significant correlations
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlGn', 
                vmin=-1, vmax=1,
                linewidths=0.5,
                fmt='.2f',
                annot_kws={'size': 20},
                mask=significance_mask,  # Mask non-significant
                ax=ax1,
                cbar_kws={"shrink": 0.8})
    
    ax1.set_title(f'Significant Z-Score Correlations Only (p ≤ {significance_level})', fontsize=25)
    ax1.set_xlabel('Lab Measurements (Z-Scores)', fontsize=25)
    ax1.set_ylabel('Log Measurements (Z-Scores)', fontsize=25)
    
    # Fix axis tick label formatting
    plt.sca(ax1)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20, rotation=0)  # Horizontal labels
    
    # Apply tight_layout AFTER subplots_adjust to prevent overrides
    plt.tight_layout()
    
    # Move the y-label to be more visible
    plt.gcf().axes[0].yaxis.set_label_coords(-0.15, 0.5)

    plt.savefig(f'imgs/{well_name}_zscore_enhanced_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary of significant correlations
    significant_count = (~significance_mask).sum().sum()
    total_count = (~pd.isna(corr_matrix)).sum().sum()
    
    print(f"\nZ-SCORE CORRELATION HEATMAP SUMMARY:")
    print(f"Total correlations: {total_count}")
    print(f"Significant correlations: {significant_count}")
    print(f"Significance rate: {(significant_count/total_count)*100:.1f}%")
    
    # Find and display strongest correlations
    strong_correlations = []
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            r = corr_matrix.loc[i, j]
            p = p_matrix.loc[i, j]
            if not pd.isna(r) and not pd.isna(p) and abs(r) >= 0.6 and p <= significance_level:
                strong_correlations.append((i, j, r, p))
    
    if strong_correlations:
        print(f"\nSTRONGEST SIGNIFICANT Z-SCORE CORRELATIONS:")
        for log_var, lab_var, r, p in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
            print(f"{log_var} ↔ {lab_var}: r={r:.3f} (p={p:.4f})")
    else:
        print(f"\nNo strong significant z-score correlations found (|r| ≥ 0.6, p ≤ {significance_level})")

def create_enhanced_correlation_heatmap(correlation_results, well_name, significance_level=0.05):
    """Create correlation heatmap with significance masking"""
    
    # Get correlation matrix and p-values
    corr_matrix = correlation_results['pearson_r']
    p_matrix = correlation_results['pearson_p']
    n_matrix = correlation_results['n_samples']
    
    # Create significance mask
    significance_mask = (p_matrix > significance_level) | pd.isna(p_matrix)
    
    # Reset matplotlib figure to ensure clean slate
    plt.clf()
    plt.close('all')
    
    # Create a new figure
    plt.figure(figsize=(24, 24))
    
    # Create the heatmap directly on the current figure
    heatmap = sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdYlGn', 
                vmin=-1, vmax=1,
                linewidths=0.5,
                fmt='.2f',
                annot_kws={'size': 20},
                mask=significance_mask,  # Mask non-significant
                cbar_kws={"shrink": 0.8})
    
    # Set titles and labels
    plt.title(f'Significant Correlations Only (p ≤ {significance_level})', fontsize=25)
    plt.xlabel('Lab Measurements', fontsize=25)
    plt.ylabel('Log Measurements', fontsize=25)
    
    # Adjust x and y tick labels
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20, rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and display
    plt.savefig(f'imgs/{well_name}_enhanced_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary of significant correlations
    significant_count = (~significance_mask).sum().sum()
    total_count = (~pd.isna(corr_matrix)).sum().sum()
    
    print(f"\nCORRELATION HEATMAP SUMMARY:")
    print(f"Total correlations: {total_count}")
    print(f"Significant correlations: {significant_count}")
    print(f"Significance rate: {(significant_count/total_count)*100:.1f}%")
    
    # Find and display strongest correlations
    strong_correlations = []
    for i in corr_matrix.index:
        for j in corr_matrix.columns:
            r = corr_matrix.loc[i, j]
            p = p_matrix.loc[i, j]
            if not pd.isna(r) and not pd.isna(p) and abs(r) >= 0.6 and p <= significance_level:
                strong_correlations.append((i, j, r, p))
    
    if strong_correlations:
        print(f"\nSTRONGEST SIGNIFICANT CORRELATIONS:")
        for log_var, lab_var, r, p in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
            print(f"{log_var} ↔ {lab_var}: r={r:.3f} (p={p:.4f})")
    else:
        print(f"\nNo strong significant correlations found (|r| ≥ 0.6, p ≤ {significance_level})")

def create_correlation_figure(data, correlations, correlation_type, significance_level=0.05):
    
    """Create scatter plots with regression lines for specified correlations"""
    if not correlations:
        return None
    
    # Calculate number of rows and columns for subplots
    n_corrs = len(correlations)
    n_cols = min(3, n_corrs)  # Maximum 3 columns
    n_rows = (n_corrs + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    
    # Make axes iterable even for single subplot
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Set custom colors for positive and negative correlations
    if correlation_type == "Positive":
        color = "#00FF7F"  
        title_prefix = "Positive"
    else:
        color = "#FF0000"  
        title_prefix = "Negative"
    
    # Find original data columns if we're using z-scores
    # This will map z-score column names back to original data columns
    original_data = joined  # Use the original joined dataframe
    
    # Create scatter plots
    for i, (log_var, lab_var, r, p, n) in enumerate(correlations):
        if i < len(axes):
            ax = axes[i]
            
            # Use original variable names (remove z-score indicators if present)
            orig_log_var = log_var
            orig_lab_var = lab_var
            
            # Clean data for this pair
            pair_data = original_data[[orig_log_var, orig_lab_var]].dropna()
            
            # Scatter plot with original data
            ax.scatter(pair_data[orig_log_var], pair_data[orig_lab_var], 
                      alpha=0.7, s=50, edgecolor='k', linewidth=0.5,
                      color=color)
            
            # Add regression line using original data
            sns.regplot(x=orig_log_var, y=orig_lab_var, data=pair_data, 
                      scatter=False, ax=ax, color=color, 
                      line_kws={'linewidth': 2})
            
            # Add correlation statistics
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}\nn = {n}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Labels and title - use more readable names
            ax.set_xlabel(orig_log_var.replace('Log_', ''), fontsize=12)
            ax.set_ylabel(orig_lab_var.replace('Lab_', ''), fontsize=12)
            ax.set_title(f'{orig_log_var.replace("Log_", "")} vs {orig_lab_var.replace("Lab_", "")}', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_corrs, len(axes)):
        axes[i].set_visible(False)
    
    # Add main title
    plt.suptitle(f'{title_prefix} Significant Correlations (p ≤ {significance_level})', 
                fontsize=16, y=1.02)
    
    # Save the figure
    plt.tight_layout()
    filename = f'imgs/{correlation_type.lower()}_correlations.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_significant_correlations(data, correlation_results, significance_level=0.05, min_correlation=0.6):
    """Visualize significant correlations with scatter plots"""
    
    # Get correlation matrix and p-values
    corr_matrix = correlation_results['pearson_r']
    p_matrix = correlation_results['pearson_p']
    n_matrix = correlation_results['n_samples']
    
    # Find significant positive correlations
    positive_correlations = []
    for log_var in corr_matrix.index:
        for lab_var in corr_matrix.columns:
            r = corr_matrix.loc[log_var, lab_var]
            p = p_matrix.loc[log_var, lab_var]
            n = n_matrix.loc[log_var, lab_var]
            
            if not pd.isna(r) and not pd.isna(p) and r >= min_correlation and p <= significance_level:
                positive_correlations.append((log_var, lab_var, r, p, n))
    
    # Find significant negative correlations
    negative_correlations = []
    for log_var in corr_matrix.index:
        for lab_var in corr_matrix.columns:
            r = corr_matrix.loc[log_var, lab_var]
            p = p_matrix.loc[log_var, lab_var]
            n = n_matrix.loc[log_var, lab_var]
            
            if not pd.isna(r) and not pd.isna(p) and r <= -min_correlation and p <= significance_level:
                negative_correlations.append((log_var, lab_var, r, p, n))
    
    # Sort correlations by strength (absolute value)
    positive_correlations.sort(key=lambda x: x[2], reverse=True)  # r is at index 2
    negative_correlations.sort(key=lambda x: x[2])  # Sort negative from strongest (most negative) to weakest
    
    # Create visualizations for positive correlations
    if positive_correlations:
        print(f"\n🟢POSITIVE SIGNIFICANT CORRELATIONS (r ≥ {min_correlation}, p ≤ {significance_level}):")
        for log_var, lab_var, r, p, n in positive_correlations:
            print(f"{log_var} ↔ {lab_var}: r = {r:.3f} (p = {p:.4f}, n = {n})")
        
        pos_fig = create_correlation_figure(data, positive_correlations, "Positive", significance_level)
    else:
        print(f"\n No significant positive correlations found (r ≥ {min_correlation}, p ≤ {significance_level})")
    
    # Create visualizations for negative correlations
    if negative_correlations:
        print(f"\n🔴NEGATIVE SIGNIFICANT CORRELATIONS (r ≤ -{min_correlation}, p ≤ {significance_level}):")
        for log_var, lab_var, r, p, n in negative_correlations:
            print(f"{log_var} ↔ {lab_var}: r = {r:.3f} (p = {p:.4f}, n = {n})")
            
        neg_fig = create_correlation_figure(data, negative_correlations, "Negative", significance_level)
    else:
        print(f"\n No significant negative correlations found (r ≤ -{min_correlation}, p ≤ {significance_level})")
    
    return {
        "positive_correlations": positive_correlations,
        "negative_correlations": negative_correlations
    }

def create_distribution_grid(data, variables, ncols=4, figsize_per_plot=(4, 3), kde=True, bins=30, 
                             color_map='viridis', highlight_outliers=True, group_by=None):
    """
    Create efficient grid of distribution plots with enhanced visualization and analysis features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the variables to plot
    variables : list
        List of column names to plot
    ncols : int
        Number of columns in the grid
    figsize_per_plot : tuple
        Figure size per subplot (width, height)
    kde : bool
        Whether to show kernel density estimate
    bins : int or str
        Number of bins or binning strategy (e.g., 'auto', 'sturges')
    color_map : str
        Matplotlib colormap name for sequential coloring
    highlight_outliers : bool
        Whether to highlight outliers in the distributions
    group_by : str, optional
        Column name to group data by (creates faceted distributions)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Get colormap for sequential coloring
    cmap = plt.cm.get_cmap(color_map)
    colors = [cmap(i/len(variables)) for i in range(len(variables))]
    
    n_vars = len(variables)
    nrows = (n_vars + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, 
                           figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    
    # Handle single row/column cases
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if i < len(axes):
            ax = axes[i]
            # Clean data
            clean_data = data[var].dropna()
            
            if len(clean_data) > 0:
                # Calculate statistics
                mean_val = clean_data.mean()
                median_val = clean_data.median()
                std_val = clean_data.std()
                skew_val = clean_data.skew()
                
                # Detect outliers (beyond 1.5 IQR)
                q1, q3 = np.percentile(clean_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                
                # Create histogram with or without KDE
                sns.histplot(clean_data, bins=bins, kde=kde, ax=ax, 
                             color=colors[i], alpha=0.7, edgecolor='k', linewidth=0.5)
                
                # Add vertical lines for mean and median
                ax.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.2f}')
                
                # Highlight outliers if requested
                if highlight_outliers and len(outliers) > 0:
                    outlier_positions = np.zeros_like(outliers)
                    ax.scatter(outliers, outlier_positions, color='red', s=20, alpha=0.6, marker='o', label=f'Outliers: {len(outliers)}')
                
                # Add comprehensive statistics text
                stats_text = (
                    f'μ = {mean_val:.2f}\n'
                    f'σ = {std_val:.2f}\n'
                    f'Median = {median_val:.2f}\n'
                    f'Skew = {skew_val:.2f}\n'
                    f'n = {len(clean_data)}'
                )
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add normalized y-axis option for better comparison
                ax_twin = ax.twinx()
                sns.kdeplot(clean_data, ax=ax_twin, color='navy', alpha=0.5, linewidth=2)
                ax_twin.set_ylabel('Density', color='navy')
                ax_twin.tick_params(axis='y', colors='navy')
                ax_twin.set_ylim(bottom=0)
                
                # Improve labels and formatting
                title = var.replace('Lab_', '').replace('Log_', '')
                ax.set_title(f"{title}", fontsize=10)
                ax.set_xlabel(var, fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.tick_params(labelsize=8)
                
                # Add compact legend
                if i == 0:  # Add legend only to first plot to avoid repetition
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(fontsize=7, loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                      ha='center', va='center', fontsize=12)
                ax.set_title(f"{var} (No Data)")
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title with dataset info
    fig.text(0.5, 0.01, f'Total variables: {n_vars} | Total samples: {len(data)}', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig

def enhanced_correlation_analysis(data, log_vars, lab_vars, significance_level=0.05):
    
    
    """Enhanced correlation with multiple statistics and significance testing"""
    
    # Initialize results dictionary
    results = {
        'pearson_r': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'pearson_p': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'spearman_r': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'spearman_p': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=float),
        'n_samples': pd.DataFrame(index=log_vars, columns=lab_vars, dtype=int)
    }
    
    print("ENHANCED CORRELATION ANALYSIS:")
    print("=" * 40)
    
    significant_correlations = []
    
    for log_var in log_vars:
        for lab_var in lab_vars:
            # Get clean data pairs
            clean_data = data[[log_var, lab_var]].dropna()
            n = len(clean_data)
            
            if n < 3:  # Need minimum 3 points for correlation
                continue
                
            try:
                # Calculate Pearson correlation
                pearson_r, pearson_p = pearsonr(clean_data[log_var], clean_data[lab_var])
                
                # Calculate Spearman correlation (rank-based, more robust)
                spearman_r, spearman_p = spearmanr(clean_data[log_var], clean_data[lab_var])
                
                # Store results
                results['pearson_r'].loc[log_var, lab_var] = pearson_r
                results['pearson_p'].loc[log_var, lab_var] = pearson_p
                results['spearman_r'].loc[log_var, lab_var] = spearman_r
                results['spearman_p'].loc[log_var, lab_var] = spearman_p
                results['n_samples'].loc[log_var, lab_var] = n
                
                # Check for statistical significance and strong correlation
                if pearson_p <= significance_level and abs(pearson_r) >= 0.6:
                    significant_correlations.append({
                        'log_var': log_var,
                        'lab_var': lab_var,
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'n_samples': n
                    })
                    
            except Exception as e:
                print(f"  Error calculating correlation for {log_var} vs {lab_var}: {e}")
                continue
    
    # Display significant correlations
    print(f"\nSIGNIFICANT STRONG CORRELATIONS (|r| ≥ 0.6, p ≤ {significance_level}):")
    if significant_correlations:
        for corr in sorted(significant_correlations, key=lambda x: abs(x['pearson_r']), reverse=True):
            print(f"{corr['log_var']} ↔ {corr['lab_var']}: "
                  f"r={corr['pearson_r']:.3f} (p={corr['pearson_p']:.3f}, n={corr['n_samples']})")
    else:
        print("    No significant strong correlations found")
    
    # Summary statistics
    all_r = results['pearson_r'].values.flatten()
    all_r = all_r[~pd.isna(all_r)]
    
    print(f"\n CORRELATION SUMMARY:")
    print(f"Total correlations calculated: {len(all_r)}")
    print(f"Mean |r|: {np.abs(all_r).mean():.3f}")
    print(f"Strong correlations (|r| ≥ 0.6): {(np.abs(all_r) >= 0.6).sum()}")
    print(f"Moderate correlations (0.3 ≤ |r| < 0.6): {((np.abs(all_r) >= 0.3) & (np.abs(all_r) < 0.6)).sum()}")
    
    return results

# Define the create_correlation_figure function 
def create_correlation_figure(data, correlations, correlation_type, significance_level, well_name):
    """Create scatter plots with regression lines for specified correlations"""
    if not correlations:
        return None
    
    # Calculate number of rows and columns for subplots
    n_corrs = len(correlations)
    n_cols = min(3, n_corrs)  # Maximum 3 columns
    n_rows = (n_corrs + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    
    # Make axes iterable even for single subplot
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Set custom colors for positive and negative correlations
    if correlation_type == "Positive":
        color = "#00FF7F"  
        title_prefix = "Positive"
    else:
        color = "#FF0000"  
        title_prefix = "Negative"
    
    # FIX: Use the passed 'data' parameter instead of global 'joined'
    original_data = data
    
    # Create scatter plots
    for i, (log_var, lab_var, r, p, n) in enumerate(correlations):
        if i < len(axes):
            ax = axes[i]
            
            # Use original variable names (remove z-score indicators if present)
            orig_log_var = log_var
            orig_lab_var = lab_var
            
            # Clean data for this pair
            pair_data = original_data[[orig_log_var, orig_lab_var]].dropna()
            
            # Scatter plot with original data
            ax.scatter(pair_data[orig_log_var], pair_data[orig_lab_var], 
                     alpha=0.7, s=50, edgecolor='k', linewidth=0.5,
                     color=color)
            
            # Add regression line using original data
            sns.regplot(x=orig_log_var, y=orig_lab_var, data=pair_data, 
                     scatter=False, ax=ax, color=color, 
                     line_kws={'linewidth': 2})
            
            # Add correlation statistics
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.4f}\nn = {n}',
                  transform=ax.transAxes, fontsize=10,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Labels and title - use more readable names
            ax.set_xlabel(orig_log_var.replace('Log_', ''), fontsize=12)
            ax.set_ylabel(orig_lab_var.replace('Lab_', ''), fontsize=12)
            ax.set_title(f'{orig_log_var.replace("Log_", "")} vs {orig_lab_var.replace("Lab_", "")}', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_corrs, len(axes)):
        axes[i].set_visible(False)
    
    # Add main title
    plt.suptitle(f'{title_prefix} Significant Correlations (p ≤ {significance_level})', 
               fontsize=16, y=1.02)
    
    # Save the figure with well name
    plt.tight_layout()
    filename = f'imgs/{well_name}_{correlation_type.lower()}_correlations.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_significant_correlations(data, correlation_results, well_name, significance_level=0.05, min_correlation=0.6):
    """Visualize significant correlations with scatter plots"""
    
    # Get correlation matrix and p-values
    corr_matrix = correlation_results['pearson_r']
    p_matrix = correlation_results['pearson_p']
    n_matrix = correlation_results['n_samples']
    
    # Find significant positive correlations
    positive_correlations = []
    for log_var in corr_matrix.index:
        for lab_var in corr_matrix.columns:
            r = corr_matrix.loc[log_var, lab_var]
            p = p_matrix.loc[log_var, lab_var]
            n = n_matrix.loc[log_var, lab_var]
            
            if not pd.isna(r) and not pd.isna(p) and r >= min_correlation and p <= significance_level:
                positive_correlations.append((log_var, lab_var, r, p, n))
    
    # Find significant negative correlations
    negative_correlations = []
    for log_var in corr_matrix.index:
        for lab_var in corr_matrix.columns:
            r = corr_matrix.loc[log_var, lab_var]
            p = p_matrix.loc[log_var, lab_var]
            n = n_matrix.loc[log_var, lab_var]
            
            if not pd.isna(r) and not pd.isna(p) and r <= -min_correlation and p <= significance_level:
                negative_correlations.append((log_var, lab_var, r, p, n))
    
    # Sort correlations by strength (absolute value)
    positive_correlations.sort(key=lambda x: x[2], reverse=True)  # r is at index 2
    negative_correlations.sort(key=lambda x: x[2])  # Sort negative from strongest (most negative) to weakest
    
    # Create visualizations for positive correlations
    if positive_correlations:
        print(f"\nPOSITIVE SIGNIFICANT CORRELATIONS (r ≥ {min_correlation}, p ≤ {significance_level}):")
        for log_var, lab_var, r, p, n in positive_correlations:
            print(f"{log_var} ↔ {lab_var}: r = {r:.3f} (p = {p:.4f}, n = {n})")
        
        pos_fig = create_correlation_figure(data, positive_correlations, "Positive", significance_level, well_name)
    else:
        print(f"\n❌ No significant positive correlations found (r ≥ {min_correlation}, p ≤ {significance_level})")
    
    # Create visualizations for negative correlations
    if negative_correlations:
        print(f"\nNEGATIVE SIGNIFICANT CORRELATIONS (r ≤ -{min_correlation}, p ≤ {significance_level}):")
        for log_var, lab_var, r, p, n in negative_correlations:
            print(f"{log_var} ↔ {lab_var}: r = {r:.3f} (p = {p:.4f}, n = {n})")
            
        neg_fig = create_correlation_figure(data, negative_correlations, "Negative", significance_level, well_name)
    else:
        print(f"\n❌ No significant negative correlations found (r ≤ -{min_correlation}, p ≤ {significance_level})")
    
    return {
        "positive_correlations": positive_correlations,
        "negative_correlations": negative_correlations
    }
    
def create_correlation_network(correlation_results, well_name, min_correlation=0.6, max_connections=50):
    """
    Create a network visualization of significant correlations.
    
    Parameters:
    correlation_results (dict): Dictionary containing correlation matrices
    min_correlation (float): Minimum absolute correlation to include
    max_connections (int): Maximum number of connections to show (for readability)
    """
    # Extract matrices from results
    pearson_r = correlation_results['pearson_r']
    pearson_p = correlation_results['pearson_p']
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes for log variables (left side)
    log_vars = pearson_r.index.tolist()
    for i, var in enumerate(log_vars):
        # Strip the 'Log_' prefix for cleaner labels
        label = var.replace('Log_', '')
        G.add_node(var, type='log', label=label, pos=(0, -i))
    
    # Add nodes for lab variables (right side)
    lab_vars = pearson_r.columns.tolist()
    for i, var in enumerate(lab_vars):
        # Strip the 'Lab_' prefix for cleaner labels
        label = var.replace('Lab_', '')
        G.add_node(var, type='lab', label=label, pos=(1, -i))
    
    # Add edges for significant correlations
    edges = []
    for log_var in log_vars:
        for lab_var in lab_vars:
            r = pearson_r.loc[log_var, lab_var]
            p = pearson_p.loc[log_var, lab_var]
            
            # Only include strong and significant correlations
            if abs(r) >= min_correlation and p <= 0.05:
                edges.append((log_var, lab_var, abs(r), r))
    
    # Sort edges by correlation strength and limit to max_connections
    edges.sort(key=lambda x: x[2], reverse=True)
    edges = edges[:max_connections]
    
    # Add edges to graph
    for u, v, weight, original_r in edges:
        G.add_edge(u, v, weight=weight, original_r=original_r)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Create figure
    plt.figure(figsize=(12, 14))
    
    # Draw nodes
    log_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'log']
    lab_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'lab']
    
    nx.draw_networkx_nodes(G, pos, nodelist=log_nodes, node_color='skyblue', 
                          node_size=500, alpha=0.8, node_shape='o')
    nx.draw_networkx_nodes(G, pos, nodelist=lab_nodes, node_color='lightgreen', 
                          node_size=500, alpha=0.8, node_shape='s')
    
    # Custom colormap for positive/negative correlations
    colors = []
    for u, v in G.edges():
        r = G[u][v]['original_r']
        if r > 0:
            colors.append('green')
        else:
            colors.append('red')
    
    # Draw edges with width proportional to correlation strength
    widths = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7, edge_color=colors)
    
    # Draw labels
    labels = {n: attr['label'] for n, attr in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_weight='bold')
    
    # Add legend
    plt.plot([0], [0], 'o', color='skyblue', markersize=10, label='Log Measurement')
    plt.plot([0], [0], 's', color='lightgreen', markersize=10, label='Lab Measurement')
    plt.plot([0], [0], '-', color='green', linewidth=2, label='Positive Correlation')
    plt.plot([0], [0], '-', color='red', linewidth=2, label='Negative Correlation')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    # Add title and adjust layout
    plt.title(f'Network of Strong Correlations (|r| ≥ {min_correlation}, p ≤ 0.05)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'imgs/{well_name}_correlation_network.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_correlations(all_wells, lab_vars, log_vars, min_corr=0.5):
    
    """
    Analyzes correlations between lab and log variables across wells.
    
    Parameters:
    - all_wells: DataFrame containing well data
    - lab_vars: List of laboratory measurement columns
    - log_vars: List of log measurement columns
    - min_corr: Minimum correlation value to highlight (absolute value)
    """
    # Store all well correlations for comparison
    well_correlations = {}
    filtered_vars_by_well = {}  # Track filtered variables for each well
    top_correlations_by_well = {}
    
    # Analyze each well separately
    for well, data in all_wells.groupby('Well'):
        # Skip wells with too few samples
        if len(data) < 5:
            print(f"Skipping {well}: insufficient samples ({len(data)})")
            continue
            
        print(f"\n{'='*50}\nAnalyzing correlations for well: {well} ({len(data)} samples)\n{'='*50}")
        
        # Filter out zero/constant/NaN columns for this well
        
        # Lab variables filtering (commented out - to enable filtering, uncomment these lines)
        # filtered_lab_vars = [col for col in lab_vars if not (data[col] == 0).all() and not data[col].isna().all()]
        # filtered_lab_vars = [col for col in filtered_lab_vars if data[col].std() > 0]
        # To disable filtering, use original lab_vars:
        filtered_lab_vars = lab_vars
        
        # Log variables filtering (commented out - to enable filtering, uncomment these lines)
        # filtered_log_vars = [col for col in log_vars if not (data[col] == 0).all() and not data[col].isna().all()]
        # filtered_log_vars = [col for col in filtered_log_vars if data[col].std() > 0]
        # To disable filtering, use original log_vars:
        filtered_log_vars = log_vars
        
        # Store filtered variables for this well for later use
        filtered_vars_by_well[well] = {
            'lab': filtered_lab_vars,
            'log': filtered_log_vars
        }
        
        print(f"Using {len(filtered_lab_vars)} lab variables and {len(filtered_log_vars)} log variables (filtering disabled)")
        
        # Calculate correlation matrix between lab and log variables
        corr = data[filtered_lab_vars + filtered_log_vars].corr().loc[filtered_log_vars, filtered_lab_vars]
        well_correlations[well] = corr
        
        # Plot heatmap with improved readability
        plt.figure(figsize=(16, 12))
        mask = np.abs(corr) < min_corr  # Mask weak correlations
        
        # Use a more readable colormap with better contrast
        sns.heatmap(corr, cmap='RdYlGn', vmin=-1, vmax=1, center=0, 
                   annot=True, fmt='.2f', mask=mask, 
                   linewidths=0.5, cbar_kws={'label': 'Correlation Coefficient'},
                   annot_kws={'size': 8})  # Smaller font for correlation values
        
        plt.title(f'Lab-Log Correlations - {well}', fontsize=16)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.show()
        
        # Identify and display top correlations in a more structured way
        corr_flat = corr.unstack()
        top_corr = corr_flat.abs().sort_values(ascending=False).head(10)
        top_correlations_by_well[well] = top_corr
        
        # Format output as a table
        print("\nTop correlations:")
        corr_table = pd.DataFrame({
            'Log Variable': [idx[0] for idx in top_corr.index],
            'Lab Variable': [idx[1] for idx in top_corr.index],
            'Correlation': top_corr.values
        })
        print(corr_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Cross-well correlation comparison (for common pairs)
    if len(well_correlations) >= 2:
        print("\n\nComparison of key correlations across wells:")
        
        # Find common variables across all wells first
        common_logs = set.intersection(*[set(filtered_vars_by_well[well]['log']) for well in well_correlations.keys()])
        common_labs = set.intersection(*[set(filtered_vars_by_well[well]['lab']) for well in well_correlations.keys()])
        
        print(f"Variables present in all wells: {len(common_logs)} log variables and {len(common_labs)} lab variables")
        
        if len(common_logs) == 0 or len(common_labs) == 0:
            print("No common variables across all wells. Using top correlations instead.")
            
            # Find correlation pairs from each well's top correlations
            all_pairs = []
            for well, corr_data in top_correlations_by_well.items():
                all_pairs.extend([(idx[0], idx[1]) for idx in corr_data.index])
            
            # Count frequency of each pair across wells
            from collections import Counter
            pair_counts = Counter(all_pairs)
            
            # Only use pairs that appear in at least 2 wells
            common_pairs = [pair for pair, count in pair_counts.items() if count >= 2]
            
            if not common_pairs:
                print("No common correlation pairs found across wells.")
                return well_correlations
                
            print(f"Found {len(common_pairs)} correlation pairs that appear in multiple wells")
            
            # Create the DataFrame with string indices
            comparison_indices = [f"{pair[0]} vs {pair[1]}" for pair in common_pairs]
            comparison = pd.DataFrame(index=comparison_indices, columns=well_correlations.keys())
            
            # Fill in correlation values where available
            valid_correlations = 0
            for i, pair in enumerate(common_pairs):
                log_var, lab_var = pair
                idx = comparison_indices[i]  # Get the string index
                
                # Debug which wells have these variables
                wells_with_var = []
                for well, corr_df in well_correlations.items():
                    if log_var in filtered_vars_by_well[well]['log'] and lab_var in filtered_vars_by_well[well]['lab']:
                        wells_with_var.append(well)
                        comparison.loc[idx, well] = corr_df.loc[log_var, lab_var]
                        valid_correlations += 1
                    else:
                        comparison.loc[idx, well] = np.nan
                
                # Debug output
                if wells_with_var:
                    print(f"Pair '{idx}' is present in wells: {', '.join(wells_with_var)}")
                else:
                    print(f"WARNING: Pair '{idx}' not found in any well's filtered variables!")
            
            if valid_correlations == 0:
                print("No valid correlation values found! Check data consistency across wells.")
                return well_correlations
            
            # Display table of correlation comparisons
            print(comparison.head(15).to_string(float_format=lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"))
            
            # Visualize comparison of top correlations across wells
            # Only include pairs with data for at least half of the wells
            min_wells_required = len(well_correlations) // 2
            top_common = comparison.dropna(thresh=min_wells_required).head(8)
            
            if not top_common.empty:
                top_common = top_common.astype(float)  # Convert to numeric before plotting
                # IMPROVED VISUALIZATION - larger figure and better formatting
                plt.figure(figsize=(14, 10))
                
                # Enhanced heatmap with RdYlGn colormap and better sizing
                sns.heatmap(top_common.T, 
                    cmap='RdYlGn',
                    center=0, 
                    annot=True, 
                    fmt='.2f',
                    annot_kws={'size': 12},
                    linewidths=1,
                    square=True)
                
                plt.title('Comparison of Key Correlations Across Wells', fontsize=16)
                plt.ylabel('Well', fontsize=14)
                plt.xlabel('Correlation Pair', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(fontsize=12)
                plt.tight_layout()
                plt.show()
            else:
                print("Not enough common correlations to create visualization")
        else:
            # If we have common variables across all wells, use these to create a consistent correlation matrix
            print("Using common variables to create consistent correlation matrix")
            
            # Prepare a DataFrame for comparison
            all_pairs = [(log, lab) for log in common_logs for lab in common_labs]
            comparison_indices = [f"{log} vs {lab}" for log, lab in all_pairs]
            comparison = pd.DataFrame(index=comparison_indices, columns=well_correlations.keys())
            
            # Fill in the correlation values
            for i, (log_var, lab_var) in enumerate(all_pairs):
                idx = comparison_indices[i]
                for well, corr_df in well_correlations.items():
                    comparison.loc[idx, well] = corr_df.loc[log_var, lab_var]
            
            # Sort by average absolute correlation value
            avg_abs_corr = comparison.abs().mean(axis=1)
            comparison = comparison.loc[avg_abs_corr.sort_values(ascending=False).index]
            
            # Display top correlations
            print(comparison.head(15).to_string(float_format=lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"))
            
            # Visualize top correlations
            top_common = comparison.head(8)
            top_common = top_common.astype(float)  # Convert to numeric before plotting
            
            # IMPROVED VISUALIZATION - larger figure and better formatting
            plt.figure(figsize=(14, 10))
            
            # Enhanced heatmap with RdYlGn colormap and better sizing
            sns.heatmap(top_common.T, 
                       cmap='RdYlGn',
                       center=0, 
                       annot=True, 
                       fmt='.2f',
                       annot_kws={'size': 12},
                       linewidths=1,
                       square=True)
            
            plt.title('Comparison of Key Correlations Across Wells', fontsize=16)
            plt.ylabel('Well', fontsize=14)
            plt.xlabel('Correlation Pair', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()
    
    return well_correlations

def plot_correlation_distribution(r_vals, well_name, stats_text):
    mean_r = np.mean(r_vals)
    median_r = np.median(r_vals)
    min_r = np.min(r_vals)
    max_r = np.max(r_vals)
    std_r = np.std(r_vals)
    strong_pos_count = np.sum(r_vals > 0.6)
    strong_neg_count = np.sum(r_vals < -0.6)
    moderate_count = np.sum((np.abs(r_vals) > 0.3) & (np.abs(r_vals) <= 0.6))
    weak_count = np.sum(np.abs(r_vals) <= 0.3)

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    sns.histplot(r_vals, bins=25, kde=True, color='skyblue', 
                 line_kws={'linewidth': 2}, alpha=0.7)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
    plt.axvline(x=mean_r, color='red', linestyle='-', label=f'Mean ({mean_r:.2f})')
    plt.axvline(x=median_r, color='green', linestyle='-', label=f'Median ({median_r:.2f})')
    plt.axvspan(-1, -0.6, alpha=0.2, color='red', label='Strong negative')
    plt.axvspan(-0.6, -0.3, alpha=0.1, color='orange', label='Moderate negative')
    plt.axvspan(-0.3, 0.3, alpha=0.1, color='gray', label='Weak')
    plt.axvspan(0.3, 0.6, alpha=0.1, color='blue', label='Moderate positive')
    plt.axvspan(0.6, 1, alpha=0.2, color='green', label='Strong positive')
    plt.text(1.02, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.title("Distribution of Pearson Correlation Coefficients", fontsize=14)
    plt.xlabel("Correlation Coefficient (r)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(f'imgs/{well_name}_correlation_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
# distribution visual
def create_distribution_grid(data, variables, ncols=4, figsize_per_plot=(4, 3), kde=True, bins=30, 
                             color_map='viridis', highlight_outliers=True, group_by=None):
    """
    Create efficient grid of distribution plots with enhanced visualization and analysis features
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the variables to plot
    variables : list
        List of column names to plot
    ncols : int
        Number of columns in the grid
    figsize_per_plot : tuple
        Figure size per subplot (width, height)
    kde : bool
        Whether to show kernel density estimate
    bins : int or str
        Number of bins or binning strategy (e.g., 'auto', 'sturges')
    color_map : str
        Matplotlib colormap name for sequential coloring
    highlight_outliers : bool
        Whether to highlight outliers in the distributions
    group_by : str, optional
        Column name to group data by (creates faceted distributions)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    # Get colormap for sequential coloring
    cmap = plt.cm.get_cmap(color_map)
    colors = [cmap(i/len(variables)) for i in range(len(variables))]
    
    n_vars = len(variables)
    nrows = (n_vars + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, 
                           figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows))
    
    # Handle single row/column cases
    if nrows * ncols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, var in enumerate(variables):
        if i < len(axes):
            ax = axes[i]
            # Clean data
            clean_data = data[var].dropna()
            
            if len(clean_data) > 0:
                # Calculate statistics
                mean_val = clean_data.mean()
                median_val = clean_data.median()
                std_val = clean_data.std()
                skew_val = clean_data.skew()
                
                # Detect outliers (beyond 1.5 IQR)
                q1, q3 = np.percentile(clean_data, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
                
                # Create histogram with or without KDE
                sns.histplot(clean_data, bins=bins, kde=kde, ax=ax, 
                             color=colors[i], alpha=0.7, edgecolor='k', linewidth=0.5)
                
                # Add vertical lines for mean and median
                ax.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Median: {median_val:.2f}')
                
                # Highlight outliers if requested
                if highlight_outliers and len(outliers) > 0:
                    outlier_positions = np.zeros_like(outliers)
                    ax.scatter(outliers, outlier_positions, color='red', s=20, alpha=0.6, marker='o', label=f'Outliers: {len(outliers)}')
                
                # Add comprehensive statistics text
                stats_text = (
                    f'μ = {mean_val:.2f}\n'
                    f'σ = {std_val:.2f}\n'
                    f'Median = {median_val:.2f}\n'
                    f'Skew = {skew_val:.2f}\n'
                    f'n = {len(clean_data)}'
                )
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Add normalized y-axis option for better comparison
                ax_twin = ax.twinx()
                sns.kdeplot(clean_data, ax=ax_twin, color='navy', alpha=0.5, linewidth=2)
                ax_twin.set_ylabel('Density', color='navy')
                ax_twin.tick_params(axis='y', colors='navy')
                ax_twin.set_ylim(bottom=0)
                
                # Improve labels and formatting
                title = var.replace('Lab_', '').replace('Log_', '')
                ax.set_title(f"{title}", fontsize=10)
                ax.set_xlabel(var, fontsize=8)
                ax.set_ylabel("Count", fontsize=8)
                ax.tick_params(labelsize=8)
                
                # Add compact legend
                if i == 0:  # Add legend only to first plot to avoid repetition
                    handles, labels = ax.get_legend_handles_labels()
                    if handles:
                        ax.legend(fontsize=7, loc='upper right')
            else:
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                      ha='center', va='center', fontsize=12)
                ax.set_title(f"{var} (No Data)")
    
    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)
    
    # Add overall title with dataset info
    fig.text(0.5, 0.01, f'Total variables: {n_vars} | Total samples: {len(data)}', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    return fig



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