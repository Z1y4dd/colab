
from dlisio import dlis
import pandas as pd
from pathlib import Path
import traceback
import numpy as np
from scipy.spatial import cKDTree


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
    vprint("🔍 STEP 1: FILE VALIDATION")
    vprint("=" * 40)
    
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"DLIS file not found: {path}")
        
        if not path.suffix.lower() in ['.dlis', '.dls']:
            vprint(f"⚠️ Warning: File extension '{path.suffix}' is not typical for DLIS files")
            
        file_size_mb = path.stat().st_size / (1024 * 1024)
        vprint(f"✅ File found: {path.name}")
        vprint(f"   • Size: {file_size_mb:.1f} MB")
        vprint(f"   • Path: {path}")
        
    except Exception as e:
        vprint(f"❌ File validation failed: {e}")
        return pd.DataFrame()
    
    # 2. DLIS FILE LOADING
    vprint(f"\n🔍 STEP 2: DLIS FILE LOADING")
    vprint("=" * 40)
    
    try:
        # Load DLIS file
        files = dlis.load(str(path))
        vprint(f"✅ DLIS file loaded successfully")
        vprint(f"   • Number of logical files: {len(files)}")
        
        if len(files) == 0:
            raise ValueError("No logical files found in DLIS file")
            
        # Use first logical file
        logical_file = files[0]
        vprint(f"   • Using logical file: {logical_file.name if hasattr(logical_file, 'name') else 'unnamed'}")
        
    except Exception as e:
        vprint(f"❌ DLIS loading failed: {e}")
        traceback.print_exc() if verbose else None
        return pd.DataFrame()
    
    # 3. FRAME VALIDATION
    vprint(f"\n🔍 STEP 3: FRAME VALIDATION")
    vprint("=" * 40)
    
    try:
        frames = logical_file.frames
        vprint(f"✅ Found {len(frames)} frame(s)")
        
        if len(frames) == 0:
            raise ValueError("No frames found in logical file")
            
        # List all available frames
        for i, frame in enumerate(frames):
            frame_name = frame.name if hasattr(frame, 'name') else f"Frame_{i}"
            vprint(f"   • Frame {i}: {frame_name}")
            
        # Validate frame index
        if frame_index >= len(frames):
            vprint(f"⚠️ Frame index {frame_index} out of range, using frame 0")
            frame_index = 0
            
        selected_frame = frames[frame_index]
        frame_name = selected_frame.name if hasattr(selected_frame, 'name') else f"Frame_{frame_index}"
        vprint(f"✅ Selected frame {frame_index}: {frame_name}")
        
    except Exception as e:
        vprint(f"❌ Frame validation failed: {e}")
        return pd.DataFrame()
    
    # 4. CHANNEL ANALYSIS
    vprint(f"\n🔍 STEP 4: CHANNEL ANALYSIS")
    vprint("=" * 40)
    
    try:
        channels = selected_frame.channels
        channel_names = [ch.name for ch in channels]
        
        vprint(f"✅ Found {len(channel_names)} channels:")
        
        # Show channel details
        for i, (ch, name) in enumerate(zip(channels, channel_names)):
            dimension = getattr(ch, 'dimension', 'Unknown')
            units = getattr(ch, 'units', 'Unknown')
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
            vprint(f"⚠️ No standard depth channel found. Available: {channel_names[:5]}...")
            
    except Exception as e:
        vprint(f"❌ Channel analysis failed: {e}")
        return pd.DataFrame()
    
    # 5. DATA EXTRACTION
    vprint(f"\n🔍 STEP 5: DATA EXTRACTION")
    vprint("=" * 40)
    
    try:
        # Get curves data
        curves_data = selected_frame.curves()
        vprint(f"✅ Curves data extracted")
        vprint(f"   • Data type: {type(curves_data)}")
        vprint(f"   • Shape: {curves_data.shape}")
        vprint(f"   • Dtype: {curves_data.dtype}")
        
        # Check if structured array
        if not (hasattr(curves_data.dtype, 'names') and curves_data.dtype.names):
            vprint(f"❌ Data is not a structured array with named fields")
            vprint(f"   • Cannot extract channel data automatically")
            return pd.DataFrame()
            
        field_names = curves_data.dtype.names
        vprint(f"✅ Found {len(field_names)} data fields")
        
    except Exception as e:
        vprint(f"❌ Data extraction failed: {e}")
        traceback.print_exc() if verbose else None
        return pd.DataFrame()
    
    # 6. FIELD TO CHANNEL MAPPING
    vprint(f"\n🔍 STEP 6: FIELD MAPPING")
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
                vprint(f"   ⚠️ Skipping multi-dimensional: {simple_name} {array.shape}")
                skipped_fields.append(f"{simple_name} (multi-dim)")
                continue
                
            # Store valid data
            field_to_channel[field] = simple_name
            data_dict[simple_name] = array
            vprint(f"   ✅ {simple_name}: {len(array)} samples")
            
        except Exception as e:
            vprint(f"   ❌ Error processing {field}: {e}")
            skipped_fields.append(f"{field} (error)")
            
    vprint(f"\n📊 MAPPING SUMMARY:")
    vprint(f"   • Successfully mapped: {len(data_dict)} fields")
    vprint(f"   • Skipped: {len(skipped_fields)} fields")
    if skipped_fields and verbose:
        vprint(f"   • Skipped list: {skipped_fields[:5]}{'...' if len(skipped_fields) > 5 else ''}")
    
    # 7. DATAFRAME CREATION
    vprint(f"\n🔍 STEP 7: DATAFRAME CREATION")
    vprint("=" * 40)
    
    if not data_dict:
        vprint(f"❌ No valid data extracted")
        return pd.DataFrame()
        
    try:
        # Check array lengths
        lengths = {name: len(arr) for name, arr in data_dict.items()}
        unique_lengths = set(lengths.values())
        
        if len(unique_lengths) > 1:
            vprint(f"⚠️ Arrays have different lengths: {dict(list(lengths.items())[:5])}")
            min_len = min(lengths.values())
            vprint(f"   • Truncating all arrays to {min_len} samples")
            data_dict = {name: arr[:min_len] for name, arr in data_dict.items()}
            
        # Create DataFrame
        df = pd.DataFrame(data_dict)
        vprint(f"✅ DataFrame created: {df.shape}")
        
        # Set depth as index
        depth_channel = found_depth if found_depth and found_depth in df.columns else None
        if depth_channel:
            df = df.set_index(depth_channel)
            vprint(f"✅ Set {depth_channel} as index")
            vprint(f"   • Depth range: {df.index.min():.2f} - {df.index.max():.2f}")
        else:
            vprint(f"⚠️ No depth channel found in extracted data")
            
        # Filter requested columns
        if needed is not None:
            available_cols = [col for col in needed if col in df.columns]
            missing_cols = [col for col in needed if col not in df.columns]
            
            if available_cols:
                df = df[available_cols]
                vprint(f"✅ Filtered to {len(available_cols)} requested columns")
            else:
                vprint(f"❌ None of requested columns found: {needed}")
                
            if missing_cols:
                vprint(f"⚠️ Missing requested columns: {missing_cols}")
                
        vprint(f"✅ Final DataFrame: {df.shape}")
        return df
        
    except Exception as e:
        vprint(f"❌ DataFrame creation failed: {e}")
        traceback.print_exc() if verbose else None
        return pd.DataFrame()

# Enhanced usage example
def load_and_validate_dlis(path, **kwargs):
    """Wrapper function with additional validation"""
    try:
        df = dlis_to_df(path, **kwargs)
        
        if df.empty:
            print("❌ Failed to load DLIS data")
            return df
            
        print(f"\n✅ FINAL VALIDATION:")
        print(f"   • Shape: {df.shape}")
        print(f"   • Columns: {list(df.columns)}")
        print(f"   • Index: {df.index.name}")
        print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        return df
        
    except Exception as e:
        print(f"❌ Load and validation failed: {e}")
        return pd.DataFrame()


def match_lab_to_log(log_df, lab_df, tol=0.1):
    """
    For each lab depth, find the nearest log depth within `tol`.
    Returns at most len(lab_df) matched pairs.
    Adds columns for Distance and Match_Type.
    """
    print(f"\nFUNCTION START - INPUT VALIDATION:")
    print(f"   • Log DataFrame shape: {log_df.shape}")
    print(f"   • Lab DataFrame shape: {lab_df.shape}")
    
    # Ensure unique indices
    log = log_df[~log_df.index.duplicated(keep='first')]
    lab = lab_df[~lab_df.index.duplicated(keep='first')]
    
    print(f"   • After deduplication - Log: {len(log)}, Lab: {len(lab)}")
    
    if len(log) == 0 or len(lab) == 0:
        print("❌ ERROR: Empty datasets after deduplication!")
        return pd.DataFrame()
    
    # Convert to float64 arrays
    log_depths = np.array(log.index.values, dtype=np.float64).reshape(-1, 1)
    lab_depths = np.array(lab.index.values, dtype=np.float64).reshape(-1, 1)
    
    print(f"   • Log depths sample: {log_depths[:3].flatten()}")
    print(f"   • Lab depths sample: {lab_depths[:3].flatten()}")
    
    # Build KD-Tree on log depths
    tree = cKDTree(log_depths)
    
    # Query each lab depth
    dists, idxs = tree.query(lab_depths, distance_upper_bound=tol)
    
    # FIXED: Check for finite distances (successful matches)
    mask = np.isfinite(dists)
    
    print(f"   • Lab depths range: {lab_depths.min():.2f} - {lab_depths.max():.2f}")
    print(f"   • Log depths range: {log_depths.min():.2f} - {log_depths.max():.2f}")
    print(f"   • Tolerance: {tol} ft")
    print(f"   • Valid matches found: {mask.sum()}/{len(mask)}")
    
    if mask.sum() > 0:
        print(f"   • Min distance: {dists[mask].min():.2f} ft")
        print(f"   • Max distance: {dists[mask].max():.2f} ft")
        
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
        print("❌ No matches found within tolerance!")
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
    
    print(f"   • Matched samples extracted: {len(matched_lab)} pairs")
    
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
