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
            vprint(f"⚠️ No standard depth channel found. Available: {channel_names[:5]}...")
        
        # Store channel descriptions in metadata to return later
        metadata = {
            'channel_descriptions': channel_descriptions,
            'channels': channels,
            'channel_names': channel_names,
            'depth_channel': found_depth
        }
            
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

def load_full_dsl_log(

    root_dir: str,
    channels: list[str] | None = None,
    frame_idx: int = 0,
    priority_folders: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Discover, filter, load, and concatenate DSL-based DLIS files into one continuous log.

    Args:
        root_dir: Root directory to search for .dlis files.
        channels: List of curve mnemonics to extract (None => all).
        frame_idx: Frame index to use for each DLIS.
        priority_folders: Ordered list of folder names to prefer when duplicates exist.

    Returns:
        full_log: DataFrame indexed by depth with concatenated log curves.
        metadata: Dictionary containing detailed loading information.
    """
    # 1. Discover all DLIS files
    found_files = glob.glob(f"{root_dir}/**/*.dlis", recursive=True)
    print(f"🔍 Found {len(found_files)} total DLIS files")

    # 2. Filter to DSL files only
    dsl_files = [f for f in found_files if "-DSL" in Path(f).stem]
    print(f"📊 Found {len(dsl_files)} DSL-specific files")

    if len(dsl_files) == 0:
        print("❌ No DSL files found!")
        return pd.DataFrame(), {"error": "No DSL files found"}

    # 3. Group by file stem to find duplicates
    stems = {}
    for f in dsl_files:
        stem = Path(f).stem
        stems.setdefault(stem, []).append(f)

    # 4. Select one file per stem based on priority_folders
    selected_files = []
    ignored_duplicates = []

    if priority_folders is None:
        priority_folders = ["Deliverables"]

    for stem, files in stems.items():
        if len(files) > 1:
            print(f"🔄 Multiple files found for {stem}: {len(files)} files")
            
        # pick highest-priority file
        selected = None
        for folder in priority_folders:
            for f in files:
                if folder in f:
                    selected = f
                    print(f"✅ Selected {Path(f).name} (priority: {folder})")
                    break
            if selected:
                break
        if not selected:
            selected = files[0]
            print(f"⚠️ No priority match for {stem}, using first file: {Path(selected).name}")
            
        selected_files.append(selected)
        # record ignored duplicates
        ignored_duplicates.extend([f for f in files if f != selected])

    print(f"📁 Selected {len(selected_files)} files for loading")
    print(f"🗑️ Ignored {len(ignored_duplicates)} duplicate files")

    # 5. Load each selected file into a DataFrame - FIXED PARAMETER NAMES
    dfs = []
    load_meta = []
    
    for i, f in enumerate(selected_files):
        try:
            print(f"\n🔧 Loading file {i+1}/{len(selected_files)}: {Path(f).name}")
            
            # FIXED: Use correct parameter names matching your Module's dlis_to_df function
            # Parameters: ['path', 'needed', 'frame_index', 'verbose']
            df = dlis_to_df(
                path=f, 
                needed=channels,        
                frame_index=frame_idx,  
                verbose=True            
            )
            
            # Handle the case where your function might return tuple or just DataFrame
            if isinstance(df, tuple):
                df, meta = df
                load_meta.append(meta)
            else:
                # If no metadata returned, create basic metadata
                meta = {
                    'path': f,
                    'shape': df.shape,
                    'columns': list(df.columns) if not df.empty else [],
                    'depth_range': (df.index.min(), df.index.max()) if not df.empty else (None, None)
                }
                load_meta.append(meta)
            
            if not df.empty:
                print(f"✅ Loaded: {df.shape[0]} samples × {df.shape[1]} channels")
                print(f"Depth range: {df.index.min():.1f} - {df.index.max():.1f} ft")
                dfs.append(df)
            else:
                print(f"⚠️ Empty DataFrame returned for {Path(f).name}")
                
        except Exception as e:
            print(f"❌ Error loading {Path(f).name}: {e}")
            # Print more detailed error for debugging
            import traceback
            print(f"   Full error: {traceback.format_exc()}")
            continue

    # 6. Concatenate and deduplicate by depth index
    if dfs:
        print(f"\n🔗 Concatenating {len(dfs)} DataFrames...")
        
        # Concatenate with error handling
        try:
            full_log = pd.concat(dfs, sort=True).sort_index()
            
            # Check for depth duplicates
            duplicates_before = full_log.index.duplicated().sum()
            if duplicates_before > 0:
                print(f"⚠️ Found {duplicates_before} duplicate depths, removing...")
                full_log = full_log[~full_log.index.duplicated(keep='first')]
                
            print(f"✅ Final combined log: {full_log.shape[0]} samples × {full_log.shape[1]} channels")
            print(f"   Combined depth range: {full_log.index.min():.1f} - {full_log.index.max():.1f} ft")
            
        except Exception as e:
            print(f"❌ Error concatenating DataFrames: {e}")
            full_log = pd.DataFrame()
    else:
        print("❌ No valid DataFrames to concatenate")
        full_log = pd.DataFrame()

    # 7. Compile metadata
    metadata = {
        "found_files": found_files,
        "dsl_files": dsl_files,
        "selected_files": selected_files,
        "ignored_duplicates": ignored_duplicates,
        "load_meta": load_meta,
        "summary": {
            "total_files_found": len(found_files),
            "dsl_files_found": len(dsl_files),
            "files_loaded": len(dfs),
            "files_failed": len(selected_files) - len(dfs),
            "final_shape": full_log.shape if not full_log.empty else (0, 0)
        }
    }

    return full_log, metadata

def load_full_log(
    root_dir: str,
    file_type: str = "dsl",  # Options: "dsl", "dlis", "all"
    channels: list[str] | None = None,
    frame_idx: int = 0,
    priority_folders: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Discover, filter, load, and concatenate DLIS files into one continuous log.
    
    Args:
        root_dir: Root directory to search for .dlis files.
        file_type: Type of files to load:
                  "dsl" - Only DSL files (with "-DSL" in the name)
                  "dlis" - Only non-DSL files (without "-DSL" in the name)
                  "all" - All DLIS files regardless of naming
        channels: List of curve mnemonics to extract (None => all).
        frame_idx: Frame index to use for each DLIS.
        priority_folders: Ordered list of folder names to prefer when duplicates exist.

    Returns:
        full_log: DataFrame indexed by depth with concatenated log curves.
        metadata: Dictionary containing detailed loading information.
    """
    # 1. Discover all DLIS files
    found_files = glob.glob(f"{root_dir}/**/*.dlis", recursive=True)
    print(f"🔍 Found {len(found_files)} total DLIS files")

    # 2. Filter files based on file_type parameter
    if file_type.lower() == "dsl":
        filtered_files = [f for f in found_files if "-DSL" in Path(f).stem]
        print(f"📊 Filtered to {len(filtered_files)} DSL-specific files")
    elif file_type.lower() == "dlis":
        filtered_files = [f for f in found_files if "-DSL" not in Path(f).stem]
        print(f"📊 Filtered to {len(filtered_files)} non-DSL DLIS files")
    else:  # "all"
        filtered_files = found_files
        print(f"📊 Using all {len(filtered_files)} DLIS files")

    if len(filtered_files) == 0:
        print(f"❌ No {file_type.upper()} files found!")
        return pd.DataFrame(), {"error": f"No {file_type.upper()} files found"}

    # 3. Group by file stem to find duplicates
    stems = {}
    for f in filtered_files:
        stem = Path(f).stem
        stems.setdefault(stem, []).append(f)

    # 4. Select one file per stem based on priority_folders
    selected_files = []
    ignored_duplicates = []

    if priority_folders is None:
        priority_folders = ["Deliverables", "FJA"]

    for stem, files in stems.items():
        if len(files) > 1:
            print(f"🔄 Multiple files found for {stem}: {len(files)} files")
            
        # pick highest-priority file
        selected = None
        for folder in priority_folders:
            for f in files:
                if folder in f:
                    selected = f
                    print(f"✅ Selected {Path(f).name} (priority: {folder})")
                    break
            if selected:
                break
        if not selected:
            selected = files[0]
            print(f"⚠️ No priority match for {stem}, using first file: {Path(selected).name}")
            
        selected_files.append(selected)
        # record ignored duplicates
        ignored_duplicates.extend([f for f in files if f != selected])

    print(f"📁 Selected {len(selected_files)} files for loading")
    print(f"🗑️ Ignored {len(ignored_duplicates)} duplicate files")

    # 5. Load each selected file into a DataFrame
    dfs = []
    load_meta = []
    
    for i, f in enumerate(selected_files):
        try:
            print(f"\n🔧 Loading file {i+1}/{len(selected_files)}: {Path(f).name}")
            
            # Try to load with robust error handling
            try:
                # Use custom error handler to handle corrupted files
                from dlisio import dlis
                
                # Configure error handler for corrupt files
                error_handler = None
                # try:
                #     error_handler = dlis.ErrorHandler(
                #         critical=dlis.actions.IGNORE,
                #         major=dlis.actions.IGNORE,
                #         minor=dlis.actions.IGNORE
                #     )
                # except Exception:
                #     # Older dlisio versions might not support this
                #     pass
                
                # Load with appropriate parameters
                if error_handler:
                    df = dlis_to_df(
                        path=f, 
                        needed=channels,        
                        frame_index=frame_idx,  
                        verbose=True,
                        error_handler=error_handler
                    )
                else:
                    df = dlis_to_df(
                        path=f, 
                        needed=channels,        
                        frame_index=frame_idx,  
                        verbose=True            
                    )
            except TypeError:
                # Fallback if error_handler parameter isn't supported
                df = dlis_to_df(
                    path=f, 
                    needed=channels,        
                    frame_index=frame_idx,  
                    verbose=True            
                )
            
            # Handle the case where your function might return tuple or just DataFrame
            if isinstance(df, tuple):
                df, meta = df
                load_meta.append(meta)
            else:
                # If no metadata returned, create basic metadata
                meta = {
                    'path': f,
                    'shape': df.shape,
                    'columns': list(df.columns) if not df.empty else [],
                    'depth_range': (df.index.min(), df.index.max()) if not df.empty else (None, None)
                }
                load_meta.append(meta)
            
            if not df.empty:
                print(f"✅ Loaded: {df.shape[0]} samples × {df.shape[1]} channels")
                print(f"Depth range: {df.index.min():.1f} - {df.index.max():.1f} ft")
                dfs.append(df)
            else:
                print(f"⚠️ Empty DataFrame returned for {Path(f).name}")
                
        except Exception as e:
            print(f"❌ Error loading {Path(f).name}: {e}")
            # Print more detailed error for debugging
            import traceback
            print(f"   Full error: {traceback.format_exc()}")
            continue

    # 6. Concatenate and deduplicate by depth index
    if dfs:
        print(f"\n🔗 Concatenating {len(dfs)} DataFrames...")
        
        # Concatenate with error handling
        try:
            full_log = pd.concat(dfs, sort=True).sort_index()
            
            # Check for depth duplicates
            duplicates_before = full_log.index.duplicated().sum()
            if duplicates_before > 0:
                print(f"⚠️ Found {duplicates_before} duplicate depths, removing...")
                full_log = full_log[~full_log.index.duplicated(keep='first')]
                
            print(f"✅ Final combined log: {full_log.shape[0]} samples × {full_log.shape[1]} channels")
            print(f"   Combined depth range: {full_log.index.min():.1f} - {full_log.index.max():.1f} ft")
            
        except Exception as e:
            print(f"❌ Error concatenating DataFrames: {e}")
            full_log = pd.DataFrame()
    else:
        print("❌ No valid DataFrames to concatenate")
        full_log = pd.DataFrame()

    # 7. Compile metadata
    metadata = {
        "found_files": found_files,
        "filtered_files": filtered_files,
        "selected_files": selected_files,
        "ignored_duplicates": ignored_duplicates,
        "file_type": file_type,
        "load_meta": load_meta,
        "summary": {
            "total_files_found": len(found_files),
            "filtered_files_found": len(filtered_files),
            "files_loaded": len(dfs),
            "files_failed": len(selected_files) - len(dfs),
            "final_shape": full_log.shape if not full_log.empty else (0, 0)
        }
    }

    # Consolidate channel descriptions if available
    all_channel_descriptions = {}
    for file_meta in load_meta:
        if 'channel_descriptions' in file_meta:
            all_channel_descriptions.update(file_meta['channel_descriptions'])
    
    metadata['consolidated_channel_descriptions'] = all_channel_descriptions

    return full_log, metadata

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
        print("❌ ERROR: Empty datasets after deduplication!")
        return pd.DataFrame()
    
    # Convert to float64 arrays
    log_depths = np.array(log.index.values, dtype=np.float64).reshape(-1, 1)
    lab_depths = np.array(lab.index.values, dtype=np.float64).reshape(-1, 1)
    
    # print(f"   • Log depths sample: {log_depths[:3].flatten()}")
    # print(f"   • Lab depths sample: {lab_depths[:3].flatten()}")
    
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
    
    vprint(f"🔍 LOADING {len(file_paths)} DLIS FILES FROM LIST")
    vprint("=" * 50)
    
    # 1. Validate input file paths
    vprint(f"\n📋 STEP 1: VALIDATING FILE PATHS")
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
                    vprint(f"⚠️ {i+1:2d}. {path_obj.name} - Invalid extension")
            else:
                invalid_paths.append((path, "File not found"))
                vprint(f"❌ {i+1:2d}. {Path(path).name} - File not found")
        except Exception as e:
            invalid_paths.append((path, str(e)))
            vprint(f"❌ {i+1:2d}. {Path(path).name} - Error: {e}")
    
    vprint(f"\n📊 VALIDATION SUMMARY:")
    vprint(f"   • Valid files: {len(valid_paths)}")
    vprint(f"   • Invalid files: {len(invalid_paths)}")
    
    if len(valid_paths) == 0:
        vprint("❌ No valid DLIS files found!")
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
                vprint(f"⚠️ Empty DataFrame returned")
                
        except Exception as e:
            failed_loads.append((file_path, str(e)))
            vprint(f"❌ Loading failed: {e}")
            
            # Print detailed error for debugging
            if verbose:
                import traceback
                vprint(f"   Full error: {traceback.format_exc()}")
    
    vprint(f"\n📊 LOADING SUMMARY:")
    vprint(f"   • Files processed: {len(valid_paths)}")
    vprint(f"   • Successfully loaded: {len(dataframes)}")
    vprint(f"   • Failed to load: {len(failed_loads)}")
    
    if len(dataframes) == 0:
        vprint("❌ No files were successfully loaded!")
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
            vprint(f"   • Unique columns across all files: {len(all_columns)}")
            
            # Concatenate all DataFrames
            combined_df = pd.concat(dataframes, sort=True).sort_index()
            vprint(f"   • Combined shape before deduplication: {combined_df.shape}")
            
            # Remove duplicates if requested
            if remove_duplicates:
                duplicates_before = combined_df.index.duplicated().sum()
                if duplicates_before > 0:
                    vprint(f"   • Found {duplicates_before} duplicate depths")
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                    vprint(f"   • Removed duplicates, new shape: {combined_df.shape}")
                else:
                    vprint(f"   • No duplicate depths found")
            
            # Final statistics
            total_memory = combined_df.memory_usage(deep=True).sum() / 1024**2
            vprint(f"\n✅ CONCATENATION COMPLETE:")
            vprint(f"   • Final shape: {combined_df.shape}")
            vprint(f"   • Depth range: {combined_df.index.min():.1f} - {combined_df.index.max():.1f}")
            vprint(f"   • Total memory: {total_memory:.1f} MB")
            vprint(f"   • Columns: {list(combined_df.columns)}")
            
            result_df = combined_df
            
        except Exception as e:
            vprint(f"❌ Concatenation failed: {e}")
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

def create_mineral_composition_bar(data, mineral_columns):
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
    plt.savefig('mineral_composition_vs_depth.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_log_vs_mineral_scatterplots(data, log_vars, mineral_vars, n_cols=3):
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
        print("❌ No valid correlations found with sufficient data points")
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
    plt.savefig('log_vs_mineral_scatterplots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_depth_trend_plots(data, variables, n_cols=3):
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
    plt.savefig('depth_trend_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def create_composite_log_plot(data, log_vars, label_cols=None):
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
    plt.savefig('composite_log_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

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

def plot_zscore_heatmap(data, z_data, vars_list, threshold=3):
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
    plt.savefig('zscore_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
