"""
Boxing IMU Preprocessing Pipeline
===================================
Loads raw IMU data + labels, filters signals,
extracts windows around each labelled event,
and saves ready-to-use ML training data.

Usage:
    python preprocess.py

Output:
    - windows_all.csv       : all labelled windows flattened for ML
    - features_all.csv      : extracted features per window
    - preprocessing_report.txt : summary statistics
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft
import os
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ─────────────────────────────────────────────────────────────

DATA_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(DATA_DIR, 'ml_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Window around each peak: ms before and after impact
WINDOW_PRE_MS  = 200   # 200ms before peak
WINDOW_POST_MS = 400   # 400ms after peak
TOTAL_WINDOW_MS = WINDOW_PRE_MS + WINDOW_POST_MS  # 600ms total

# Target sample rate after resampling
TARGET_HZ = 50
TARGET_SAMPLES = int(TOTAL_WINDOW_MS / 1000 * TARGET_HZ)  # 30 samples per window

# Low-pass filter settings
FILTER_CUTOFF_HZ = 20
FILTER_ORDER     = 4

# Sessions and their label files
SESSIONS = [
    {
        'data_file':  'Session1Boxing_D1_Right_Wrist_BNO055.csv',
        'label_file': 'labels_Session1Boxing_device1.csv',
        'session':    'Session1Boxing',
        'device_id':  1,
        'sensor':     'BNO055',
        'location':   'Right Wrist',
    },
    {
        'data_file':  'Session2_4sensors_D1_Right_Wrist_BNO055.csv',
        'label_file': 'labels_Session2_4sensors_device1.csv',
        'session':    'Session2_4sensors',
        'device_id':  1,
        'sensor':     'BNO055',
        'location':   'Right Wrist',
    },
]

# Columns to use for ML
SIGNAL_COLS = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']

# ── Signal Processing ──────────────────────────────────────────────────────────

def butter_lowpass_filter(data, cutoff=FILTER_CUTOFF_HZ, fs=TARGET_HZ, order=FILTER_ORDER):
    nyq = fs / 2.0
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def estimate_sample_rate(wall_clock_ms):
    diffs = np.diff(wall_clock_ms)
    median_dt = np.median(diffs)
    return 1000.0 / median_dt  # Hz

def remove_gravity(df):
    """Remove gravity component using rolling mean baseline."""
    for col in ['accel_x', 'accel_y', 'accel_z']:
        baseline = df[col].rolling(window=50, min_periods=1, center=True).mean()
        df[f'{col}_dynamic'] = df[col] - baseline
    return df

def compute_magnitude(df):
    """Compute acceleration and gyro magnitudes."""
    df['accel_magnitude'] = np.sqrt(
        df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2
    )
    df['dynamic_accel'] = np.abs(df['accel_magnitude'] - 9.81)
    df['gyro_magnitude'] = np.sqrt(
        df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2
    )
    return df

# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_features(window_df, label, session, device_id, location, peak_time_ms):
    """Extract ML features from a single window."""
    features = {
        'label':      label,
        'session':    session,
        'device_id':  device_id,
        'location':   location,
        'peak_time_ms': peak_time_ms,
        'n_samples':  len(window_df),
    }

    for col in SIGNAL_COLS + ['accel_magnitude', 'dynamic_accel', 'gyro_magnitude']:
        if col not in window_df.columns:
            continue
        vals = window_df[col].values.astype(float)
        if len(vals) == 0:
            continue

        features[f'{col}_mean']   = np.mean(vals)
        features[f'{col}_std']    = np.std(vals)
        features[f'{col}_max']    = np.max(vals)
        features[f'{col}_min']    = np.min(vals)
        features[f'{col}_range']  = np.max(vals) - np.min(vals)
        features[f'{col}_abs_mean'] = np.mean(np.abs(vals))

        # Root mean square
        features[f'{col}_rms'] = np.sqrt(np.mean(vals**2))

        # Zero crossing rate
        signs = np.sign(vals)
        features[f'{col}_zcr'] = np.sum(np.diff(signs) != 0) / len(vals)

        # Peak count within window
        if len(vals) > 5:
            peaks, _ = find_peaks(np.abs(vals), height=np.std(vals))
            features[f'{col}_peak_count'] = len(peaks)
        else:
            features[f'{col}_peak_count'] = 0

        # FFT energy in frequency bands
        if len(vals) >= 10:
            fft_vals = np.abs(fft(vals))[:len(vals)//2]
            freqs = np.fft.fftfreq(len(vals), d=1.0/TARGET_HZ)[:len(vals)//2]
            features[f'{col}_energy_0_5hz']  = np.sum(fft_vals[(freqs >= 0)  & (freqs < 5)]**2)
            features[f'{col}_energy_5_15hz'] = np.sum(fft_vals[(freqs >= 5)  & (freqs < 15)]**2)
            features[f'{col}_energy_15hz+']  = np.sum(fft_vals[freqs >= 15]**2)
            features[f'{col}_dominant_freq'] = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0

    # Quaternion features (BNO055 only)
    if 'quat_w' in window_df.columns:
        qw = window_df['quat_w'].values
        qx = window_df['quat_x'].values
        qy = window_df['quat_y'].values
        qz = window_df['quat_z'].values
        features['quat_w_mean'] = np.mean(qw)
        features['quat_x_mean'] = np.mean(qx)
        features['quat_y_mean'] = np.mean(qy)
        features['quat_z_mean'] = np.mean(qz)
        # Rotation angle change
        features['rotation_range'] = np.max(np.abs(qw)) - np.min(np.abs(qw))

    return features

def extract_raw_window(window_df, target_samples=TARGET_SAMPLES):
    """Flatten raw signal values for deep learning approaches."""
    raw = {}
    for col in SIGNAL_COLS:
        if col not in window_df.columns:
            vals = np.zeros(target_samples)
        else:
            vals = window_df[col].values.astype(float)
            # Resample to fixed length
            if len(vals) != target_samples:
                indices = np.linspace(0, len(vals)-1, target_samples)
                vals = np.interp(indices, np.arange(len(vals)), vals)
        for i, v in enumerate(vals):
            raw[f'{col}_{i}'] = v
    return raw

# ── Main Pipeline ─────────────────────────────────────────────────────────────

def process_session(session_config):
    data_path  = os.path.join(DATA_DIR, session_config['data_file'])
    label_path = os.path.join(DATA_DIR, session_config['label_file'])

    print(f"\n{'='*60}")
    print(f"Processing: {session_config['session']} — D{session_config['device_id']}")
    print(f"{'='*60}")

    # Load data
    if not os.path.exists(data_path):
        print(f"  ERROR: Data file not found: {data_path}")
        return None, None

    if not os.path.exists(label_path):
        print(f"  ERROR: Label file not found: {label_path}")
        return None, None

    df = pd.read_csv(data_path)
    labels_df = pd.read_csv(label_path)

    print(f"  Data rows: {len(df)}")
    print(f"  Labelled events: {len(labels_df)}")
    print(f"  Labels: {labels_df['label'].value_counts().to_dict()}")

    # Estimate and print sample rate
    fs = estimate_sample_rate(df['wall_clock_ms'].values)
    print(f"  Sample rate: {fs:.1f} Hz")

    # Sort by time
    df = df.sort_values('wall_clock_ms').reset_index(drop=True)

    # Compute magnitudes
    df = compute_magnitude(df)
    df = remove_gravity(df)

    # Apply low-pass filter
    for col in SIGNAL_COLS:
        if col in df.columns:
            try:
                df[f'{col}_filtered'] = butter_lowpass_filter(
                    df[col].values.astype(float), fs=fs
                )
            except Exception:
                df[f'{col}_filtered'] = df[col]

    # Extract windows around each labelled peak
    all_features = []
    all_raw      = []
    skipped      = 0

    for _, label_row in labels_df.iterrows():
        peak_ms   = label_row['peak_time_ms']
        label     = label_row['label']
        win_start = peak_ms - WINDOW_PRE_MS
        win_end   = peak_ms + WINDOW_POST_MS

        # Extract window
        window = df[
            (df['wall_clock_ms'] >= win_start) &
            (df['wall_clock_ms'] <= win_end)
        ].copy()

        if len(window) < 5:
            print(f"  Skipping peak at {peak_ms}ms — too few samples ({len(window)})")
            skipped += 1
            continue

        # Extract features
        features = extract_features(
            window, label,
            session_config['session'],
            session_config['device_id'],
            session_config['location'],
            peak_ms
        )
        all_features.append(features)

        # Extract raw window for deep learning
        raw = {
            'label':    label,
            'session':  session_config['session'],
            'device_id': session_config['device_id'],
            'peak_time_ms': peak_ms,
        }
        raw.update(extract_raw_window(window))
        all_raw.append(raw)

    print(f"  Windows extracted: {len(all_features)}")
    print(f"  Windows skipped:   {skipped}")

    features_df = pd.DataFrame(all_features)
    raw_df      = pd.DataFrame(all_raw)

    return features_df, raw_df

def main():
    print("\nBoxing IMU Preprocessing Pipeline")
    print("="*60)

    all_features = []
    all_raw      = []
    report_lines = []

    for session_config in SESSIONS:
        features_df, raw_df = process_session(session_config)

        if features_df is not None and len(features_df) > 0:
            all_features.append(features_df)
            all_raw.append(raw_df)

            # Report
            report_lines.append(f"\nSession: {session_config['session']} D{session_config['device_id']}")
            report_lines.append(f"  Windows: {len(features_df)}")
            report_lines.append(f"  Labels:  {features_df['label'].value_counts().to_dict()}")

    if not all_features:
        print("\nERROR: No data processed. Check file paths.")
        return

    # Combine all sessions
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_raw      = pd.concat(all_raw,      ignore_index=True)

    # Save outputs
    features_path = os.path.join(OUTPUT_DIR, 'features_all.csv')
    raw_path      = os.path.join(OUTPUT_DIR, 'windows_raw_all.csv')
    report_path   = os.path.join(OUTPUT_DIR, 'preprocessing_report.txt')

    combined_features.to_csv(features_path, index=False)
    combined_raw.to_csv(raw_path, index=False)

    print(f"\n{'='*60}")
    print(f"DONE — Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"  features_all.csv    : {len(combined_features)} windows x {len(combined_features.columns)} features")
    print(f"  windows_raw_all.csv : {len(combined_raw)} windows x {TARGET_SAMPLES * len(SIGNAL_COLS)} raw values")

    # Label distribution
    print(f"\nLabel distribution:")
    label_counts = combined_features['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:25s}: {count} windows")

    # Write report
    report_lines.insert(0, "Boxing IMU Preprocessing Report")
    report_lines.insert(1, "="*60)
    report_lines.append(f"\nTotal windows: {len(combined_features)}")
    report_lines.append(f"Total features per window: {len(combined_features.columns)}")
    report_lines.append(f"Window size: {TOTAL_WINDOW_MS}ms ({TARGET_SAMPLES} samples at {TARGET_HZ}Hz)")
    report_lines.append(f"\nLabel distribution:")
    for label, count in label_counts.items():
        report_lines.append(f"  {label}: {count}")

    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nReport saved: {report_path}")
    print("\nNext step: run train_model.py to train the ML classifier!")

if __name__ == '__main__':
    main()
