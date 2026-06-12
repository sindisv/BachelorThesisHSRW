"""
Boxing IMU Labelling Tool
=========================
Labels impact events in your session CSV files.

Usage:
    python label_boxing_data.py

Requirements:
    pip install pandas numpy matplotlib scipy

How it works:
1. Load a session CSV
2. Auto-detect impact peaks from accelerometer magnitude
3. Show each peak in context — you confirm or change the label
4. Save a labels CSV for ML training

Controls during labelling:
    Press ENTER to accept the suggested label
    Type a number to pick a different label
    Type 's' to skip this peak
    Type 'q' to quit and save
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks, butter, filtfilt
import os
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

# Movement sequences per session — edit these to match what you did
SESSION_SEQUENCES = {
    'Session1Boxing': {
        'devices': {
            1: {'location': 'Right Wrist (BNO055)', 'sequence': ['Jab', 'Cross', 'Hook', 'Uppercut']},
            3: {'location': 'Left Wrist (LSM6DS3)',  'sequence': ['Jab', 'Cross', 'Hook', 'Uppercut']},
        }
    },
    'Session2_4sensors': {
        'devices': {
            1: {'location': 'Right Wrist (BNO055)', 'sequence': ['Jab', 'Cross', 'Hook', 'Uppercut', 'Roundhouse_Right', 'LowKick_Left']},
            2: {'location': 'Right Ankle (LSM6DS3)', 'sequence': ['Jab', 'Cross', 'Hook', 'Uppercut', 'Roundhouse_Right', 'LowKick_Left']},
            3: {'location': 'Left Wrist (LSM6DS3)',  'sequence': ['Jab', 'Cross', 'Hook', 'Uppercut', 'Roundhouse_Right', 'LowKick_Left']},
            4: {'location': 'Left Ankle (LSM6DS3)',  'sequence': ['Jab', 'Cross', 'Hook', 'Uppercut', 'Roundhouse_Right', 'LowKick_Left']},
        }
    }
}

# All possible movement labels
ALL_LABELS = [
    'Jab', 'Cross', 'Hook', 'Uppercut',
    'Roundhouse_Right', 'Roundhouse_Left',
    'LowKick_Right', 'LowKick_Left',
    'FrontKick_Right', 'FrontKick_Left',
    'Rest', 'Unknown'
]

# Impact detection thresholds
BNO055_THRESHOLD = 5.0    # lowered to catch more events
SENSE_THRESHOLD  = 5.0
MIN_PEAK_DISTANCE_MS = 300  # minimum 300ms between peaks

# ── Signal Processing ──────────────────────────────────────────────────────────

def butter_lowpass(data, cutoff=20, fs=50, order=4):
    """Low-pass filter to remove high frequency noise."""
    nyq = fs / 2
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def compute_accel_magnitude(df):
    """Compute total acceleration magnitude and remove gravity."""
    mag = np.sqrt(df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)
    dynamic = np.abs(mag - 9.81)
    return mag, dynamic

def detect_peaks(dynamic_accel, wall_clock_ms, threshold, min_distance_ms=300):
    """Find impact peaks above threshold with minimum distance between them."""
    # Estimate sample rate
    dt_ms = np.diff(wall_clock_ms).mean()
    min_samples = int(min_distance_ms / dt_ms)

    peaks, properties = find_peaks(
        dynamic_accel,
        height=threshold,
        distance=max(min_samples, 5)
    )
    return peaks

# ── Labelling ──────────────────────────────────────────────────────────────────

def label_session(df, device_id, session_name, sequence, location, output_path):
    """Interactive labelling of detected peaks."""

    print(f"\n{'='*60}")
    print(f"Labelling: {session_name} — Device {device_id} ({location})")
    print(f"Expected sequence: {' → '.join(sequence)} → repeat")
    print(f"{'='*60}")

    # Filter signal
    try:
        df['accel_x_f'] = butter_lowpass(df['accel_x'].values)
        df['accel_y_f'] = butter_lowpass(df['accel_y'].values)
        df['accel_z_f'] = butter_lowpass(df['accel_z'].values)
    except Exception:
        df['accel_x_f'] = df['accel_x']
        df['accel_y_f'] = df['accel_y']
        df['accel_z_f'] = df['accel_z']

    mag, dynamic = compute_accel_magnitude(df[['accel_x_f', 'accel_y_f', 'accel_z_f']].rename(
        columns={'accel_x_f': 'accel_x', 'accel_y_f': 'accel_y', 'accel_z_f': 'accel_z'}))

    df['accel_magnitude'] = mag
    df['dynamic_accel']   = dynamic

    threshold = BNO055_THRESHOLD if device_id == 1 else SENSE_THRESHOLD
    peaks = detect_peaks(dynamic.values, df['wall_clock_ms'].values, threshold)

    print(f"\nFound {len(peaks)} impact events above {threshold} m/s² threshold")

    if len(peaks) == 0:
        print("No peaks detected — try lowering threshold in config")
        return None

    # Time axis in seconds from session start
    t0 = df['wall_clock_ms'].iloc[0]
    df['time_s'] = (df['wall_clock_ms'] - t0) / 1000.0

    # Labels storage
    labels = []
    seq_idx = 0

    print("\nControls:")
    print("  ENTER = accept suggested label")
    for i, lbl in enumerate(ALL_LABELS):
        print(f"  {i+1} = {lbl}")
    print("  s = skip this peak")
    print("  q = quit and save what we have")
    print()

    # Show full overview plot first
    fig_overview, ax_overview = plt.subplots(figsize=(14, 4))
    ax_overview.plot(df['time_s'], dynamic.values, color='steelblue', linewidth=0.8, label='Dynamic accel')
    ax_overview.axhline(threshold, color='red', linestyle='--', linewidth=1, label=f'Threshold ({threshold} m/s²)')
    ax_overview.scatter(df['time_s'].iloc[peaks], dynamic.values[peaks],
                       color='orange', zorder=5, s=40, label='Detected peaks')
    ax_overview.set_xlabel('Time (seconds)')
    ax_overview.set_ylabel('Dynamic acceleration (m/s²)')
    ax_overview.set_title(f'Overview — {session_name} D{device_id} {location}')
    ax_overview.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)

    # Label each peak
    for peak_num, peak_idx in enumerate(peaks):
        suggested = sequence[seq_idx % len(sequence)]

        # Show zoomed plot around this peak
        window = 100  # samples around peak
        start = max(0, peak_idx - window)
        end   = min(len(df), peak_idx + window)
        df_win = df.iloc[start:end]

        fig, axes = plt.subplots(2, 1, figsize=(12, 6))

        # Accelerometer
        axes[0].plot(df_win['time_s'], df_win['accel_x_f'], label='X', color='red', linewidth=1)
        axes[0].plot(df_win['time_s'], df_win['accel_y_f'], label='Y', color='green', linewidth=1)
        axes[0].plot(df_win['time_s'], df_win['accel_z_f'], label='Z', color='blue', linewidth=1)
        axes[0].axvline(df['time_s'].iloc[peak_idx], color='orange', linewidth=2, label='Peak')
        axes[0].set_ylabel('Accel (m/s²)')
        axes[0].legend(loc='upper right')
        axes[0].set_title(f'Peak {peak_num+1}/{len(peaks)} at t={df["time_s"].iloc[peak_idx]:.2f}s — Suggested: {suggested}')

        # Dynamic magnitude
        axes[1].plot(df_win['time_s'], df_win['dynamic_accel'], color='purple', linewidth=1.5)
        axes[1].axvline(df['time_s'].iloc[peak_idx], color='orange', linewidth=2)
        axes[1].axhline(threshold, color='red', linestyle='--', linewidth=1)
        axes[1].set_ylabel('Dynamic accel (m/s²)')
        axes[1].set_xlabel('Time (seconds)')

        # Mark all peaks on both axes
        for ax in axes:
            for p in peaks:
                if start <= p < end:
                    ax.axvline(df['time_s'].iloc[p], color='orange', alpha=0.3, linewidth=1)

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)

        # Get user input
        print(f"Peak {peak_num+1}/{len(peaks)} at t={df['time_s'].iloc[peak_idx]:.2f}s | "
              f"magnitude={dynamic.values[peak_idx]:.1f} m/s² | Suggested: [{suggested}]")

        user_input = input("Label (ENTER=accept, number=change, s=skip, q=quit): ").strip().lower()

        if user_input == 'q':
            plt.close('all')
            break
        elif user_input == 's':
            print(f"  Skipped peak {peak_num+1}")
            plt.close(fig)
            continue
        elif user_input == '':
            label = suggested
            seq_idx += 1
        else:
            try:
                idx = int(user_input) - 1
                if 0 <= idx < len(ALL_LABELS):
                    label = ALL_LABELS[idx]
                    seq_idx += 1
                else:
                    label = suggested
                    seq_idx += 1
            except ValueError:
                label = suggested
                seq_idx += 1

        peak_time_ms = df['wall_clock_ms'].iloc[peak_idx]
        peak_time_s  = df['time_s'].iloc[peak_idx]
        peak_mag     = dynamic.values[peak_idx]

        # Window around peak: 200ms before, 400ms after
        win_start_ms = peak_time_ms - 200
        win_end_ms   = peak_time_ms + 400

        labels.append({
            'session_name':   session_name,
            'device_id':      device_id,
            'body_location':  location,
            'peak_index':     peak_idx,
            'peak_time_s':    round(peak_time_s, 3),
            'peak_time_ms':   peak_time_ms,
            'window_start_ms': win_start_ms,
            'window_end_ms':   win_end_ms,
            'peak_magnitude':  round(peak_mag, 3),
            'label':           label,
            'sequence_position': seq_idx,
        })

        print(f"  ✓ Labelled as: {label}")
        plt.close(fig)

    plt.close('all')

    if not labels:
        print("No labels saved.")
        return None

    labels_df = pd.DataFrame(labels)
    labels_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(labels_df)} labels to: {output_path}")

    # Print summary
    print("\nLabel summary:")
    print(labels_df['label'].value_counts().to_string())

    return labels_df

# ── File Selection ─────────────────────────────────────────────────────────────

def select_file():
    """Let user pick which session and device to label."""
    print("\n" + "="*60)
    print("Boxing IMU Labelling Tool")
    print("="*60)
    print("\nWhich session do you want to label?")
    print("1. Session1Boxing — Device 1 (Right Wrist BNO055)")
    print("2. Session1Boxing — Device 3 (Left Wrist LSM6DS3)")
    print("3. Session2_4sensors — Device 1 (Right Wrist BNO055)")
    print("4. Session2_4sensors — Device 2 (Right Ankle LSM6DS3)")
    print("5. Session2_4sensors — Device 3 (Left Wrist LSM6DS3)")
    print("6. Session2_4sensors — Device 4 (Left Ankle LSM6DS3)")
    print("7. Load custom CSV file")

    choices = [
        ('Session1Boxing',    1, 'Session1Boxing_device1_dominant_wrist.csv'),
        ('Session1Boxing',    3, 'Session1Boxing_device3_dominant_ankle.csv'),
        ('Session2_4sensors', 1, 'Session2_4sensors_device1_dominant_wrist.csv'),
        ('Session2_4sensors', 2, 'Session2_4sensors_device2_nondominant_wrist.csv'),
        ('Session2_4sensors', 3, 'Session2_4sensors_device3_dominant_ankle.csv'),
        ('Session2_4sensors', 4, 'Session2_4sensors_device4_dominant_shin.csv'),
    ]

    choice = input("\nEnter number (1-7): ").strip()

    if choice == '7':
        csv_path = input("Enter full path to CSV file: ").strip()
        session_name = input("Session name: ").strip()
        device_id = int(input("Device ID (1-4): ").strip())
        location = input("Body location: ").strip()
        sequence = input("Movement sequence (comma separated, e.g. Jab,Cross,Hook,Uppercut): ").strip().split(',')
        sequence = [s.strip() for s in sequence]
        return csv_path, session_name, device_id, location, sequence

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(choices):
            session_name, device_id, filename = choices[idx]
            # Look for file in current directory or common locations
            search_paths = [
                filename,
                os.path.join(os.path.dirname(__file__), filename),
                os.path.join(os.path.expanduser('~'), 'Desktop', filename),
                os.path.join(os.path.expanduser('~'), 'Downloads', filename),
            ]
            csv_path = None
            for p in search_paths:
                if os.path.exists(p):
                    csv_path = p
                    break
            if not csv_path:
                csv_path = input(f"File '{filename}' not found. Enter full path: ").strip()

            dev_config = SESSION_SEQUENCES[session_name]['devices'][device_id]
            location = dev_config['location']
            sequence = dev_config['sequence']
            return csv_path, session_name, device_id, location, sequence
    except (ValueError, KeyError, IndexError):
        pass

    print("Invalid choice")
    sys.exit(1)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    csv_path, session_name, device_id, location, sequence = select_file()

    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")

    # Check required columns
    required = ['accel_x', 'accel_y', 'accel_z', 'wall_clock_ms']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Available: {list(df.columns)}")
        sys.exit(1)

    # Output path
    out_dir = os.path.dirname(csv_path) or '.'
    out_file = os.path.join(out_dir, f"labels_{session_name}_device{device_id}.csv")

    # Check if labels already exist
    if os.path.exists(out_file):
        overwrite = input(f"\nLabels file already exists: {out_file}\nOverwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Aborted.")
            sys.exit(0)

    labels_df = label_session(df, device_id, session_name, sequence, location, out_file)

    if labels_df is not None:
        print(f"\n✓ Done! Labels saved to: {out_file}")
        print("\nNext step: run the preprocessing pipeline with these labels")

if __name__ == '__main__':
    main()
