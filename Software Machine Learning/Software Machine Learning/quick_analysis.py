"""
Quick Post-Session Analysis
============================
Run this immediately after uploading a session to the Pi.
Downloads latest session data and generates instant plots.

Usage:
    python quick_analysis.py

Or with session name:
    python quick_analysis.py "Boxing_Session1"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt, find_peaks
import requests
import sys
import os
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
PI_URL   = "http://100.81.213.4:5000"
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_plots")
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_INFO = {
    1: {'name': 'BNO055',     'color': '#2196F3', 'unit': 'm/s²'},
    2: {'name': 'LSM6DS3-D2', 'color': '#4CAF50', 'unit': 'm/s²'},
    3: {'name': 'LSM6DS3-D3', 'color': '#FF9800', 'unit': 'm/s²'},
    4: {'name': 'LSM6DS3-D4', 'color': '#E91E63', 'unit': 'm/s²'},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def butter_lowpass(data, cutoff=20, fs=50, order=4):
    nyq = fs / 2.0
    b, a = butter(order, cutoff/nyq, btype='low')
    return filtfilt(b, a, data)

def get_sessions():
    try:
        r = requests.get(f"{PI_URL}/sessions", timeout=5)
        return r.json()
    except Exception as e:
        print(f"Cannot reach Pi: {e}")
        return []

def download_session(session_name):
    try:
        r = requests.get(f"{PI_URL}/export/session/name/{session_name}", timeout=30)
        from io import StringIO
        df = pd.read_csv(StringIO(r.text))
        return df
    except Exception as e:
        print(f"Download error: {e}")
        return None

# ── Analysis ──────────────────────────────────────────────────────────────────

def analyse_session(df, session_name):
    print(f"\nAnalysing: {session_name}")
    print(f"Total rows: {len(df)}")

    devices = sorted(df['device_id'].unique())
    print(f"Devices: {devices}")

    results = {}
    for dev_id in devices:
        dev_df = df[df['device_id'] == dev_id].copy().sort_values('wall_clock_ms')
        if len(dev_df) < 10:
            continue

        # Compute timing
        dev_df['time_s'] = (dev_df['wall_clock_ms'] - dev_df['wall_clock_ms'].iloc[0]) / 1000.0
        duration = dev_df['time_s'].iloc[-1]

        # Actual sample rate
        diffs = np.diff(dev_df['wall_clock_ms'].values)
        median_dt = np.median(diffs)
        actual_hz = 1000.0 / median_dt if median_dt > 0 else 0

        # Accel magnitude
        ax = dev_df['accel_x'].values.astype(float)
        ay = dev_df['accel_y'].values.astype(float)
        az = dev_df['accel_z'].values.astype(float)
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        dynamic = np.abs(mag - 9.81)

        # Filter
        try:
            mag_f = butter_lowpass(mag, fs=max(actual_hz, 10))
            dynamic_f = np.abs(mag_f - 9.81)
        except Exception:
            mag_f = mag
            dynamic_f = dynamic

        # Detect peaks
        threshold = 5.0
        min_dist  = max(int(actual_hz * 0.3), 3)
        peaks, _  = find_peaks(dynamic_f, height=threshold, distance=min_dist)

        loc = dev_df['body_location'].iloc[0] if 'body_location' in dev_df.columns else f"Device {dev_id}"

        results[dev_id] = {
            'df':        dev_df,
            'time_s':    dev_df['time_s'].values,
            'mag':       mag,
            'mag_f':     mag_f,
            'dynamic_f': dynamic_f,
            'peaks':     peaks,
            'duration':  duration,
            'actual_hz': actual_hz,
            'location':  loc,
            'n_impacts': len(peaks),
        }

        print(f"\n  Device {dev_id} ({loc}):")
        print(f"    Rows: {len(dev_df)}")
        print(f"    Duration: {duration:.1f}s")
        print(f"    Sample rate: {actual_hz:.1f} Hz")
        print(f"    Max accel magnitude: {mag.max():.2f} m/s²")
        print(f"    Impact events (>{threshold} m/s² dynamic): {len(peaks)}")

    return results

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_session(results, session_name):
    n_devices = len(results)
    if n_devices == 0:
        print("No data to plot")
        return

    fig = plt.figure(figsize=(16, 4 * n_devices + 2))
    gs  = gridspec.GridSpec(n_devices + 1, 2, height_ratios=[1]*n_devices + [0.3])

    fig.suptitle(f'Session Analysis: {session_name}', fontsize=14, fontweight='bold', y=0.98)

    for row, (dev_id, res) in enumerate(results.items()):
        info  = DEVICE_INFO.get(dev_id, {'name': f'D{dev_id}', 'color': 'gray'})
        color = info['color']
        t     = res['time_s']

        # Left — accel XYZ
        ax_plot = fig.add_subplot(gs[row, 0])
        df = res['df']
        ax_plot.plot(t, df['accel_x'].values, alpha=0.7, linewidth=0.8, label='X', color='red')
        ax_plot.plot(t, df['accel_y'].values, alpha=0.7, linewidth=0.8, label='Y', color='green')
        ax_plot.plot(t, df['accel_z'].values, alpha=0.7, linewidth=0.8, label='Z', color='blue')
        ax_plot.set_ylabel('Accel (m/s²)', fontsize=8)
        ax_plot.set_title(f'D{dev_id} {res["location"]} — XYZ Accelerometer\n'
                         f'({len(df)} samples @ {res["actual_hz"]:.1f}Hz, {res["duration"]:.1f}s)',
                         fontsize=9)
        ax_plot.legend(loc='upper right', fontsize=7)
        ax_plot.grid(True, alpha=0.3)
        if row < n_devices - 1:
            ax_plot.set_xticklabels([])
        else:
            ax_plot.set_xlabel('Time (seconds)', fontsize=8)

        # Right — magnitude + impacts
        ax_mag = fig.add_subplot(gs[row, 1])
        ax_mag.plot(t, res['mag_f'], color=color, linewidth=1, label='Accel magnitude (filtered)')
        ax_mag.axhline(9.81, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, label='Gravity (9.81)')

        if len(res['peaks']) > 0:
            ax_mag.scatter(t[res['peaks']], res['mag_f'][res['peaks']],
                          color='red', zorder=5, s=30, label=f'Impacts ({res["n_impacts"]})')

        ax_mag.set_ylabel('Magnitude (m/s²)', fontsize=8)
        ax_mag.set_title(f'D{dev_id} — Accel Magnitude + Impact Detection\n'
                        f'({res["n_impacts"]} impacts detected)', fontsize=9)
        ax_mag.legend(loc='upper right', fontsize=7)
        ax_mag.grid(True, alpha=0.3)
        if row < n_devices - 1:
            ax_mag.set_xticklabels([])
        else:
            ax_mag.set_xlabel('Time (seconds)', fontsize=8)

    # Summary bar at bottom
    ax_sum = fig.add_subplot(gs[n_devices, :])
    ax_sum.axis('off')
    summary_text = '   |   '.join([
        f"D{dev_id} {res['location']}: {res['n_impacts']} impacts, "
        f"{res['actual_hz']:.1f}Hz, {res['duration']:.1f}s"
        for dev_id, res in results.items()
    ])
    ax_sum.text(0.5, 0.5, summary_text, ha='center', va='center',
               fontsize=9, transform=ax_sum.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = os.path.join(OUT_DIR, f"{session_name}_{timestamp}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved: {out_path}")

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("Quick Post-Session Analysis")
    print("="*60)

    # Check Pi connection
    try:
        r = requests.get(f"{PI_URL}/ping", timeout=5)
        print(f"Pi connected: {r.json().get('message')}")
    except Exception:
        print("WARNING: Cannot reach Pi — using local CSV files instead")
        # Try local file
        if len(sys.argv) > 1:
            local_path = sys.argv[1]
            if os.path.exists(local_path):
                df = pd.read_csv(local_path)
                session_name = os.path.splitext(os.path.basename(local_path))[0]
                results = analyse_session(df, session_name)
                plot_session(results, session_name)
            else:
                print(f"File not found: {local_path}")
        return

    # Get session to analyse
    if len(sys.argv) > 1:
        session_name = sys.argv[1]
    else:
        # Show available sessions and let user pick
        sessions = get_sessions()
        if not sessions:
            print("No sessions found on Pi")
            return

        print(f"\nAvailable sessions ({len(sessions)}):")
        for i, s in enumerate(sessions):
            start = s.get('start_germany', s.get('start_ms', ''))
            print(f"  {i+1}. {s['session_name']} — {s['row_count']} rows — {start}")

        choice = input("\nEnter session number or name: ").strip()
        try:
            idx = int(choice) - 1
            session_name = sessions[idx]['session_name']
        except (ValueError, IndexError):
            session_name = choice

    print(f"\nDownloading session: {session_name}")
    df = download_session(session_name)

    if df is None or len(df) == 0:
        print("No data downloaded")
        return

    # Also save locally
    local_path = os.path.join(OUT_DIR, f"{session_name}_raw.csv")
    df.to_csv(local_path, index=False)
    print(f"Raw data saved: {local_path}")

    # Analyse and plot
    results = analyse_session(df, session_name)
    plot_session(results, session_name)

    print("\nDone! Check session_plots folder for saved plots.")
    print("\nNext steps:")
    print("  1. Run label_boxing_data.py to label the events")
    print("  2. Run preprocess.py to extract features")
    print("  3. Run train_model.py to update the ML model")

if __name__ == '__main__':
    main()
