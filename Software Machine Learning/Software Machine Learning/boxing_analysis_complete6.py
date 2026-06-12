"""
Complete Boxing Analysis — All Sessions + Full ML Pipeline
===========================================================
Sensor layout (orthodox stance, left side forward):
  D1 = Left Ankle    D2 = Right Ankle
  D3 = Left Wrist    D4 = Right Wrist

Features (57 per sensor window):
  Acceleration XYZ: mean, std, max, min
  Gyroscope XYZ: mean, std, max
  Magnitudes: accel mag, gyro mag, dynamic accel
  Jerk: acceleration rate + DECELERATION (jerk_min)
  Rise/fall time: distinguishes real events from retractions
  Impulse, angle change, orientation (quaternion)
  Frequency domain: dominant freq, energy in 3 bands

Models compared:
  Random Forest, SVM (RBF), SVM (Linear),
  Gradient Boosting, KNN, Logistic Regression

Usage:
  Place all CSV files in same folder as this script.
  python boxing_analysis_complete.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings, pickle
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
OUT_DIR  = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(OUT_DIR, "plots_complete")
ML_DIR   = os.path.join(OUT_DIR, "ml_complete")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(ML_DIR,   exist_ok=True)

DEVICE_COLORS = {1:'#2196F3', 2:'#4CAF50', 3:'#FF9800', 4:'#E53935'}
DEVICE_NAMES  = {1:'Left Ankle', 2:'Right Ankle',
                 3:'Left Wrist',  4:'Right Wrist'}
BG = '#FAFAFA'
PRE_MS, POST_MS = 300, 500

# ── Session definitions ───────────────────────────────────────────────────────
S3 = {
    '1a':{'label':'Jab',       'side':'Left', 'group':'Punch (air)','bag':False,'primary':3,'secondary':[1,2,4]},
    '1b':{'label':'Cross',     'side':'Right','group':'Punch (air)','bag':False,'primary':4,'secondary':[1,2,3]},
    '1c':{'label':'Hook',      'side':'Left', 'group':'Punch (air)','bag':False,'primary':3,'secondary':[1,2,4]},
    '1d':{'label':'Hook',      'side':'Right','group':'Punch (air)','bag':False,'primary':4,'secondary':[1,2,3]},
    '1e':{'label':'Uppercut',  'side':'Left', 'group':'Punch (air)','bag':False,'primary':3,'secondary':[1,2,4]},
    '1f':{'label':'Uppercut',  'side':'Right','group':'Punch (air)','bag':False,'primary':4,'secondary':[1,2,3]},
    '2a':{'label':'Jab',       'side':'Left', 'group':'Punch (bag)','bag':True, 'primary':3,'secondary':[1,2,4]},
    '2b':{'label':'Cross',     'side':'Right','group':'Punch (bag)','bag':True, 'primary':4,'secondary':[1,2,3]},
    '2c':{'label':'Hook',      'side':'Left', 'group':'Punch (bag)','bag':True, 'primary':3,'secondary':[1,2,4]},
    '2d':{'label':'Hook',      'side':'Right','group':'Punch (bag)','bag':True, 'primary':4,'secondary':[1,2,3]},
    '3a':{'label':'Low Kick',  'side':'Left', 'group':'Kick (bag)', 'bag':True, 'primary':1,'secondary':[2,3,4]},
    '3b':{'label':'Low Kick',  'side':'Right','group':'Kick (bag)', 'bag':True, 'primary':2,'secondary':[1,3,4]},
    '3c':{'label':'Mid Kick',  'side':'Left', 'group':'Kick (bag)', 'bag':True, 'primary':1,'secondary':[2,3,4]},
    '3d':{'label':'Mid Kick',  'side':'Right','group':'Kick (bag)', 'bag':True, 'primary':2,'secondary':[1,3,4]},
    '3e':{'label':'High Kick', 'side':'Left', 'group':'Kick (bag)', 'bag':True, 'primary':1,'secondary':[2,3,4]},
    '3f':{'label':'High Kick', 'side':'Right','group':'Kick (bag)', 'bag':True, 'primary':2,'secondary':[1,3,4]},
    '4a':{'label':'Front Kick','side':'Left', 'group':'Kick (bag)', 'bag':True, 'primary':1,'secondary':[2,3,4]},
    '4b':{'label':'Front Kick','side':'Right','group':'Kick (bag)', 'bag':True, 'primary':2,'secondary':[1,3,4]},
    '4c':{'label':'Side Kick', 'side':'Left', 'group':'Kick (bag)', 'bag':True, 'primary':1,'secondary':[2,3,4]},
    '4d':{'label':'Side Kick', 'side':'Right','group':'Kick (bag)', 'bag':True, 'primary':2,'secondary':[1,3,4]},
}

S2 = {
    'Device1':          {'label':'Jab',     'side':'Right','group':'Punch (air)','bag':False,'primary':1},
    'Device1Uppercut':  {'label':'Uppercut','side':'Right','group':'Punch (air)','bag':False,'primary':1},
    'Device1hookfront': {'label':'Hook',    'side':'Right','group':'Punch (air)','bag':False,'primary':1},
    'Device3Jab':       {'label':'Jab',     'side':'Left', 'group':'Punch (air)','bag':False,'primary':3},
    'Device3uppercut':  {'label':'Uppercut','side':'Left', 'group':'Punch (air)','bag':False,'primary':3},
    'Boxinghookdevice3':{'label':'Hook',    'side':'Left', 'group':'Punch (air)','bag':False,'primary':3},
}

VALIDATION = ['full_round', 'full_round_2']

# ── Data loading ──────────────────────────────────────────────────────────────
def prep(d):
    d = d.sort_values('wall_clock_ms').copy().reset_index(drop=True)
    if len(d) < 5: return None
    t0 = d['wall_clock_ms'].iloc[0]
    d['time_s']    = (d['wall_clock_ms'] - t0) / 1000.0
    d['accel_mag'] = np.sqrt(d['accel_x']**2 + d['accel_y']**2 + d['accel_z']**2)
    d['gyro_mag']  = np.sqrt(d['gyro_x']**2  + d['gyro_y']**2  + d['gyro_z']**2)
    d['dynamic']   = np.abs(d['accel_mag'] - 9.81)
    diffs = np.diff(d['wall_clock_ms'].values)
    diffs = diffs[diffs > 0]
    d['_fs'] = 1000.0 / np.median(diffs) if len(diffs) > 0 else 17.0
    return d

def load_s3(fname, dev_id):
    path = os.path.join(OUT_DIR, f"{fname}.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    d  = df[df['device_id'] == dev_id]
    return prep(d) if len(d) >= 5 else None

def load_s2(fname, dev_id):
    """Session 2 has shifted columns — apply known remapping."""
    path = os.path.join(OUT_DIR, f"{fname}.csv")
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    df_f = pd.DataFrame({
        'wall_clock_ms': df['session_name'].values,
        'accel_x':       df['device_id'].values,
        'accel_y':       df['body_location'].values,
        'accel_z':       df['device_ts_ms'].values,
        'gyro_x':        df['wall_clock_ms'].values,
        'gyro_y':        df['accel_x'].values,
        'gyro_z':        df['accel_y'].values,
        'quat_w':        df['accel_z'].values,
        'quat_x':        df['gyro_x'].values,
        'quat_y':        df['gyro_y'].values,
        'quat_z':        df['gyro_z'].values,
        'device_num':    df['quat_z'].values,
    })
    d = df_f[df_f['device_num'] == dev_id]
    return prep(d) if len(d) >= 5 else None

# ── Peak detection ────────────────────────────────────────────────────────────
def get_peaks(d):
    """
    Adaptive threshold: mean + 2*std of dynamic acceleration.
    Minimum 400ms between peaks to avoid double-counting.
    """
    fs   = d['_fs'].iloc[0]
    dyn  = d['dynamic'].values
    thr  = max(dyn.mean() + 2.0 * dyn.std(), 5.0)
    dist = max(int(fs * 0.4), 3)
    peaks, _ = find_peaks(dyn, height=thr, distance=dist)
    return peaks, thr

# ── Feature extraction ────────────────────────────────────────────────────────
def extract(win, fs, prefix=''):
    """
    57 features per sensor window including:
    - Acceleration XYZ (mean, std, max, min)
    - Gyroscope XYZ (mean, std, max)
    - Magnitudes (accel, gyro, dynamic)
    - Jerk: acceleration rate AND deceleration (jerk_min_dyn)
    - Rise/fall time: distinguishes real punches from retractions
    - Impulse, angle change (rotation), orientation (quaternion)
    - Frequency domain (dominant freq, energy bands)
    """
    if len(win) < 4: return None
    ax = win['accel_x'].values.astype(float)
    ay = win['accel_y'].values.astype(float)
    az = win['accel_z'].values.astype(float)
    gx = win['gyro_x'].values.astype(float)
    gy = win['gyro_y'].values.astype(float)
    gz = win['gyro_z'].values.astype(float)
    mag  = np.sqrt(ax**2 + ay**2 + az**2)
    gyro = np.sqrt(gx**2 + gy**2 + gz**2)
    dyn  = np.abs(mag - 9.81)
    dt   = 1.0 / fs

    # Jerk per axis and total
    jx = np.diff(ax)/dt; jy = np.diff(ay)/dt; jz = np.diff(az)/dt
    jmag = np.sqrt(jx**2+jy**2+jz**2) if len(jx)>0 else np.array([0.0])

    # Jerk on dynamic signal — captures deceleration as negative values
    j_dyn = np.diff(dyn)/dt if len(dyn)>1 else np.array([0.0])
    jerk_min_dyn = float(j_dyn.min())  # peak deceleration (most negative)
    jerk_max_dyn = float(j_dyn.max())  # peak acceleration rate
    decel_ratio  = abs(jerk_min_dyn) / (jerk_max_dyn + 1e-6)

    # Rise and fall time — key for distinguishing punch from retraction/walking
    peak_idx = int(np.argmax(dyn))
    peak_val = dyn[peak_idx]
    half_val = peak_val * 0.5

    rise_start = 0
    for i in range(peak_idx, -1, -1):
        if dyn[i] < half_val: rise_start = i; break
    rise_time_ms = (peak_idx - rise_start) / fs * 1000

    fall_end = len(dyn) - 1
    for i in range(peak_idx, len(dyn)):
        if dyn[i] < half_val: fall_end = i; break
    fall_time_ms = (fall_end - peak_idx) / fs * 1000

    symmetry = rise_time_ms / (fall_time_ms + 1e-6)

    # Impulse and rotation
    impulse = float(np.trapezoid(dyn, dx=dt))
    ang_x   = float(np.trapezoid(np.abs(gx), dx=dt))
    ang_y   = float(np.trapezoid(np.abs(gy), dx=dt))
    ang_z   = float(np.trapezoid(np.abs(gz), dx=dt))

    # Orientation
    has_q = all(c in win.columns for c in ['quat_w','quat_x','quat_y','quat_z'])
    qw  = win['quat_w'].values.astype(float) if has_q else np.zeros(1)
    qxv = win['quat_x'].values.astype(float) if has_q else np.zeros(1)
    qyv = win['quat_y'].values.astype(float) if has_q else np.zeros(1)
    qzv = win['quat_z'].values.astype(float) if has_q else np.zeros(1)

    # Frequency domain
    fv = np.abs(fft(mag))[:len(mag)//2]
    fr = fftfreq(len(mag), d=1.0/fs)[:len(mag)//2]

    p = prefix
    return {
        # Acceleration XYZ
        f'{p}ax_mean':ax.mean(), f'{p}ax_std':ax.std(),
        f'{p}ax_max':ax.max(),   f'{p}ax_min':ax.min(),
        f'{p}ay_mean':ay.mean(), f'{p}ay_std':ay.std(),
        f'{p}ay_max':ay.max(),   f'{p}ay_min':ay.min(),
        f'{p}az_mean':az.mean(), f'{p}az_std':az.std(),
        f'{p}az_max':az.max(),   f'{p}az_min':az.min(),
        # Gyroscope XYZ
        f'{p}gx_mean':gx.mean(), f'{p}gx_std':gx.std(), f'{p}gx_max':np.abs(gx).max(),
        f'{p}gy_mean':gy.mean(), f'{p}gy_std':gy.std(), f'{p}gy_max':np.abs(gy).max(),
        f'{p}gz_mean':gz.mean(), f'{p}gz_std':gz.std(), f'{p}gz_max':np.abs(gz).max(),
        # Magnitudes
        f'{p}mag_mean':mag.mean(),   f'{p}mag_std':mag.std(),   f'{p}mag_max':mag.max(),
        f'{p}gyro_mean':gyro.mean(), f'{p}gyro_std':gyro.std(), f'{p}gyro_max':gyro.max(),
        f'{p}dyn_mean':dyn.mean(),   f'{p}dyn_max':dyn.max(),
        # Jerk — acceleration rate (positive) and deceleration (negative)
        f'{p}jerk_max':jmag.max(),   f'{p}jerk_mean':jmag.mean(),
        f'{p}jerk_x_max':np.abs(jx).max() if len(jx)>0 else 0,
        f'{p}jerk_y_max':np.abs(jy).max() if len(jy)>0 else 0,
        f'{p}jerk_z_max':np.abs(jz).max() if len(jz)>0 else 0,
        f'{p}jerk_min_dyn':jerk_min_dyn,   # peak deceleration
        f'{p}jerk_max_dyn':jerk_max_dyn,   # peak acceleration rate
        f'{p}decel_ratio':decel_ratio,      # >1 = deceleration dominated
        # Rise and fall time
        f'{p}rise_time_ms':rise_time_ms,
        f'{p}fall_time_ms':fall_time_ms,
        f'{p}symmetry':symmetry,
        # Impulse and rotation
        f'{p}impulse':impulse,
        f'{p}angle_x':ang_x, f'{p}angle_y':ang_y, f'{p}angle_z':ang_z,
        f'{p}total_angle':ang_x + ang_y + ang_z,
        # Orientation
        f'{p}qw_mean':qw.mean(), f'{p}qx_mean':qxv.mean(),
        f'{p}qy_mean':qyv.mean(),f'{p}qz_mean':qzv.mean(),
        f'{p}qw_std':qw.std(),   f'{p}qx_std':qxv.std(),
        f'{p}qy_std':qyv.std(),  f'{p}qz_std':qzv.std(),
        # Frequency domain
        f'{p}dom_freq':float(fr[np.argmax(fv)]) if len(fv)>0 else 0,
        f'{p}energy_low':float(np.sum(fv[(fr>=0)&(fr<5)]**2)),
        f'{p}energy_mid':float(np.sum(fv[(fr>=5)&(fr<15)]**2)),
        f'{p}energy_high':float(np.sum(fv[fr>=15]**2)),
    }

# ── Plot helpers ──────────────────────────────────────────────────────────────
def style_ax(ax, ylabel, xlabel=None, show_xticks=True):
    ax.set_facecolor(BG)
    ax.grid(True, alpha=0.2, color='#CCC', lw=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=8, color='#555')
    if xlabel: ax.set_xlabel(xlabel, fontsize=8, color='#555')
    ax.tick_params(labelsize=7, colors='#666')
    if not show_xticks: ax.set_xticklabels([])

def plot_session(fname, label, side, group, all_devs):
    n   = len(all_devs)
    fig = plt.figure(figsize=(14, 3*n+1), facecolor='white')
    fig.suptitle(f"{label} ({side})  ·  {fname}  ·  {group}",
                 fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(n, 2, hspace=0.35, wspace=0.25,
                           top=0.93, bottom=0.06, left=0.07, right=0.97)
    for row, (dev_id, d) in enumerate(sorted(all_devs.items())):
        color = DEVICE_COLORS.get(dev_id, '#888')
        t     = d['time_s'].values
        dur   = t[-1]
        peaks, thr = get_peaks(d)
        fs    = d['_fs'].iloc[0]

        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(t, d['accel_x'], '#E53935', lw=0.8, alpha=0.9, label='X')
        ax0.plot(t, d['accel_y'], '#43A047', lw=0.8, alpha=0.9, label='Y')
        ax0.plot(t, d['accel_z'], '#1E88E5', lw=0.8, alpha=0.9, label='Z')
        ax0.axhline(0, color='#BBB', ls='--', lw=0.5)
        for p in peaks:
            ax0.axvspan(max(0,t[p]-.15), min(dur,t[p]+.3),
                        color=color, alpha=0.07)
        ax0.set_title(f"D{dev_id} {DEVICE_NAMES.get(dev_id,'')}  "
                      f"{len(d)} rows · {dur:.1f}s · {fs:.0f}Hz",
                      fontsize=7.5)
        ax0.legend(fontsize=6, loc='upper right', ncol=3)
        style_ax(ax0, 'Accel (m/s²)', show_xticks=(row==n-1))

        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(t, d['accel_mag'], color=color, lw=1.0, zorder=3)
        ax1.fill_between(t, 9.81, d['accel_mag'].clip(lower=9.81),
                         color=color, alpha=0.12)
        ax1.axhline(9.81,     color='#9E9E9E', ls='--', lw=0.7,
                    label='Gravity (9.81 m/s²)')
        ax1.axhline(9.81+thr, color='#FF8F00', ls=':',  lw=0.8,
                    label=f'Threshold (+{thr:.1f})')
        if len(peaks) > 0:
            ax1.scatter(t[peaks], d['accel_mag'].values[peaks],
                        color='#D32F2F', s=35, zorder=5,
                        marker='v', linewidths=0,
                        label=f'{len(peaks)} events')
        ax1.set_title(f"D{dev_id} |a| max={d['accel_mag'].max():.1f} m/s²  "
                      f"peaks={len(peaks)}", fontsize=7.5)
        ax1.legend(fontsize=6, loc='upper right', ncol=2)
        style_ax(ax1, '|a| (m/s²)', show_xticks=(row==n-1))

    out = os.path.join(PLOT_DIR, f"{fname}_all_devices.png")
    plt.savefig(out, dpi=120, facecolor='white', bbox_inches='tight')
    plt.close()

def plot_overview(summaries):
    n = len(summaries)
    cols = 4; rows = int(np.ceil(n/cols))
    fig  = plt.figure(figsize=(6*cols, 4*rows+1), facecolor='white')
    fig.suptitle('All Sessions — Primary Device Acceleration Magnitude\n'
                 'Red ▼ = detected motion event',
                 fontsize=12, fontweight='bold', y=0.99)
    for idx, s in enumerate(summaries):
        ax    = fig.add_subplot(rows, cols, idx+1)
        d     = s['d']
        color = DEVICE_COLORS.get(s['dev'], '#888')
        t     = d['time_s'].values
        peaks, thr = get_peaks(d)
        ax.plot(t, d['accel_mag'], color=color, lw=0.9)
        ax.fill_between(t, 9.81, d['accel_mag'].clip(lower=9.81),
                        color=color, alpha=0.12)
        ax.axhline(9.81,     color='#9E9E9E', ls='--', lw=0.6)
        ax.axhline(9.81+thr, color='#FF8F00', ls=':',  lw=0.7)
        if len(peaks) > 0:
            ax.scatter(t[peaks], d['accel_mag'].values[peaks],
                       color='#D32F2F', s=25, zorder=5,
                       marker='v', linewidths=0)
        ax.set_title(f"{s['label']} {s['side']}\n"
                     f"{s['fname']}  D{s['dev']} "
                     f"{DEVICE_NAMES.get(s['dev'],'')}  "
                     f"{t[-1]:.0f}s  {len(peaks)} events",
                     fontsize=7.5, fontweight='bold')
        style_ax(ax, '|a| (m/s²)', xlabel='Time (s)')
    for i in range(n, rows*cols):
        fig.add_subplot(rows, cols, i+1).axis('off')
    plt.subplots_adjust(hspace=0.5, wspace=0.25, top=0.92, bottom=0.05)
    plt.savefig(os.path.join(PLOT_DIR, 'overview_all_sessions.png'),
                dpi=120, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  overview_all_sessions.png")

def plot_air_vs_bag(session_map):
    pairs = [('1a','2a','Jab Left'), ('1b','2b','Cross Right'),
             ('1c','2c','Hook Left'), ('1d','2d','Hook Right')]
    available = [(a,b,l) for a,b,l in pairs
                 if a in session_map and b in session_map]
    if not available: return
    fig, axes = plt.subplots(2, len(available),
                              figsize=(5*len(available), 8),
                              facecolor='white')
    fig.suptitle('Same Punch: Air (no contact) vs Bag (with impact)',
                 fontsize=12, fontweight='bold')
    for col, (ak, bk, label) in enumerate(available):
        for row, (key, tag) in enumerate([(ak,'Air'),(bk,'Bag')]):
            ax    = axes[row][col] if len(available)>1 else axes[row]
            d     = session_map[key]['d']
            dev   = session_map[key]['dev']
            color = DEVICE_COLORS.get(dev, '#888')
            t     = d['time_s'].values
            peaks, _ = get_peaks(d)
            ax.plot(t, d['accel_mag'], color=color, lw=0.9)
            ax.fill_between(t, 9.81, d['accel_mag'].clip(lower=9.81),
                            color=color, alpha=0.12)
            ax.axhline(9.81, color='#9E9E9E', ls='--', lw=0.7)
            if len(peaks) > 0:
                ax.scatter(t[peaks], d['accel_mag'].values[peaks],
                           color='#D32F2F', s=30, zorder=5,
                           marker='v', linewidths=0)
            ax.set_title(f"{label} — {tag}\n"
                         f"max={d['accel_mag'].max():.1f} m/s²  "
                         f"events={len(peaks)}",
                         fontsize=8,
                         fontweight='bold' if tag=='Bag' else 'normal')
            style_ax(ax, '|a| (m/s²)',
                     xlabel='Time (s)' if row==1 else None,
                     show_xticks=(row==1))
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'compare_air_vs_bag.png'),
                dpi=120, facecolor='white', bbox_inches='tight')
    plt.close()
    print("  compare_air_vs_bag.png")

def plot_lr_comparison(session_map):
    pairs = [('3a','3b','Low Kick'), ('3c','3d','Mid Kick'),
             ('3e','3f','High Kick'), ('4a','4b','Front Kick'),
             ('4c','4d','Side Kick')]
    for a, b, label in pairs:
        if a not in session_map or b not in session_map: continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor='white')
        fig.suptitle(f'Left vs Right: {label}',
                     fontsize=11, fontweight='bold')
        for col, (key, tag) in enumerate([(a,'Left'),(b,'Right')]):
            d     = session_map[key]['d']
            dev   = session_map[key]['dev']
            color = DEVICE_COLORS.get(dev, '#888')
            t     = d['time_s'].values
            peaks, _ = get_peaks(d)
            ax = axes[col]
            ax.plot(t, d['accel_mag'], color=color, lw=1.0)
            ax.fill_between(t, 9.81, d['accel_mag'].clip(lower=9.81),
                            color=color, alpha=0.12)
            ax.axhline(9.81, color='#9E9E9E', ls='--', lw=0.7)
            if len(peaks) > 0:
                ax.scatter(t[peaks], d['accel_mag'].values[peaks],
                           color='#D32F2F', s=35, zorder=5,
                           marker='v', linewidths=0)
            ax.set_title(f"{tag}  D{dev} {DEVICE_NAMES.get(dev,'')}  "
                         f"max={d['accel_mag'].max():.1f}  peaks={len(peaks)}",
                         fontsize=8, fontweight='bold')
            style_ax(ax, '|a| (m/s²)', xlabel='Time (s)')
        plt.tight_layout()
        safe = label.replace(' ', '_')
        plt.savefig(os.path.join(PLOT_DIR, f'compare_{safe}_LR.png'),
                    dpi=120, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"  compare_{safe}_LR.png")

# ── ML pipeline ───────────────────────────────────────────────────────────────
def build_models():
    return {
        'Random Forest':      Pipeline([('s', StandardScaler()),
            ('c', RandomForestClassifier(n_estimators=200, random_state=42,
                                          class_weight='balanced'))]),
        'SVM (RBF)':          Pipeline([('s', StandardScaler()),
            ('c', SVC(kernel='rbf', C=10, gamma='scale',
                      class_weight='balanced', random_state=42,
                      probability=True))]),
        'SVM (Linear)':       Pipeline([('s', StandardScaler()),
            ('c', SVC(kernel='linear', C=1,
                      class_weight='balanced', random_state=42,
                      probability=True))]),
        'Gradient Boosting':  Pipeline([('s', StandardScaler()),
            ('c', GradientBoostingClassifier(n_estimators=200, random_state=42,
                                              max_depth=4))]),
        'KNN (k=5)':          Pipeline([('s', StandardScaler()),
            ('c', KNeighborsClassifier(n_neighbors=5))]),
        'Logistic Regression':Pipeline([('s', StandardScaler()),
            ('c', LogisticRegression(max_iter=1000, class_weight='balanced',
                                      random_state=42))]),
    }

def run_task(name, y_raw, X_use, models, cv, log):
    le  = LabelEncoder()
    y   = le.fit_transform(y_raw)
    n   = len(le.classes_)
    baseline = 100.0 / n

    print(f"\n{'='*65}")
    print(f"{name}")
    print(f"Classes ({n}): {list(le.classes_)}")
    print(f"Windows: {len(y)}  |  Random baseline: {baseline:.1f}%")
    print(f"{'─'*65}")
    log.append(f"\n{'='*65}\n{name}")
    log.append(f"Classes: {list(le.classes_)}  Windows: {len(y)}  "
               f"Baseline: {baseline:.1f}%\n")

    results = {}
    best_acc = 0; best_name = ''; best_pred = None

    for mname, model in models.items():
        try:
            scores = cross_val_score(model, X_use, y, cv=cv, scoring='accuracy')
            y_pred = cross_val_predict(model, X_use, y, cv=cv)
            acc    = scores.mean()
            report = classification_report(y, y_pred,
                         target_names=le.classes_, zero_division=0)
            marker = ' ← BEST' if acc == max(
                cross_val_score(m, X_use, y, cv=cv,
                                scoring='accuracy').mean()
                for m in models.values()) else ''
            print(f"  {mname:<25}: {acc*100:.1f}% ± {scores.std()*100:.1f}%")
            log.append(f"  {mname}: {acc*100:.1f}% ± {scores.std()*100:.1f}%")
            log.append(report)
            results[mname] = {'acc': acc, 'pred': y_pred}
            if acc > best_acc:
                best_acc = acc; best_name = mname; best_pred = y_pred
        except Exception as e:
            print(f"  {mname}: error — {e}")

    print(f"\n  Best: {best_name} ({best_acc*100:.1f}%)")
    log.append(f"\n  Best: {best_name} ({best_acc*100:.1f}%)")

    # Confusion matrix for best model
    if best_pred is not None:
        fig, ax = plt.subplots(figsize=(max(5,n), max(4,n)),
                               facecolor='white')
        cm = confusion_matrix(y, best_pred)
        ax.imshow(cm, cmap='Blues', vmin=0)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(le.classes_, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(le.classes_, fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f"{name}\n"
                     f"Best: {best_name} — {best_acc*100:.1f}%  "
                     f"(random baseline: {baseline:.1f}%)",
                     fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, cm[i,j], ha='center', va='center',
                        fontsize=9,
                        color='white' if cm[i,j]>cm.max()/2 else 'black')
        plt.tight_layout()
        safe = name.replace(' ','_').replace('/','_').replace('(','').replace(')','')
        plt.savefig(os.path.join(ML_DIR, f'confusion_{safe}.png'),
                    dpi=120, facecolor='white', bbox_inches='tight')
        plt.close()

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    mnames  = list(results.keys())
    accs    = [results[m]['acc']*100 for m in mnames]
    colors  = ['#2196F3' if a==max(accs) else '#BBBBBB' for a in accs]
    bars    = ax.barh(mnames, accs, color=colors, edgecolor='white', height=0.6)
    ax.axvline(baseline, color='#E53935', ls='--', lw=1.5,
               label=f'Random baseline ({baseline:.1f}%)')
    ax.set_xlim(0, 105)
    ax.set_xlabel('Accuracy (%)', fontsize=10)
    ax.set_title(f'Model Comparison — {name}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=9, fontweight='bold')
    ax.set_facecolor(BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(ML_DIR, f'model_comparison_{safe}.png'),
                dpi=120, facecolor='white', bbox_inches='tight')
    plt.close()

    return le, best_acc, best_name, best_pred, results

# ── Full round prediction ─────────────────────────────────────────────────────
def predict_full_round(df_prim, fc_prim, imp):
    print(f"\n{'='*65}")
    print("FULL ROUND PREDICTION")
    print(f"{'='*65}")

    # Train on all labelled data
    X_fit = imp.transform(df_prim[fc_prim].values.astype(float))
    le    = LabelEncoder()
    y     = le.fit_transform(df_prim['label_side'])
    model = Pipeline([('s', StandardScaler()),
                      ('c', RandomForestClassifier(n_estimators=300,
                                                    random_state=42,
                                                    class_weight='balanced'))])
    model.fit(X_fit, y)
    print(f"Model trained: {len(y)} windows, {len(le.classes_)} classes")

    with open(os.path.join(ML_DIR, 'primary_model.pkl'), 'wb') as f:
        pickle.dump({'model':model, 'label_encoder':le,
                     'feature_cols':fc_prim, 'imputer':imp}, f)
    print("primary_model.pkl saved")

    for fr_name in VALIDATION:
        for candidate in [f'{fr_name}.csv',
                          f'{fr_name.replace("_2","__2_")}.csv']:
            path = os.path.join(OUT_DIR, candidate)
            if os.path.exists(path): break
        else:
            print(f"  {fr_name}: file not found"); continue

        df_fr = pd.read_csv(path)
        total_rows = len(df_fr)
        print(f"\nPredicting: {fr_name} ({total_rows} total rows)")
        all_preds = []

        for dev_id in [1, 2, 3, 4]:
            d = prep(df_fr[df_fr['device_id']==dev_id].copy())
            if d is None or len(d) < 10: continue
            peaks, _ = get_peaks(d)
            fs = d['_fs'].iloc[0]
            dur = d['time_s'].iloc[-1]
            print(f"  D{dev_id} {DEVICE_NAMES.get(dev_id,'')}: "
                  f"{len(d)} rows  {dur:.1f}s  {len(peaks)} events detected")

            for p in peaks:
                ms  = d['wall_clock_ms'].iloc[p]
                win = d[(d['wall_clock_ms']>=ms-PRE_MS) &
                        (d['wall_clock_ms']<=ms+POST_MS)]
                f   = extract(win, fs, prefix='p_')
                if f is None: continue
                fv  = np.array([f.get(c, 0.0) for c in fc_prim]).reshape(1,-1)
                fv  = imp.transform(fv)
                idx = model.predict(fv)[0]
                probs = model.predict_proba(fv)[0]
                conf  = float(probs.max())
                pred  = le.inverse_transform([idx])[0]
                top3_idx = probs.argsort()[-3:][::-1]
                top3 = [(le.inverse_transform([i])[0], round(float(probs[i]),3))
                        for i in top3_idx]
                all_preds.append({
                    'time_s':     round(float(d['time_s'].iloc[p]), 2),
                    'device':     dev_id,
                    'sensor':     DEVICE_NAMES.get(dev_id, ''),
                    'prediction': pred,
                    'confidence': round(conf, 3),
                    'peak_mag':   round(float(d['accel_mag'].iloc[p]), 1),
                    'top2':       top3[1][0] if len(top3)>1 else '',
                    'top2_conf':  top3[1][1] if len(top3)>1 else 0,
                })

        if not all_preds:
            print("  No predictions made"); continue

        pred_df = pd.DataFrame(all_preds).sort_values('time_s')
        pred_df.to_csv(os.path.join(ML_DIR,
                       f'predictions_{fr_name}.csv'), index=False)

        print(f"\n  Predicted distribution ({len(pred_df)} events total):")
        vc = pred_df['prediction'].value_counts()
        for motion, count in vc.items():
            pct = count/len(pred_df)*100
            print(f"    {motion:<25}: {count:3d}  ({pct:.1f}%)")

        hi = pred_df[pred_df['confidence'] >= 0.50]
        print(f"\n  High-confidence predictions (≥50%): {len(hi)}")
        if len(hi) > 0:
            print(hi[['time_s','sensor','prediction','confidence',
                       'peak_mag']].to_string(index=False))

        # Timeline plot
        labels_u = sorted(pred_df['prediction'].unique())
        cmap     = plt.cm.get_cmap('tab20', len(labels_u))
        lmap     = {l:i for i,l in enumerate(labels_u)}
        dur_total= pred_df['time_s'].max()

        fig, ax = plt.subplots(figsize=(16, 5), facecolor='white')
        for dev_id in [1,2,3,4]:
            sub = pred_df[pred_df['device']==dev_id]
            for _, row in sub.iterrows():
                c = cmap(lmap[row['prediction']])
                ax.scatter(row['time_s'], dev_id, color=c,
                           s=max(40, row['confidence']*300),
                           zorder=3, alpha=0.85,
                           edgecolors='white', linewidths=0.5)
        ax.set_yticks([1,2,3,4])
        ax.set_yticklabels([DEVICE_NAMES[i] for i in [1,2,3,4]], fontsize=9)
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_xlim(-1, dur_total+1)
        ax.set_title(f'{fr_name} ({total_rows} rows) — '
                     f'Predicted Motion Timeline\n'
                     f'Dot size = confidence  ·  '
                     f'Colour = predicted motion type  ·  '
                     f'{len(pred_df)} events classified',
                     fontsize=11, fontweight='bold')
        ax.set_facecolor(BG)
        ax.grid(True, alpha=0.2)
        handles = [plt.scatter([],[],color=cmap(lmap[l]),s=80,label=l)
                   for l in labels_u]
        ax.legend(handles=handles, fontsize=7, loc='upper right',
                  ncol=3, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(os.path.join(ML_DIR, f'timeline_{fr_name}.png'),
                    dpi=120, facecolor='white', bbox_inches='tight')
        plt.close()
        print(f"\n  timeline_{fr_name}.png saved")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("="*65)
    print("Complete Boxing Analysis — All Sessions")
    print("="*65)

    wins_prim  = []
    wins_multi = []
    summaries  = []
    session_map= {}

    # ── Load Session 3 ────────────────────────────────────────────────────────
    print("\nLoading Session 3...")
    for fname, info in S3.items():
        prim = load_s3(fname, info['primary'])
        if prim is None:
            print(f"  {fname}: not found — place CSV in script folder")
            continue

        all_devs = {}
        for did in [1,2,3,4]:
            d = load_s3(fname, did)
            if d is not None: all_devs[did] = d

        peaks, _ = get_peaks(prim)
        fs = prim['_fs'].iloc[0]
        print(f"  {fname} ({info['label']} {info['side']}): "
              f"{len(prim)} rows {prim['time_s'].iloc[-1]:.1f}s "
              f"{fs:.0f}Hz  max={prim['accel_mag'].max():.1f}  "
              f"peaks={len(peaks)}")

        summaries.append({'fname':fname, 'label':info['label'],
                          'side':info['side'], 'd':prim,
                          'dev':info['primary']})
        session_map[fname] = {'info':info, 'd':prim, 'dev':info['primary']}

        sec = {}
        for sid in info.get('secondary', []):
            sd = load_s3(fname, sid)
            if sd is not None: sec[sid] = sd

        n_win = 0
        for p in peaks:
            ms  = prim['wall_clock_ms'].iloc[p]
            wp  = prim[(prim['wall_clock_ms']>=ms-PRE_MS) &
                       (prim['wall_clock_ms']<=ms+POST_MS)]
            fp  = extract(wp, fs, prefix='p_')
            if fp is None: continue
            fp.update({'label':info['label'], 'side':info['side'],
                       'group':info['group'], 'bag':info['bag'],
                       'label_side':f"{info['label']} {info['side']}",
                       'limb_type':'Kick' if 'Kick' in info['label'] else 'Punch',
                       'contact':'Bag' if info['bag'] else 'Air',
                       'source':'S3'})
            wins_prim.append(fp)
            fm = fp.copy()
            for sid, sd in sec.items():
                ws = sd[(sd['wall_clock_ms']>=ms-PRE_MS) &
                        (sd['wall_clock_ms']<=ms+POST_MS)]
                f2 = extract(ws, sd['_fs'].iloc[0], prefix=f's{sid}_')
                if f2: fm.update(f2)
            wins_multi.append(fm)
            n_win += 1

        print(f"    → {n_win} windows extracted")
        plot_session(fname, info['label'], info['side'],
                     info['group'], all_devs)

    # ── Load Session 2 ────────────────────────────────────────────────────────
    print("\nLoading Session 2 (column remapping applied)...")
    for fname, info in S2.items():
        prim = load_s2(fname, info['primary'])
        if prim is None:
            print(f"  {fname}: not found — copy to script folder")
            continue
        peaks, _ = get_peaks(prim)
        fs = prim['_fs'].iloc[0]
        print(f"  {fname} ({info['label']} {info['side']}): "
              f"{len(prim)} rows {prim['time_s'].iloc[-1]:.1f}s  "
              f"peaks={len(peaks)}")
        n_win = 0
        for p in peaks:
            ms = prim['wall_clock_ms'].iloc[p]
            wp = prim[(prim['wall_clock_ms']>=ms-PRE_MS) &
                      (prim['wall_clock_ms']<=ms+POST_MS)]
            fp = extract(wp, fs, prefix='p_')
            if fp is None: continue
            fp.update({'label':info['label'], 'side':info['side'],
                       'group':info['group'], 'bag':info['bag'],
                       'label_side':f"{info['label']} {info['side']}",
                       'limb_type':'Punch', 'contact':'Air', 'source':'S2'})
            wins_prim.append(fp)
            wins_multi.append(fp.copy())
            n_win += 1
        print(f"    → {n_win} windows extracted")

    if not wins_prim:
        print("\nNo data loaded. Place all CSV files in the script folder.")
        return

    df_prim  = pd.DataFrame(wins_prim)
    df_multi = pd.DataFrame(wins_multi)
    drop     = ['label','side','group','bag','label_side',
                'limb_type','contact','source']
    fc_prim  = [c for c in df_prim.columns  if c not in drop]
    fc_multi = [c for c in df_multi.columns if c not in drop]

    # Imputers
    imp_p = SimpleImputer(strategy='mean')
    imp_m = SimpleImputer(strategy='mean')
    X_p   = imp_p.fit_transform(df_prim[fc_prim].values.astype(float))
    X_m   = imp_m.fit_transform(df_multi[fc_multi].values.astype(float))

    print(f"\n{'='*65}")
    print(f"DATASET: {len(df_prim)} windows")
    print(f"  Primary features: {len(fc_prim)}")
    print(f"  Multi-sensor features: {len(fc_multi)}")
    print(df_prim['label_side'].value_counts().to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\nGenerating signal plots...")
    plot_overview(summaries)
    plot_air_vs_bag(session_map)
    plot_lr_comparison(session_map)

    # ── ML ────────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("MACHINE LEARNING — MODEL COMPARISON")
    print(f"{'='*65}")

    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    log  = ["Boxing ML Results\n" + "="*65]
    models = build_models()

    # Task 1 — Left vs Right
    run_task("Task 1: Left vs Right Side",
             df_prim['side'], X_p, models, cv, log)

    # Task 2 — Punch vs Kick
    run_task("Task 2: Punch vs Kick",
             df_prim['limb_type'], X_p, models, cv, log)

    # Task 3 — Punch type
    mask_p = df_prim['limb_type']=='Punch'
    if mask_p.sum() >= 20:
        run_task("Task 3: Punch Type (Jab / Cross / Hook / Uppercut)",
                 df_prim.loc[mask_p,'label'], X_p[mask_p],
                 models, cv, log)

    # Task 4 — Air vs Bag
    if mask_p.sum() >= 20:
        run_task("Task 4: Air vs Bag Contact",
                 df_prim.loc[mask_p,'contact'], X_p[mask_p],
                 models, cv, log)

    # Task 5 — Kick type
    mask_k = df_prim['limb_type']=='Kick'
    if mask_k.sum() >= 20:
        run_task("Task 5: Kick Type (Low / Mid / High / Front / Side)",
                 df_prim.loc[mask_k,'label'], X_p[mask_k],
                 models, cv, log)

    # Task 6 — All motions (16 classes)
    run_task("Task 6: All Motions — 16 classes (primary sensor only)",
             df_prim['label_side'], X_p, models, cv, log)

    # Task 7 — All motions with multi-sensor
    print(f"\n{'='*65}")
    print("MULTI-SENSOR MODEL (primary + secondary sensors, "
          f"{len(fc_multi)} features)")
    print(f"{'='*65}")
    run_task("Task 7: All Motions — multi-sensor (thesis accuracy)",
             df_multi['label_side'], X_m,
             {'Random Forest': models['Random Forest'],
              'SVM (RBF)': models['SVM (RBF)']},
             cv, log)

    # Save log
    df_prim.to_csv(os.path.join(ML_DIR, 'all_windows.csv'), index=False)
    with open(os.path.join(ML_DIR, 'ml_results.txt'), 'w') as f:
        f.write('\n'.join(log))

    # ── Full round prediction ─────────────────────────────────────────────────
    predict_full_round(df_prim, fc_prim, imp_p)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}\nDONE\n{'='*65}")
    print(f"Plots → {PLOT_DIR}")
    print(f"ML    → {ML_DIR}")
    print("\nKey output files:")
    for folder in [PLOT_DIR, ML_DIR]:
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(('.png','.csv','.txt','.pkl')):
                print(f"  {fn}")

if __name__ == '__main__':
    main()
