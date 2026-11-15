"""Rebuild the ten figures of the paper from the released result files.

Usage:
    python analysis/make_figures.py            # all figures at 600 dpi
    python analysis/make_figures.py 4 8        # only Fig. 4 and Fig. 8
    python analysis/make_figures.py --dpi 200  # faster, smaller files

Figures are written to ``figures/``. Fig. 1 (framework) and Fig. 2 (fault
schematic) are drawings and do not use the result files. Fig. 2 applies
illustrative fault operators to a synthetic trace; it is not the injector
used in the benchmark run.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import dqtb
from dqtb import MODELS, NAME, ORDER

DPI = 600

# Use the first sans-serif font that is installed, so Matplotlib does not warn about missing ones.
_installed = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
FONT = next((f for f in ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans') if f in _installed), 'sans-serif')

plt.rcParams.update({
    'font.family': FONT, 'font.size': 8, 'axes.titlesize': 8.5,
    'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'xtick.major.size': 3, 'ytick.major.size': 3,
    'axes.edgecolor': '#444444', 'text.color': '#1a1a1a', 'axes.labelcolor': '#1a1a1a',
    'xtick.color': '#444444', 'ytick.color': '#444444', 'figure.dpi': 150,
    'mathtext.fontset': 'dejavusans', 'axes.grid': False, 'legend.frameon': False,
})
INK = '#1a1a1a'; MUTED = '#6b6b6b'; GRID = '#e3e3e3'
# Architecture colours (colour-blind safe); markers repeat the encoding.
COL = {'LSTM': '#0072B2', 'CNN': '#D55E00', 'Transformer': '#009E73'}
MRK = {'LSTM': 'o', 'CNN': 's', 'Transformer': '^'}
SHORT = dict(NAME); SHORT['channel_desync'] = 'Channel desync.'
SEQ = LinearSegmentedColormap.from_list('seq', ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'])
SEQ_R = LinearSegmentedColormap.from_list('seqr', ['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#f7fbff'])
DIV = LinearSegmentedColormap.from_list('div', ['#1b7837', '#a6dba0', '#f2f2f2', '#f4a582', '#b2182b'])

sw, bl, m = dqtb.load_all()


def panel_label(ax, s, x=-0.02, y=1.06):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom', ha='right', color=INK)


def save(fig, name):
    dqtb.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(dqtb.FIGURES / (name + '.png'), dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote figures/' + name + '.png')


# ---------------------------------------------------------------- Fig 1 framework
def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 4.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 70); ax.axis('off')
    bands = [(54.0, 15.0, 'Physical space', '#eef4fa'), (36.5, 16.0, 'Data-quality layer', '#fdf1ea'),
             (18.5, 16.5, 'Virtual space', '#eef7f2'), (1.0, 16.0, 'Evaluation layer', '#f4f1f8')]
    for y, h, lab, c in bands:
        ax.add_patch(FancyBboxPatch((0.6, y), 98.8, h, boxstyle='round,pad=0,rounding_size=1.2', fc=c, ec='none'))
        ax.text(2.0, y + h - 1.3, lab.upper(), fontsize=6.6, fontweight='bold', color=MUTED, va='top')

    def box(x, y, w, h, title, lines, ec, fc='white'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.0', fc=fc, ec=ec, lw=1.1))
        n = len(lines)
        top = y + h / 2 + (n * 2.3) / 2
        ax.text(x + w / 2, top, title, ha='center', va='center', fontsize=7.7, fontweight='bold', color=INK)
        for i, l in enumerate(lines):
            ax.text(x + w / 2, top - 2.75 - i * 2.45, l, ha='center', va='center', fontsize=6.7, color='#333333')

    def arrow(p, q, color='#444444', ls='-', lw=1.1):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=9, lw=lw, color=color, linestyle=ls,
                                     shrinkA=0, shrinkB=0))

    def line(xs, ys, color='#444444'):
        ax.plot(xs, ys, color=color, lw=1.1, solid_capstyle='butt')

    B, O, G, P, E = '#0072B2', '#D55E00', '#009E73', '#7b5ea7', '#8a6d1d'
    W = 26; X = [8, 37, 66]
    # physical (boxes y 55.2 .. 63.6)
    box(X[0], 55.2, W, 8.6, 'Turbofan fleet', ['C-MAPSS FD001 to FD004', '707 test engines'], B)
    box(X[1], 55.2, W, 8.6, 'Multi-source telemetry', ['21 sensors', '3 operating settings'], B)
    box(X[2], 55.2, W, 8.6, 'Quality-assurance gate', ['missing values, duplicates,', 'constant channels'], '#777777')
    arrow((X[0] + W, 59.5), (X[1], 59.5)); arrow((X[1] + W, 59.5), (X[2], 59.5))
    # data-quality (boxes y 38.2 .. 47.8)
    box(X[0], 38.2, 36, 9.6, 'Preprocessing', ['14 channels, RUL cap 125 cycles', 'regime-wise z-score, window W = 30'], G)
    box(56, 38.2, 36, 9.6, 'Fault injector (inference only)', ['12 fault types x 5 severities', 'x 10 injection seeds'], O, fc='#fff7f2')
    line([79, 79], [55.2, 51.2]); line([26, 79], [51.2, 51.2]); arrow((26, 51.2), (26, 47.8))
    arrow((44, 43.0), (56, 43.0), color=O)
    ax.text(50, 44.0, 'test windows', ha='center', va='bottom', fontsize=6.2, color=O)
    # virtual (boxes y 20.5 .. 30.3)
    box(X[0], 20.5, W, 9.8, 'Surrogate models', ['LSTM, 1D-CNN, Transformer', '5 training seeds, 60 models'], P)
    box(X[1], 20.5, W, 9.8, 'Monte Carlo dropout', ['30 stochastic passes', 'mean and standard deviation'], P)
    box(X[2], 20.5, W, 9.8, 'Conformal calibration', ['split-conformal quantile from', 'validation engines, 95% interval'], P)
    arrow((20, 38.2), (20, 30.3))
    ax.text(21.0, 35.75, 'training windows', ha='left', va='center', fontsize=6.2, color=MUTED)
    line([74, 74], [38.2, 34.2], color=O); line([30, 74], [34.2, 34.2], color=O)
    arrow((30, 34.2), (30, 30.3), color=O)
    ax.text(53, 34.9, 'degraded test windows', ha='center', va='bottom', fontsize=6.2, color=O)
    arrow((X[0] + W, 25.4), (X[1], 25.4)); arrow((X[1] + W, 25.4), (X[2], 25.4))
    # evaluation (boxes y 3.0 .. 12.4)
    box(X[0], 3.0, W, 9.4, 'Accuracy', ['RMSE, MAE, PHM08 score', 'robustness index (CRI)'], E)
    box(X[1], 3.0, W, 9.4, 'Uncertainty', ['interval coverage (PICP)', 'interval width (MPIW)'], E)
    box(X[2], 3.0, W, 9.4, 'Maintenance decision', ['alarm threshold 30 cycles', 'false alarms, misses, cost'], E)
    line([79, 79], [20.5, 15.6]); line([21, 79], [15.6, 15.6])
    for x in (21, 50, 79):
        arrow((x, 15.6), (x, 12.4))
    save(fig, 'fig01_framework')


# ---------------------------------------------------------------- Fig 2 fault schematic
def fig2():
    rng = np.random.default_rng(7)
    t = np.arange(30)
    clean = 0.25 + 0.045 * t + 0.0016 * t ** 2 + rng.normal(0, 0.10, 30)

    def ffill(x, mask):
        y = x.copy(); last = x[0]
        for i in range(len(x)):
            if mask[i] and i > 0:
                y[i] = last
            else:
                last = y[i]
        return y

    out = {}
    out['drift'] = clean + np.linspace(0, 1.0, 30)
    order_desc = np.argsort(-clean)
    mk = np.zeros(30, bool); mk[order_desc[:11]] = True
    out['missing_mnar'] = ffill(clean, mk)
    out['concept_drift'] = 1.3 * clean + 0.25
    ref = np.sin(t / 3.0) ; mk = ref > 0.35
    out['missing_mar'] = ffill(clean, mk)
    out['bias_offset'] = clean + 0.6
    out['sensor_dropout'] = np.zeros(30)
    out['nongaussian_noise'] = clean + 0.2 * rng.standard_t(3, 30)
    y = clean.copy()
    for i in range(30):
        y[i] = clean[(i // 4) * 4]
    out['sampling_reduction'] = y
    out['gaussian_noise'] = clean + rng.normal(0, 0.25, 30)
    mk = rng.random(30) < 0.35
    out['missing_mcar'] = ffill(clean, mk)
    y = np.empty(30); y[4:] = clean[:-4]; y[:4] = clean[0]
    out['channel_desync'] = y
    y = clean.copy()
    for i in rng.choice(np.arange(0, 28, 2), 8, replace=False):
        y[i], y[i + 1] = y[i + 1], y[i]
    out['timestamp_jitter'] = y
    fam = {'drift': 'value', 'missing_mnar': 'completeness', 'concept_drift': 'context', 'missing_mar': 'completeness',
           'bias_offset': 'value', 'sensor_dropout': 'availability', 'nongaussian_noise': 'value',
           'sampling_reduction': 'timing', 'gaussian_noise': 'value', 'missing_mcar': 'completeness',
           'channel_desync': 'timing', 'timestamp_jitter': 'timing'}
    layout = ['gaussian_noise', 'nongaussian_noise', 'bias_offset', 'drift', 'missing_mcar', 'missing_mar',
              'missing_mnar', 'sensor_dropout', 'timestamp_jitter', 'channel_desync', 'sampling_reduction',
              'concept_drift']
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 4.3), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), layout):
        ax.plot(t, clean, color='#9a9a9a', lw=1.0, label='clean reading')
        ax.plot(t, out[k], color='#D55E00', lw=1.2, label='degraded reading')
        ax.set_title(SHORT[k], fontsize=8, pad=3, loc='left', fontweight='bold')
        ax.text(0.04, 0.95, fam[k], transform=ax.transAxes, ha='left', va='top', fontsize=6.8, color=MUTED,
                style='italic')
        ax.set_ylim(-0.4, 4.0); ax.set_xticks([0, 10, 20, 29]); ax.set_xticklabels(['1', '11', '21', '30'])
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('Cycle within window')
    for ax in axes[:, 0]:
        ax.set_ylabel('Normalised reading')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.03), fontsize=8)
    fig.tight_layout(h_pad=0.9, w_pad=0.8)
    save(fig, 'fig02_fault_schematic')


# ---------------------------------------------------------------- Fig 3 clean baselines
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75))
    subs = ['FD001', 'FD002', 'FD003', 'FD004']
    for ax, col, yl in zip(axes, ['RMSE', 'PICP'], ['Clean-input RMSE (cycles)', 'Clean-input coverage of 95% intervals']):
        for j, mod in enumerate(MODELS):
            for i, s in enumerate(subs):
                v = bl[(bl.subset == s) & (bl.model == mod)][col].values
                x = i + (j - 1) * 0.24
                ax.scatter(np.full(len(v), x), v, s=9, color=COL[mod], alpha=0.35, lw=0, zorder=2)
                ax.errorbar(x, v.mean(), yerr=v.std(ddof=1), fmt=MRK[mod], color=COL[mod], ms=5, capsize=2.5, lw=1.0,
                            mec='white', mew=0.6, zorder=3, label=mod if i == 0 else None)
        ax.set_xticks(range(4)); ax.set_xticklabels(subs); ax.set_ylabel(yl)
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylim(12, 19)
    axes[1].axhline(0.95, color=INK, lw=0.8, ls=(0, (4, 2)))
    axes[1].set_ylim(0.82, 1.0)
    axes[0].legend(loc='upper left', ncol=3, handletextpad=0.2, columnspacing=1.0, borderaxespad=0.1)
    panel_label(axes[0], '(a)'); panel_label(axes[1], '(b)')
    fig.tight_layout(w_pad=2.0)
    save(fig, 'fig03_clean_baselines')


# ---------------------------------------------------------------- Fig 4 propagation map
def fig4():
    cri = dqtb.cri(m, ['corruption', 'severity']).groupby(['corruption', 'severity']).mean().unstack().loc[ORDER]
    picp = m.groupby(['corruption', 'severity']).PICP.mean().unstack().loc[ORDER]
    cost = (m.groupby(['corruption', 'severity']).MaintCost.mean().unstack().loc[ORDER]) / bl.MaintCost.mean()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.55), gridspec_kw={'wspace': 0.10})
    specs = [(cri, SEQ, dict(vmin=1, vmax=4.7), '{:.2f}', 'RMSE ratio to clean input', lambda v: v > 2.9),
             (picp, SEQ_R, dict(vmin=0.05, vmax=0.95), '{:.2f}', 'Coverage of 95% intervals', lambda v: v < 0.45),
             (cost, DIV, dict(norm=TwoSlopeNorm(vmin=np.log10(0.5), vcenter=0, vmax=np.log10(12))), '{:.1f}',
              'Cost ratio to clean input', lambda v: v > 6.5)]
    for k, (ax, (d, cmap, kw, fmt, title, dark)) in enumerate(zip(axes, specs)):
        arr = d.values
        show = np.log10(arr) if k == 2 else arr
        ax.imshow(show, cmap=cmap, aspect='auto', **kw)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                ax.text(j, i, fmt.format(arr[i, j]), ha='center', va='center', fontsize=6.6,
                        color='white' if dark(arr[i, j]) else INK)
        ax.set_xticks(range(5)); ax.set_xticklabels([1, 2, 3, 4, 5]); ax.set_xlabel('Fault severity')
        ax.set_yticks(range(12))
        ax.set_yticklabels([SHORT[c] for c in ORDER] if k == 0 else [])
        ax.set_title(title, fontsize=8, pad=5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True); ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
        ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
        ax.text(0.0, 1.085, '(' + 'abc'[k] + ')', transform=ax.transAxes, fontsize=9, fontweight='bold', ha='left')
    save(fig, 'fig04_propagation_map')


# ---------------------------------------------------------------- Fig 5 degradation small multiples
def fig5():
    c = dqtb.cri(m, ['corruption', 'severity'])
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.0), sharex=True, sharey=True)
    for ax, k in zip(axes.ravel(), ORDER):
        ax.axhline(1, color='#9a9a9a', lw=0.7, ls=(0, (3, 2)))
        for mod in MODELS:
            d = c.loc[k].xs(mod, level='model').unstack('subset')
            mean = d.mean(axis=1)
            ax.fill_between(d.index, d.min(axis=1), d.max(axis=1), color=COL[mod], alpha=0.13, lw=0)
            ax.plot(d.index, mean, color=COL[mod], lw=1.3, marker=MRK[mod], ms=3.6, mec='white', mew=0.5, label=mod)
        ax.set_title(SHORT[k], fontsize=8, pad=3, loc='left', fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0.7, 6.2); ax.set_yticks([1, 2, 3, 4, 5, 6])
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for ax in axes[-1]:
        ax.set_xlabel('Fault severity')
    for ax in axes[:, 0]:
        ax.set_ylabel('RMSE ratio to clean')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.025), fontsize=8)
    fig.tight_layout(h_pad=0.9, w_pad=0.7)
    save(fig, 'fig05_degradation_curves')


# ---------------------------------------------------------------- Fig 6 missingness mechanisms
def fig6():
    mech = [('missing_mcar', 'MCAR', '#0072B2', 'o'), ('missing_mar', 'MAR', '#009E73', '^'),
            ('missing_mnar', 'MNAR', '#D55E00', 's')]
    c = dqtb.cri(m, ['corruption', 'severity'])
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
    for k, name, col, mk in mech:
        d = c.loc[k].unstack(['model', 'subset'])
        axes[0].fill_between(d.index, d.min(axis=1), d.max(axis=1), color=col, alpha=0.14, lw=0)
        axes[0].plot(d.index, d.mean(axis=1), color=col, lw=1.4, marker=mk, ms=4, mec='white', mew=0.5, label=name)
        for ax, colname in ((axes[1], 'PICP'), (axes[2], 'MissRate')):
            d = m[m.corruption == k].groupby(['severity', 'model', 'subset'])[colname].mean().unstack(['model', 'subset'])
            ax.fill_between(d.index, d.min(axis=1), d.max(axis=1), color=col, alpha=0.14, lw=0)
            ax.plot(d.index, d.mean(axis=1), color=col, lw=1.4, marker=mk, ms=4, mec='white', mew=0.5, label=name)
    axes[0].axhline(1, color='#9a9a9a', lw=0.7, ls=(0, (3, 2)))
    axes[1].axhline(bl.PICP.mean(), color='#9a9a9a', lw=0.7, ls=(0, (3, 2)))
    axes[2].axhline(bl.MissRate.mean(), color='#9a9a9a', lw=0.7, ls=(0, (3, 2)))
    axes[0].set_ylabel('RMSE ratio to clean'); axes[1].set_ylabel('Coverage of 95% intervals')
    axes[2].set_ylabel('Missed-alarm rate')
    axes[1].set_ylim(0.4, 1.0); axes[2].set_ylim(-0.03, 1.05); axes[0].set_ylim(0.8, 4.4)
    for i, ax in enumerate(axes):
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xlabel('Fault severity')
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
        panel_label(ax, '(' + 'abc'[i] + ')', x=-0.12)
    axes[0].legend(loc='upper left', borderaxespad=0.2)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig06_missingness')


# ---------------------------------------------------------------- Fig 7 uncertainty
def fig7():
    c = dqtb.cri(m, ['corruption', 'severity']).groupby(['corruption', 'severity']).mean()
    p = m.groupby(['corruption', 'severity']).PICP.mean()
    w = m.groupby(['corruption', 'severity']).width_ratio.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.05), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axes[0]
    hi = {'drift': '#D55E00', 'missing_mnar': '#0072B2', 'concept_drift': '#009E73'}
    for k in ORDER:
        if k in hi:
            continue
        ax.plot(c.loc[k].values, p.loc[k].values, color='#b5b5b5', lw=0.8, marker='o', ms=2.8, mfc='#b5b5b5', mec='white',
                mew=0.3, zorder=2)
    mk = {'drift': 's', 'missing_mnar': 'o', 'concept_drift': '^'}
    for k, col in hi.items():
        ax.plot(c.loc[k].values, p.loc[k].values, color=col, lw=1.3, marker=mk[k], ms=4.2, mec='white', mew=0.5, zorder=3)
    ax.axhline(0.95, color=INK, lw=0.7, ls=(0, (4, 2)))
    ax.text(4.78, 0.962, 'nominal 0.95', ha='right', va='bottom', fontsize=6.8)
    ax.text(4.78, 0.17, 'Calibration drift', ha='right', va='bottom', fontsize=7.2, color=INK)
    ax.text(3.55, 0.665, 'MNAR missingness', ha='left', va='bottom', fontsize=7.2, color=INK)
    ax.text(2.20, 0.33, 'Regime shift', ha='right', va='top', fontsize=7.2, color=INK)
    ax.text(2.76, 0.50, 'other nine faults (grey)', ha='left', va='center', fontsize=7.0, color=MUTED)
    ax.set_xlabel('RMSE ratio to clean (severity 1 to 5 along each line)'); ax.set_ylabel('Coverage of 95% intervals')
    ax.set_xlim(0.9, 4.85); ax.set_ylim(0, 1.03)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    sel = [('drift', '#D55E00', 's', 'Calibration drift'), ('concept_drift', '#009E73', '^', 'Regime shift'),
           ('bias_offset', '#8a8a8a', 'D', 'Calibration bias'), ('missing_mnar', '#0072B2', 'o', 'MNAR missingness'),
           ('missing_mar', '#8a8a8a', 'v', 'MAR missingness'), ('sensor_dropout', '#8a8a8a', 'P', 'Sensor dropout')]
    ax.axhline(1, color='#9a9a9a', lw=0.7, ls=(0, (3, 2)))
    yoff = {'drift': 0.0, 'concept_drift': -0.02, 'bias_offset': 0.03, 'missing_mnar': 0.03, 'missing_mar': 0.035,
            'sensor_dropout': -0.045}
    for k, col, mkr, lab in sel:
        ax.plot([1, 2, 3, 4, 5], w.loc[k].values, color=col, lw=1.3 if col != '#8a8a8a' else 0.9, marker=mkr, ms=4,
                mec='white', mew=0.5)
        ax.text(5.12, w.loc[k].values[-1] + yoff[k], lab, ha='left', va='center', fontsize=7.0, color=INK)
    ax.set_xlim(0.8, 7.0); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_ylim(0.1, 1.55)
    ax.set_xlabel('Fault severity'); ax.set_ylabel('Interval width ratio to clean')
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.spines['bottom'].set_bounds(0.8, 5.2)
    panel_label(axes[0], '(a)', x=-0.08); panel_label(axes[1], '(b)', x=-0.10)
    fig.tight_layout(w_pad=1.6)
    save(fig, 'fig07_uncertainty')


# ---------------------------------------------------------------- Fig 8 decision plane
def fig8():
    far = m.groupby(['corruption', 'severity']).FalseAlarmRate.mean()
    mis = m.groupby(['corruption', 'severity']).MissRate.mean()
    cost5 = (m[m.severity == 5].groupby('corruption').MaintCost.mean() / bl.MaintCost.mean())
    f0, m0 = bl.FalseAlarmRate.mean(), bl.MissRate.mean()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.25), gridspec_kw={'width_ratios': [1.15, 1]})
    ax = axes[0]
    opt = {'missing_mnar': ('#0072B2', 'o'), 'missing_mar': ('#0072B2', '^'), 'sensor_dropout': ('#0072B2', 'P')}
    pes = {'drift': ('#D55E00', 's'), 'concept_drift': ('#D55E00', 'D'), 'bias_offset': ('#D55E00', 'v')}
    for k in ORDER:
        x = np.r_[f0, far.loc[k].values]; y = np.r_[m0, mis.loc[k].values]
        if k in opt or k in pes:
            col, mkr = (opt.get(k) or pes.get(k))
            ax.plot(x, y, color=col, lw=1.2, zorder=3)
            ax.scatter(x[1:], y[1:], s=[8, 12, 17, 23, 30], color=col, marker=mkr, edgecolor='white', lw=0.5, zorder=4)
        else:
            ax.plot(x, y, color='#b5b5b5', lw=0.8, zorder=2)
            ax.scatter(x[1:], y[1:], s=6, color='#b5b5b5', zorder=2, lw=0)
    ax.scatter([f0], [m0], s=42, color=INK, marker='*', zorder=6)
    ax.annotate('clean input', (f0, m0), xytext=(0.115, 0.30), fontsize=7, color=INK,
                arrowprops=dict(arrowstyle='-', lw=0.5, color=MUTED, shrinkA=1, shrinkB=3))
    ax.text(0.035, 1.012, 'MNAR missingness', fontsize=7.1, va='bottom', ha='left')
    ax.text(0.035, 0.872, 'MAR missingness', fontsize=7.1, va='center', ha='left')
    ax.text(0.035, 0.775, 'Sensor dropout', fontsize=7.1, va='center', ha='left')
    ax.text(0.80, 0.045, 'Calibration drift', fontsize=7.1, va='bottom', ha='center')
    ax.text(0.40, 0.03, 'Regime shift', fontsize=7.1, va='bottom', ha='center')
    ax.text(0.25, 0.118, 'Calibration bias', fontsize=7.1, va='center', ha='left')
    ax.text(0.50, 0.60, 'silent misses', fontsize=7.6, color='#0072B2', style='italic', ha='center')
    ax.annotate('', xy=(0.36, 0.92), xytext=(0.43, 0.66), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#0072B2'))
    ax.text(0.62, 0.42, 'false-alarm storms', fontsize=7.6, color='#D55E00', style='italic', ha='center')
    ax.annotate('', xy=(0.61, 0.04), xytext=(0.62, 0.39), arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#D55E00'))
    ax.set_xlim(-0.02, 0.92); ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel('False-alarm rate'); ax.set_ylabel('Missed-alarm rate')
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax = axes[1]
    order = cost5.sort_values().index
    cols = ['#0072B2' if k in opt else '#D55E00' if k in pes else '#b5b5b5' for k in order]
    ax.barh(range(12), cost5.loc[order].values, color=cols, height=0.62)
    for i, k in enumerate(order):
        ax.text(cost5.loc[k] + 0.2, i, f'{cost5.loc[k]:.1f}', va='center', fontsize=6.9, color=INK)
    ax.axvline(1, color=INK, lw=0.7, ls=(0, (3, 2)))
    ax.set_yticks(range(12)); ax.set_yticklabels([SHORT[k] for k in order], fontsize=7.1)
    ax.set_xlabel('Cost ratio to clean input at severity 5'); ax.set_xlim(0, 13.2)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    panel_label(axes[0], '(a)', x=-0.08); panel_label(axes[1], '(b)', x=-0.42)
    fig.tight_layout(w_pad=1.0)
    save(fig, 'fig08_decision_plane')


# ---------------------------------------------------------------- Fig 9 architecture profile
def fig9():
    allc = dqtb.cri(m, ['corruption']).groupby(['corruption', 'model']).mean().unstack()
    c5 = dqtb.cri(m[m.severity == 5], ['corruption']).groupby(['corruption', 'model']).mean().unstack()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5), sharey=True)
    for ax, d, title in zip(axes, [allc, c5], ['All severities pooled', 'Severity 5 only']):
        for i, k in enumerate(ORDER[::-1]):
            v = d.loc[k]
            ax.plot([v.min(), v.max()], [i, i], color='#c9c9c9', lw=1.6, zorder=1, solid_capstyle='round')
            for mod in MODELS:
                ax.scatter(v[mod], i, s=26, color=COL[mod], marker=MRK[mod], edgecolor='white', lw=0.5, zorder=3,
                           label=mod if i == 0 else None)
        ax.axvline(1, color='#9a9a9a', lw=0.7, ls=(0, (3, 2)))
        ax.set_yticks(range(12)); ax.set_yticklabels([SHORT[k] for k in ORDER[::-1]])
        ax.set_xlabel('RMSE ratio to clean input'); ax.set_title(title, fontsize=8.2, pad=4)
        ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
    axes[0].set_xlim(0.9, 3.2); axes[1].set_xlim(0.8, 5.6)
    axes[0].legend(loc='lower right', handletextpad=0.2)
    panel_label(axes[0], '(a)', x=-0.02); panel_label(axes[1], '(b)', x=-0.02)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'fig09_architecture')


# ---------------------------------------------------------------- Fig 10 channel identity
def fig10():
    cv = dqtb.injection_seed_cv(sw, severity=3)
    order = cv.groupby('corruption').cv.median().sort_values().index
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    data = [cv[cv.corruption == k].cv.values * 100 for k in order]
    hi = {'sensor_dropout', 'bias_offset'}
    style = dict(widths=0.58, patch_artist=True, showfliers=True,
                 flierprops=dict(marker='o', ms=2, mfc='#9a9a9a', mec='none'), medianprops=dict(color=INK, lw=1.0),
                 whiskerprops=dict(color='#666666', lw=0.7), capprops=dict(color='#666666', lw=0.7))
    try:
        bp = ax.boxplot(data, orientation='horizontal', **style)   # Matplotlib 3.10 and newer
    except TypeError:
        bp = ax.boxplot(data, vert=False, **style)
    for patch, k in zip(bp['boxes'], order):
        patch.set(facecolor='#D55E00' if k in hi else '#c9d6e3', edgecolor='#555555', lw=0.6)
    ax.set_yticks(range(1, 13)); ax.set_yticklabels([SHORT[k] for k in order])
    ax.set_xlabel('Coefficient of variation of RMSE across 10 injection seeds (%)')
    ax.xaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.tick_params(axis='y', length=0)
    fig.tight_layout()
    save(fig, 'fig10_channel_identity')


def main():
    global DPI
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('figures', nargs='*', help='figure numbers to build (default: all)')
    ap.add_argument('--dpi', type=int, default=600)
    args = ap.parse_args()
    DPI = args.dpi
    for n in args.figures or [str(i) for i in range(1, 11)]:
        globals()['fig' + n]()


if __name__ == '__main__':
    main()
