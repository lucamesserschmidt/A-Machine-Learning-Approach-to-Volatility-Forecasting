from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ale import ale_1d


def plot_relative_mse(summary, out_path, title):
    if summary.empty:
        return
    s = summary.sort_values('relative_mse_vs_har')
    colors = ['#cc4444' if v > 1 else '#3a7d44' for v in s['relative_mse_vs_har']]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(s['model'], s['relative_mse_vs_har'], color=colors)
    ax.axhline(1.0, ls='--', lw=1, color='k')
    ax.set_ylabel('Relative MSE vs HAR'); ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ale_panel(model, X, features, out_path, title, n_bins=40):
    feats = [f for f in features if f in X.columns]
    if not feats:
        return
    ncol = min(2, len(feats)); nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 3.4 * nrow), squeeze=False)
    for i, f in enumerate(feats):
        ax = axes[i // ncol][i % ncol]
        xc, ale = ale_1d(model, X, f, n_bins=n_bins)
        ax.plot(xc, ale, color='#244d90', lw=1.8)
        ax.axhline(0, color='grey', lw=0.6)
        ax.set_title(f'ALE: {f}'); ax.set_xlabel(f); ax.set_ylabel('ALE on RV')
    for j in range(len(feats), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_ale_importance(imp, out_path, title, top_n=12):
    if imp.empty:
        return
    s = imp.head(top_n).sort_values('ale_vi')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(s['feature'], s['ale_vi'], color='#244d90')
    ax.set_xlabel('ALE variable importance (normalized)'); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_cross_sectional_heatmap(cs, out_path, data_source, feature_set):
    sub = cs[(cs['data_source'] == data_source) & (cs['feature_set'] == feature_set)]
    if sub.empty:
        return
    piv = sub.pivot(index='model', columns='horizon', values='mean_rel_mse')
    piv = piv.sort_values(piv.columns[0])
    fig, ax = plt.subplots(figsize=(6, 0.45 * len(piv) + 2))
    im = ax.imshow(piv.values, aspect='auto', cmap='RdYlGn_r', vmin=0.7, vmax=1.3)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f'h={h}' for h in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8)
    ax.set_title(f'Mean relative MSE vs HAR\n{data_source} / {feature_set}')
    fig.colorbar(im, ax=ax, shrink=0.7); fig.tight_layout()
    fig.savefig(out_path, dpi=150); plt.close(fig)
