"""Matched Burgers/KS paper figures from verified saved validation fields."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np

LABELS = {'fno': 'FNO', 'afno': 'AFNO'}
CASES = {'burgers': 'Burgers', 'ks': 'Kuramoto–Sivashinsky (KS)'}
COLORS = {'fno': '#646464', 'afno': '#007F86'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'mathtext.fontset': 'dejavuserif', 'font.size': 9, 'axes.titlesize': 10,
        'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 9, 'axes.linewidth': .65, 'lines.linewidth': 1.7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 400,
        'figure.facecolor': 'white'})


def values(bundle, sample):
    truth = bundle['fno']['target'][sample, :, 0]
    fno, afno = [bundle[n]['prediction'][sample, :, 0] for n in LABELS]
    # Error arithmetic is float64, matching the population metrics.
    errors = [np.abs(p.astype(np.float64)-truth.astype(np.float64)) for p in (fno, afno)]
    vmax = float(max(np.abs(v).max() for v in (truth, fno, afno)))
    emax = float(max(v.max() for v in errors))
    return (truth, fno, afno, *errors), Normalize(-vmax, vmax), Normalize(0, emax)


def heatmap(ax, data, bundle, norm, error, case, letter, show_y=False):
    x, t = bundle['fno']['x'], bundle['fno']['time']
    dx, dt = x[1]-x[0], t[1]-t[0]
    length = x[-1]+dx
    # Repeat the periodic endpoint for display only, retaining exact sample centers.
    displayed = np.concatenate((data, data[:, :1]), axis=1)
    image = ax.imshow(displayed, origin='lower', aspect='auto', interpolation='nearest',
        extent=(-dx/2, length+dx/2, t[0]-dt/2, t[-1]+dt/2),
        cmap='magma' if error else 'RdBu_r', norm=norm, rasterized=True)
    ax.set_xlim(0, length)
    if case == 'burgers':
        ax.set_xticks([0, np.pi, 2*np.pi], ['0', r'$\pi$', r'$2\pi$'])
    else:
        ax.set_xticks([0, 32, 64])
    ax.set_yticks(np.linspace(t[-1]/5, t[-1], 5))
    ax.set_xlabel(r'Position, $x$')
    ax.tick_params(labelleft=show_y)
    ax.text(.045, .95, f'({letter})', transform=ax.transAxes, va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=1.8))
    return image


def colorbar(fig, image, ax, label):
    bar = fig.colorbar(image, cax=ax)
    bar.ax.set_title(label, fontsize=9, pad=7)
    bar.locator = MaxNLocator(5)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=7.5, length=2.5)
    bar.outline.set_linewidth(.5)


def combined(bundles, sample):
    fig = plt.figure(figsize=(13.8, 6.2), layout='constrained')
    grid = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, .055, 1, 1, .055], hspace=.14)
    scales = {}
    for row, case in enumerate(CASES):
        bundle = bundles[case]
        data, fnorm, enorm = values(bundle, sample)
        for col, (slot, title) in enumerate(zip([0, 1, 2, 4, 5],
                ['Ground truth', 'FNO prediction', 'AFNO prediction', 'FNO absolute error', 'AFNO absolute error'])):
            ax = fig.add_subplot(grid[row, slot])
            image = heatmap(ax, data[col], bundle, enorm if col >= 3 else fnorm,
                            col >= 3, case, chr(97+5*row+col), col == 0)
            ax.set_title(title, pad=9)
            if col == 0: ax.set_ylabel(CASES[case]+'\n'+r'Time, $t$')
            if col == 2: colorbar(fig, image, fig.add_subplot(grid[row, 3]), r'$u(x,t)$')
            if col == 4: colorbar(fig, image, fig.add_subplot(grid[row, 6]), r'$|\hat{u}-u|$')
        scales[case] = {'field_min': fnorm.vmin, 'field_max': fnorm.vmax,
                        'error_min': enorm.vmin, 'error_max': enorm.vmax}
    fig.suptitle(f'100-step forecasts · validation trajectory {sample}', fontsize=12)
    return fig, scales


def individual(case, bundle, sample):
    fig = plt.figure(figsize=(8.2, 5.1), layout='constrained')
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, .055, 1, .055])
    data, fnorm, enorm = values(bundle, sample)
    for row, name in enumerate(LABELS):
        for col, slot in enumerate([0, 1, 3]):
            value = (data[0], data[row+1], data[row+3])[col]
            ax = fig.add_subplot(grid[row, slot])
            image = heatmap(ax, value, bundle, enorm if col == 2 else fnorm,
                            col == 2, case, chr(97+3*row+col), col == 0)
            if col == 0: ax.set_ylabel(LABELS[name]+'\n'+r'Time, $t$')
            if row == 0: ax.set_title(['Ground truth', 'Prediction', 'Absolute error'][col], pad=8)
            if row == 1 and col == 1: colorbar(fig, image, fig.add_subplot(grid[:, 2]), r'$u(x,t)$')
            if row == 1 and col == 2: colorbar(fig, image, fig.add_subplot(grid[:, 4]), r'$|\hat{u}-u|$')
    fig.suptitle(f'{CASES[case]}: 100-step forecast · validation trajectory {sample}', fontsize=11)
    return fig


def population(bundles):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), layout='constrained')
    steps = np.arange(1, 101)
    for row, (case, bundle) in enumerate(bundles.items()):
        for name, a in bundle.items():
            mean = a['mse'].mean(0)
            kw = dict(color=COLORS[name], label=LABELS[name], linestyle='--' if name == 'fno' else '-')
            axes[row, 0].plot(steps[:5], mean[:5], marker='o', markersize=3, **kw)
            axes[row, 1].semilogy(steps, mean, **kw)
        for col in range(2):
            axes[row, col].set_xlabel('Prediction step')
            axes[row, col].grid(alpha=.16, linewidth=.6)
            axes[row, col].spines[['right', 'top']].set_visible(False)
            axes[row, col].set_title(f'({chr(97+row*2+col)}) '+CASES[case] +
                                      (' · first five' if col == 0 else ' · full 100'), fontsize=9.5)
        axes[row, 0].set_ylabel('Mean squared error')
        axes[row, 1].set_ylabel('Mean squared error (log scale)')
        axes[row, 0].set_xticks(range(1, 6))
        axes[row, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes[row, 0].legend(frameon=False)
        axes[row, 1].set_xlim(1, 100)
    fig.suptitle('Prediction error across all 100 validation trajectories', fontsize=11)
    return fig


def statistics(bundles):
    result = {}
    for case, bundle in bundles.items():
        f, a = [bundle[n]['mse'] for n in LABELS]
        result[case] = {'metrics': {LABELS[n]: {'first5_mse': float(v['mse'][:, :5].mean()),
                      'full100_mse': float(v['mse'].mean())} for n, v in bundle.items()},
            'afno_lower_counts': {'first5': int((a[:, :5].mean(1) < f[:, :5].mean(1)).sum()),
                                  'full100': int((a.mean(1) < f.mean(1)).sum()),
                                  'both': int(((a[:, :5].mean(1) < f[:, :5].mean(1)) & (a.mean(1) < f.mean(1))).sum()),
                                  'steps': int((a.mean(0) < f.mean(0)).sum())}}
    return result


def save(fig, root, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(root/f'{name}.{suffix}', dpi=400, facecolor='white')
    plt.close(fig)


def load_bundles():
    """Load fixed trajectory-zero fields and the separate 100-trajectory errors."""
    from .demo import load_sample, verify_assets
    from .workflow import ASSETS
    verify_assets()
    bundles = {}
    for case in CASES:
        sample = load_sample(case)
        with np.load(ASSETS / 'examples' / f'{case}_population.npz', allow_pickle=False) as a:
            population_mse = {name: a[name + '_mse'] for name in LABELS}
        bundles[case] = {}
        for name in LABELS:
            entry = {k: sample[k] for k in ('initial', 'target', 'indices', 'time', 'x')}
            # `mse` describes ALL validation trajectories; `prediction` contains sample zero only.
            entry.update(prediction=sample[name + '_prediction'], mse=population_mse[name])
            bundles[case][name] = entry
    return bundles


def render(output, bundles=None, include_population=True):
    """Write PNG/PDF comparisons; never choose samples or rescale errors by model."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if bundles is None:
        bundles = load_bundles()
    style()
    scales = {}
    if set(bundles) == set(CASES):
        figure, scales = combined(bundles, 0)
        save(figure, root, 'burgers_ks_comparison')
    for case, bundle in bundles.items():
        save(individual(case, bundle, 0), root, case + '_comparison')
    if include_population:
        save(population(bundles), root, 'burgers_ks_error_curves')
    write(root / 'figure_metadata.json', {
        'labels': LABELS, 'split': 'val', 'sample_index': 0, 'scales': scales,
        'population_curves_included': include_population,
        'statistics': statistics(bundles),
        'statistics_trajectories': 100 if include_population else 1,
        'training': 'FNO: 500 epochs. AFNO: 500 base + 60 warmup + 40 balanced epochs.',
        'evidence': 'Single-seed validation results; unequal training compute.'})
    print(f'Figures written to {root.resolve()}', flush=True)
