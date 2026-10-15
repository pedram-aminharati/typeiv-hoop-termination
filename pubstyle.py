"""Publication figure style shared by the three papers.

Follows Elsevier artwork guidance: final sizes of 90 mm (single column),
140 mm (1.5 column) and 190 mm (double column); sans-serif type of 7-8 pt
at final size; vector PDF with embedded TrueType fonts plus a 600 dpi PNG.
Colours: a colour-vision-deficiency-validated categorical set, always
combined with marker or line-style redundancy so that figures survive
greyscale printing.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

MM = 1/25.4
W1, W15, W2 = 90*MM, 140*MM, 190*MM

# validated categorical slots (fixed order) and neutral inks
C = dict(blue='#2a78d6', orange='#eb6834', aqua='#1baf7a', yellow='#eda100',
         magenta='#e87ba4', green='#008300')
SERIES = [C['blue'], C['orange'], C['aqua'], C['yellow'], C['magenta'], C['green']]
INK, INK2, MUTED, LIGHT = '#1a1a1a', '#4d4d4d', '#8c8c8c', '#e6e6e6'
MARKERS = ['o', 's', '^', 'D', 'v', 'P']
LINES = ['-', '--', '-.', ':', (0, (5, 1.5, 1, 1.5)), (0, (1, 1))]
ZONE = dict(barrel=C['blue'], junction=C['orange'], dome=C['aqua'], boss=C['yellow'])


def use():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'Liberation Sans', 'mathtext.it': 'Liberation Sans:italic',
        'mathtext.bf': 'Liberation Sans:bold', 'mathtext.fallback': 'stixsans',
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.6, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.top': True, 'ytick.right': True,
        'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
        'xtick.minor.size': 1.6, 'ytick.minor.size': 1.6,
        'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'xtick.minor.width': 0.45, 'ytick.minor.width': 0.45,
        'xtick.minor.visible': True, 'ytick.minor.visible': True,
        'axes.grid': False, 'axes.spines.top': True, 'axes.spines.right': True,
        'lines.linewidth': 1.1, 'lines.markersize': 3.6,
        'lines.markeredgewidth': 0.6,
        'legend.frameon': False, 'legend.handlelength': 2.0,
        'legend.borderaxespad': 0.4, 'legend.labelspacing': 0.3,
        'axes.prop_cycle': mpl.cycler(color=SERIES),
        'figure.dpi': 150, 'savefig.dpi': 600,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'axes.unicode_minus': True,
    })


def label(ax, text, x=-0.02, y=1.02):
    """panel label '(a)' placed above the top-left corner"""
    ax.text(x, y, f'({text})', transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, fontweight='bold')


def minor(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


def save(fig, path):
    fig.savefig(path + '.pdf')
    fig.savefig(path + '.png', dpi=600)
    plt.close(fig)
