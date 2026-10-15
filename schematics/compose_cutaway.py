import json, sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, '.')
import pubstyle as P
P.use()


def crop(fn, pad=12, box=None):
    im = np.asarray(Image.open(fn).convert('RGB'))
    if box is None:
        m = (im < 245).any(axis=2)
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min() - pad, ys.max() + pad, xs.min() - pad, xs.max() + pad
    else:
        h, w = im.shape[:2]
        x0, y0, x1, y1 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
    H, W = im.shape[:2]
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, W), min(y1, H)
    out = im[y0:y1, x0:x1]
    MAXW = 2500                                   # ~330 dpi at 190 mm
    if out.shape[1] > MAXW:
        f = MAXW/out.shape[1]
        out = np.asarray(Image.fromarray(out).resize((MAXW, int(out.shape[0]*f)), Image.LANCZOS))
    return out, (x0/W, y0/H, x1/W, y1/H)


def to_ax(a, cb):
    """anchor in full-image fraction -> cropped-image fraction (origin top-left)"""
    return ((a[0] - cb[0])/(cb[2] - cb[0]), (a[1] - cb[1])/(cb[3] - cb[1]))


def main():
    A1 = json.load(open('cutaway.json'))['anchors']
    A2 = json.load(open('cutaway_zoom.json'))['anchors']
    im1, cb1 = crop('cutaway.png')
    im2, cb2 = crop('cutaway_zoom.png', box=(0.0, 0.11, 1.0, 1.0))
    fig = plt.figure(figsize=(P.W2, 0.86*P.W2))
    ax1 = fig.add_axes([0.0, 0.50, 1.0, 0.46])
    ax2 = fig.add_axes([0.17, 0.0, 0.83, 0.47])
    for ax, im in ((ax1, im1), (ax2, im2)):
        ax.imshow(im, extent=(0, 1, 1, 0), aspect='auto', interpolation='lanczos')
        ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.axis('off')
    # keep the image aspect: adjust axes boxes
    for ax, im in ((ax1, im1), (ax2, im2)):
        h, w = im.shape[:2]
        bb = ax.get_position()
        figw, figh = fig.get_size_inches()
        want_h = bb.width*figw*h/w/figh
        if want_h <= bb.height:
            ax.set_position([bb.x0, bb.y0 + (bb.height - want_h)/2, bb.width, want_h])
        else:
            want_w = bb.height*figh*w/h/figw
            ax.set_position([bb.x0 + (bb.width - want_w)/2, bb.y0, want_w, bb.height])
    ink = P.INK
    lk = dict(arrowstyle='-', lw=0.5, color=ink, shrinkA=0, shrinkB=1.5)
    # ---- (a) labels
    def lab(ax, anc, txt, xy_text, ha='center', va='bottom', dot=True):
        x, y = anc
        ax.annotate(txt, xy=(x, y), xytext=xy_text, textcoords='data', ha=ha, va=va, fontsize=7,
                    arrowprops=lk, annotation_clip=False)
        if dot:
            ax.plot([x], [y], 'o', ms=2.0, color=ink, mec='white', mew=0.4, clip_on=False, zorder=5)
    a = {k: to_ax(v, cb1) for k, v in A1.items()}
    names = {'L1': 'polymer liner', 'L2': 'hoop', 'L3': r'helical $\pm\alpha_1$', 'L4': r'helical $\pm30^\circ$',
             'L5': 'hoop', 'L6': r'helical $\pm\alpha_1$', 'L7': r'helical $\pm30^\circ$ (outer)'}
    order = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7']
    xt = np.linspace(0.07, 0.56, 7)
    for k, xq, yq in zip(order, xt, [0.02, 0.10, 0.02, 0.10, 0.02, 0.10, 0.02]):
        lab(ax1, a[k], names[k], (xq, yq), va='bottom')
    lab(ax1, a['boss'], 'polar boss', (0.02, 0.97), ha='left', va='top')
    lab(ax1, a['section'], 'quarter section, see (b)', (0.86, 0.02), ha='center', va='bottom')
    ax1.text(0.995, 0.03, 'laminate thickness exaggerated ×1.6', ha='right', va='bottom', fontsize=6.5, color=P.INK2, transform=ax1.transAxes)
    P.label(ax1, 'a', x=0.0, y=1.0)
    # ---- (b) labels: layer stack on the left, transition features on the right
    b = {k: to_ax(v, cb2) for k, v in A2.items()}
    stack = [('F1b', r'helical $\pm30^\circ$'), ('F0b', r'helical $\pm\alpha_1$'), ('Hb', 'hoop group 2'),
             ('F1a', r'helical $\pm30^\circ$'), ('F0a', r'helical $\pm\alpha_1$'), ('Ha', 'hoop group 1'), ('liner', 'liner')]
    ys = np.linspace(0.04, 0.62, len(stack))
    # convert positions in ax2 data to figure for labels left of ax2: use annotate with xytext in axes fraction
    for (k, t), yt in zip(stack, ys):
        x, y = b[k]
        ax2.annotate(t, xy=(x, y), xytext=(-0.03, yt), textcoords=('axes fraction', 'data'), ha='right', va='center',
                     fontsize=7, arrowprops=dict(arrowstyle='-', lw=0.5, color=ink, shrinkA=1, shrinkB=1.5),
                     annotation_clip=False)
        ax2.plot([x], [y], 'o', ms=2.0, color=ink, mec='white', mew=0.4, clip_on=False, zorder=5)
    # tangent line marker
    x0, y0 = b['tl']; x1, y1 = b['tl_top']
    dx, dy = x1 - x0, y1 - y0
    ax2.plot([x0 - 0.25*dx, x1 + 0.9*dx], [y0 - 0.25*dy, y1 + 0.9*dy], color=ink, lw=0.7, ls=(0, (4, 2)))
    ax2.text(x1 + 0.95*dx, y1 + 0.95*dy - 0.01, 'tangent line', ha='center', va='bottom', fontsize=7)
    # drop-off and hoop end
    r0, h1 = b['ramp0'], b['hend']
    ax2.annotate('', xy=(h1[0], h1[1]), xytext=(r0[0], r0[1]), arrowprops=dict(arrowstyle='<->', lw=0.6, color=P.C['orange'],
                 shrinkA=0, shrinkB=0, mutation_scale=7))
    ax2.annotate(r'hoop drop-off $\ell_h$', xy=(0.5*(r0[0] + h1[0]), 0.5*(r0[1] + h1[1])), xytext=(0.62, 0.10),
                 ha='left', va='center', fontsize=7, arrowprops=lk, annotation_clip=False)
    ax2.plot([h1[0]], [h1[1]], 'o', ms=2.6, color=P.C['orange'], mec='white', mew=0.4, zorder=6)
    ax2.annotate(r'hoop end $e$ (on dome shoulder)', xy=h1, xytext=(0.66, 0.22), ha='left', va='center', fontsize=7,
                 arrowprops=lk, annotation_clip=False)
    ax2.annotate('dome', xy=b['dome'], xytext=(0.95, 0.42), ha='center', va='bottom', fontsize=7, arrowprops=lk,
                 annotation_clip=False)
    P.label(ax2, 'b', x=-0.17, y=0.98)
    P.save(fig, 'fig_cutaway')


if __name__ == '__main__':
    main()
