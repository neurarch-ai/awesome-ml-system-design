import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = ("/private/tmp/claude-501/-Users-xingao-Projects-Neurarch/"
       "65adcdb4-d882-4642-bf5c-d42bf1be1f61/scratchpad/"
       "awesome-ml-system-design/book/speech/assets/")

plt.rcParams.update({'figure.dpi': 130, 'font.size': 11, 'figure.autolayout': True})

GREEN  = '#16a34a'
ORANGE = '#ea7317'
RED    = '#dc2626'
BLUE   = '#2563eb'
GRAY   = '#64748b'
L_GREEN  = '#dcfce7'
L_ORANGE = '#fed7aa'
L_RED    = '#fee2e2'
L_BLUE   = '#dbeafe'
L_GRAY   = '#f1f5f9'

# Alignment (5 columns):
# REF:  CALL   ME     ---      BACK   TODAY
# OP:   =      S      I        =      D
# HYP:  CALL   YOU    PLEASE   BACK   ---
#
# S=1, D=1, I=1, N=4 (reference has 4 real words)
# WER = (1 + 1 + 1) / 4 = 0.75

ref_toks = ['CALL', 'ME',  '',       'BACK', 'TODAY']
ops       = ['=',   'S',   'I',      '=',    'D']
hyp_toks  = ['CALL', 'YOU', 'PLEASE', 'BACK', '']

op_style = {
    '=': (L_GREEN,  GREEN,  'match'),
    'S': (L_ORANGE, ORANGE, 'substitution'),
    'D': (L_RED,    RED,    'deletion'),
    'I': (L_BLUE,   BLUE,   'insertion'),
}

NCOLS = len(ops)
FIG_W = 11.0
FIG_H = 4.8
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor('white')

# Top 70% for alignment grid, bottom 30% for formula
ax_grid = fig.add_axes([0.01, 0.30, 0.98, 0.66])
ax_form = fig.add_axes([0.01, 0.00, 0.98, 0.30])

# ---- alignment grid ----
ax_grid.set_xlim(-1.8, NCOLS + 0.3)
ax_grid.set_ylim(-0.15, 3.8)
ax_grid.axis('off')
ax_grid.set_title(
    'WER: aligning reference vs. hypothesis word by word (illustrative)',
    fontsize=11, pad=6
)

row_y = {'ref': 3.0, 'op': 2.0, 'hyp': 1.0}
row_lbl = {
    'ref': 'Reference\n(N = 4 words):',
    'op':  'Edit\noperation:',
    'hyp': 'Hypothesis:',
}

for row, y in row_y.items():
    ax_grid.text(-0.05, y, row_lbl[row],
                 ha='right', va='center', fontsize=9.5,
                 color='#1e293b', fontweight='bold')

CW, CH = 0.88, 0.70
PAD_X  = 0.05  # horizontal gap between cells
X0     = 0.10  # left edge of first cell

for i, (rt, op, ht) in enumerate(zip(ref_toks, ops, hyp_toks)):
    bg, fg, _ = op_style[op]
    x = X0 + i * (CW + PAD_X)

    # ref cell
    if rt:
        r = patches.FancyBboxPatch(
            (x, row_y['ref'] - CH / 2), CW, CH,
            boxstyle='round,pad=0.05',
            facecolor=bg, edgecolor='#94a3b8', lw=1.2
        )
        ax_grid.add_patch(r)
        ax_grid.text(x + CW / 2, row_y['ref'], rt,
                     ha='center', va='center', fontsize=9.5,
                     color='#1e293b', fontweight='bold')
    else:
        # empty ref (insertion): draw dashed outline
        r = patches.FancyBboxPatch(
            (x, row_y['ref'] - CH / 2), CW, CH,
            boxstyle='round,pad=0.05',
            facecolor='#f8fafc', edgecolor=fg, lw=1.2, ls='--'
        )
        ax_grid.add_patch(r)
        ax_grid.text(x + CW / 2, row_y['ref'], '(none)',
                     ha='center', va='center', fontsize=8,
                     color=fg, style='italic')

    # operation label cell
    r2 = patches.FancyBboxPatch(
        (x, row_y['op'] - CH / 2), CW, CH,
        boxstyle='round,pad=0.05',
        facecolor=bg, edgecolor=fg, lw=1.8
    )
    ax_grid.add_patch(r2)
    ax_grid.text(x + CW / 2, row_y['op'], op,
                 ha='center', va='center', fontsize=14,
                 color=fg, fontweight='bold')

    # hyp cell
    if ht:
        r3 = patches.FancyBboxPatch(
            (x, row_y['hyp'] - CH / 2), CW, CH,
            boxstyle='round,pad=0.05',
            facecolor=bg, edgecolor='#94a3b8', lw=1.2
        )
        ax_grid.add_patch(r3)
        ax_grid.text(x + CW / 2, row_y['hyp'], ht,
                     ha='center', va='center', fontsize=9.5,
                     color='#1e293b', fontweight='bold')
    else:
        # empty hyp (deletion)
        r3 = patches.FancyBboxPatch(
            (x, row_y['hyp'] - CH / 2), CW, CH,
            boxstyle='round,pad=0.05',
            facecolor='#f8fafc', edgecolor=fg, lw=1.2, ls='--'
        )
        ax_grid.add_patch(r3)
        ax_grid.text(x + CW / 2, row_y['hyp'], '(none)',
                     ha='center', va='center', fontsize=8,
                     color=fg, style='italic')

# Legend
legend_handles = []
for op, (bg, fg, lbl) in op_style.items():
    legend_handles.append(
        plt.Line2D([0], [0], marker='s', color='w',
                   markerfacecolor=fg, markersize=11,
                   label=f'{op}  = {lbl}')
    )
ax_grid.legend(handles=legend_handles, loc='upper right',
               fontsize=9, frameon=True, framealpha=0.92,
               edgecolor='#e2e8f0')

# ---- formula row ----
ax_form.axis('off')
ax_form.text(
    0.50, 0.60,
    r'$\text{WER} = \dfrac{S + D + I}{N} = \dfrac{1 + 1 + 1}{4} = 0.75$'
    r'     (S = substitutions, D = deletions, I = insertions, N = reference word count)',
    ha='center', va='center', transform=ax_form.transAxes,
    fontsize=11.5, color='#1e293b'
)

fig.savefig(OUT + 'fig-wer-computation.png', dpi=130, bbox_inches='tight')
plt.close(fig)
print('wrote fig-wer-computation.png')
