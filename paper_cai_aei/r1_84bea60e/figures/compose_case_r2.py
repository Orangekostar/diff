"""Assemble unchanged c8-16 states; no research model or event processing."""
from pathlib import Path
import csv
import importlib.util
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
ART = ROOT/'artifacts/cai_agent_v3/manuscript_revision/r2_f4758829'
# Set PANEL_ALIGNMENT_TOOL to the installed alignment-audit script on other systems.
qa_path = Path(os.environ.get('PANEL_ALIGNMENT_TOOL', '/home/ww/.codex/skills/nature-figure/scripts/audit_panel_alignment.py'))
spec = importlib.util.spec_from_file_location('panel_alignment', qa_path)
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)
manifest = list(csv.DictReader((ROOT/'results/cai_agent_v3/paper_evidence/r1_e2a11154/case_figure_reuse.csv').open()))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig, axes = plt.subplots(1, 2, figsize=(6.535, 3.85))
fig.subplots_adjust(left=.01,right=.99,bottom=.015,top=.92,wspace=.035)
for ax,step,label in zip(axes,[1,8],['a','b']):
    name=f'74t7kcdgkr_c8-16_measured_{step}.png'
    assert any(r['specimen_key']=='74t7kcdgkr:c8-16' and r['source_path'].endswith('/'+name) for r in manifest)
    original=Image.open(OUT/name)
    ax.imshow(original,interpolation='none',aspect='equal')
    ax.set_axis_off()
    ax.set_title(f'{label}   After {step} acquisition'+('s' if step>1 else ''),loc='left',fontsize=10,pad=6)
fig.canvas.draw()
qa.require_matplotlib_panel_alignment(fig,json_out=ART/'CASE_PANEL_ALIGNMENT.json')
fig.savefig(OUT/'Fig4_case_process.pdf',facecolor='white')
fig.savefig(OUT/'Fig4_case_process.svg',facecolor='white')
fig.savefig(OUT/'Fig4_case_process.png',dpi=300,facecolor='white')
plt.close(fig)
print('Assembled two unchanged case images at native aspect ratio')
