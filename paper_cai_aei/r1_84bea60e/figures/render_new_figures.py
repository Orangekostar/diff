"""Render the two authorized figures from code facts and saved contributions."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
                     'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name):
    for ext in ('pdf','svg','png'):
        fig.savefig(OUT/f'{name}.{ext}',dpi=300,facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,4.5));fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.96)
ax.set(xlim=(0,10),ylim=(0,7));ax.axis('off')
def box(x,y,w,h,text,color='#e8edf2'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06',
                               linewidth=.8,edgecolor='#506275',facecolor=color))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
def arrow(a,b,**kw):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,
                               linewidth=1,color='#506275',**kw))
box(.2,5.7,2.5,1,'Surface image\n64 local descriptors')
box(3.25,5.7,3,1,'Cached surface VLM prior\nRegions + ordinal confidence')
box(.2,3.35,3.0,1.3,'Visible decision state\nSurface + measured cells\nMask, history, prediction\nAcquisition budget','#e6f1ec')
box(4.0,3.45,2.3,1.1,'Spatial actor\nScores for legal cells','#e6f1ec')
box(7.1,3.45,2.6,1.1,'Acquire one legal cell\nNative-pixel cap B = 0.25')
box(6.85,1.1,2.85,1.1,'Update observed mask\nReveal requested descriptor')
box(.2,1.1,3.0,1.1,'Frozen CAI predictor\nSurface + acquired cells\nUpdated strength estimate')
arrow((1.45,5.7),(1.45,4.65));arrow((4.75,5.7),(2.8,4.65))
arrow((3.2,4.0),(4,4.0));arrow((6.3,4.0),(7.1,4.0))
arrow((8.4,3.45),(8.4,2.2));arrow((6.85,1.65),(3.2,1.65))
arrow((1.7,2.2),(1.7,3.35))
ax.text(5,2.45,'Only acquired internal evidence enters the loop',ha='center',fontsize=8)
ax.text(5,.38,'True strength: training and retrospective scoring only. Parameters stay frozen during acquisition.',ha='center',fontsize=8)
save(fig,'Fig1_framework')

labels={'CENTER_FIRST':'Center-first','GEOMETRY_SPREAD':'Geometry-spread','SERPENTINE':'Serpentine',
        'RANDOM':'Random','LEARNED_STATIC_TRUE':'Learned-static','NO_VLM_SPATIAL_FEEDBACK':'No-VLM spatial feedback',
        'VLM_MEAN_FEEDBACK':'VLM mean feedback','VLM_SPATIAL_FEEDBACK':'VLM spatial feedback (main)',
        'VLM_SPATIAL_OPEN_LOOP':'VLM spatial open-loop'}
with (OUT.parent/'analysis/timing_contributions.csv').open() as f:rows=list(csv.DictReader(f))
assert len(rows)==36 and {r['method'] for r in rows}==set(labels)
fig,ax=plt.subplots(figsize=(7.2,5.3));fig.subplots_adjust(left=.34,right=.98,bottom=.12,top=.83)
colors=['#226b8a','#68a1ac','#b48a55','#77717d']
for stage in range(1,5):
    values={r['method']:float(r['contribution_mpa']) for r in rows if int(r['stage'])==stage}
    ys=[i+(stage-2.5)*.18 for i in range(9)]
    ax.barh(ys,[values[m] for m in labels],height=.16,color=colors[stage-1],
            label=['(0, 0.0625]','(0.0625, 0.125]','(0.125, 0.1875]','(0.1875, 0.25]'][stage-1])
ax.set_yticks(range(9),labels.values());ax.invert_yaxis();ax.axvline(0,color='#555555',lw=.7)
ax.set_xlim(-1.6,14);ax.set_xlabel('Timing-weighted contribution (MPa)')
ax.legend(loc='lower center',bbox_to_anchor=(.45,1.03),ncol=2,frameon=False,
          title='Acquisition completion stage',fontsize=8,title_fontsize=9)
ax.tick_params(axis='y',length=0)
save(fig,'Fig4_timing')
print('Rendered two figures; 36 signed saved values; no new study analysis')
