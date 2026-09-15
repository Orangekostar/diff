"""Display saved diagnostic attention only. No inference, smoothing or cell scoring."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53'

def main():
 status=json.loads((OUT/'attention_status.json').read_text())
 if status['status']!='ATTENTION_EXPORTED':print('No attention to plot:',status['reason']);return
 arrays=[np.load(OUT/f'attention_{n}.npy',allow_pickle=False) for n in ['clean','numbered']]
 images=[Image.open(OUT/f).convert('RGB') for f in ['02_vlm_clean_input.png','03_vlm_numbered_input.png']]
 vmax=max(float(a.max()) for a in arrays);norm=Normalize(vmin=0,vmax=vmax);cmap=plt.get_cmap('magma')
 meta=json.loads((OUT/'attention_preflight.json').read_text());masses=json.loads((OUT/'attention_mass.json').read_text());ident=json.loads((OUT/'identity.json').read_text());observations={}
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
 for name,a,im,png in zip(['clean','numbered'],arrays,images,['09_pre_cell_attention_clean.png','10_pre_cell_attention_numbered.png']):
  native=np.asarray(Image.fromarray(a).resize(im.size,Image.Resampling.NEAREST),dtype=np.float32)
  np.save(OUT/f'attention_{name}_native_nearest.npy',native)
  Image.fromarray(np.uint8(cmap(norm(native))*255),'RGBA').convert('RGB').save(OUT/f'attention_{name}_native_unsmoothed.png')
  fig,axs=plt.subplots(1,2,figsize=(12,6.5),layout='constrained');extent=(-.5,im.width-.5,im.height-.5,-.5)
  axs[0].imshow(a,origin='upper',extent=extent,interpolation='nearest',norm=norm,cmap=cmap)
  axs[0].set_title(f'{a.shape[0]} × {a.shape[1]} merged tokens (unsmoothed)')
  axs[1].imshow(im,origin='upper',interpolation='none');axs[1].imshow(a,origin='upper',extent=extent,interpolation='nearest',norm=norm,cmap=cmap,alpha=.38)
  axs[1].set_title('Overlay, constant alpha 0.38; nearest-neighbour display')
  for ax in axs:ax.set_xlim(-.5,im.width-.5);ax.set_ylim(im.height-.5,-.5);ax.axis('off')
  fig.suptitle(f'{name}: diagnostic replay / pre-first-cell token attention\nquery {meta["query_index"]}, before digit "3" of historical cell 36; image mass={masses[name]:.6f}',fontsize=11)
  cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axs,shrink=.72,pad=.015);cb.set_label('Raw attention probability per visual token\nSame scale across both images; no per-image normalization')
  fig.savefig(OUT/png,dpi=150,facecolor='white');plt.close(fig)
  r,c=np.unravel_index(a.argmax(),a.shape);x=(c+.5)*im.width/a.shape[1];y=(r+.5)*im.height/a.shape[0]
  x0,x1=c*im.width/a.shape[1],(c+1)*im.width/a.shape[1];y0,y1=r*im.height/a.shape[0],(r+1)*im.height/a.shape[0]
  overlaps=[rr*8+cc for rr in range(8) for cc in range(8) if max(x0,cc*im.width/8)<min(x1,(cc+1)*im.width/8) and max(y0,rr*im.height/8)<min(y1,(rr+1)*im.height/8)]
  observations[name]={'maximum_token_footprint_xyxy':[float(x0),float(y0),float(x1),float(y1)],'overlapping_cells':overlaps,'maximum_token_row':int(r),'maximum_token_col':int(c),'maximum_token_weight':float(a[r,c]),'render_center_xy':[float(x),float(y)],'containing_cell_at_token_center':int(y//(im.height/8))*8+int(x//(im.width/8)),'interpretation':'descriptive token maximum, NOT best cell, damage GT or CAI value'}
 fig,axs=plt.subplots(1,2,figsize=(12,6.8),layout='constrained')
 for ax,name,a,im in zip(axs,['clean','numbered'],arrays,images):
  ax.imshow(im,origin='upper',interpolation='none');ax.imshow(a,origin='upper',extent=(-.5,im.width-.5,im.height-.5,-.5),interpolation='nearest',norm=norm,cmap=cmap,alpha=.38)
  xe=np.rint(np.linspace(0,im.width,9));ye=np.rint(np.linspace(0,im.height,9))
  for i in ident['raw_cells']:
   r,c=divmod(i,8);ax.add_patch(Rectangle((xe[c]-.5,ye[r]-.5),xe[c+1]-xe[c],ye[r+1]-ye[r],fill=False,ec='#00ffff',lw=1.6))
  r,c=divmod(ident['historical_first_action'],8);ax.add_patch(Rectangle((xe[c]-.5,ye[r]-.5),xe[c+1]-xe[c],ye[r+1]-ye[r],fill=False,ec='#00ff33',lw=2.4,ls='--'))
  ax.set_title(name+f' (visual mass {masses[name]:.6f})');ax.set_xlim(-.5,im.width-.5);ax.set_ylim(im.height-.5,-.5);ax.axis('off')
 fig.suptitle('Diagnostic replay / pre-first-cell token attention\nCyan: historical raw cells; dashed green: historical Actor first action; no candidates inferred from heatmap',fontsize=10)
 cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),ax=axs,shrink=.7,pad=.015);cb.set_label('Raw attention per token — shared scale')
 fig.savefig(OUT/'11_attention_with_raw_cells.png',dpi=150,facecolor='white');plt.close(fig)
 (OUT/'attention_display.json').write_text(json.dumps({'vmin':0,'vmax_shared':vmax,'conditional_normalization':False,'interpolation':'nearest, no Gaussian or bilinear smoothing','native_npy_meaning':'display-only nearest-neighbour replication, not new pixel-level evidence','observations':observations},indent=2)+'\n')
 print(json.dumps(observations))
if __name__=='__main__':main()
