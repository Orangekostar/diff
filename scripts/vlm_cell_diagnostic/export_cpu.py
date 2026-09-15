"""Read-only single-case cache/coordinate export. No model execution."""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='4'
import csv, gzip, hashlib, io, json, shutil, sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
# The local environment has another editable cmc_bbdm install; bind this checkout.
import types
package=types.ModuleType('cmc_bbdm');package.__path__=[str(ROOT/'src/cmc_bbdm')]
sys.modules['cmc_bbdm']=package
from cmc_bbdm.vlm_cscan.runtime import render_surface_inputs, _image_sha256
from cmc_bbdm.learned_cscan.perception import parse_surface_percept, SurfacePerceptRequest, SURFACE_PERCEPT_PROMPT
from cmc_bbdm.cai_agent_v3.vlm_perception import _features, _prompt_sha256
from cmc_bbdm.cai_active_image.environment import NativeCellGrid
# Execute only the inspected pure crop function, avoiding unrelated encoder imports.
import ast
crop_source=ROOT/'src/cmc_bbdm/cai_agent_v3/feature_bank.py'
crop_node=next(n for n in ast.parse(crop_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_cell_crops')
exec(compile(ast.Module(body=[crop_node],type_ignores=[]),str(crop_source),'exec'))
draw_source=ROOT/'src/cmc_bbdm/cai_agent_v3/diagnostics.py'
draw_node=next(n for n in ast.parse(draw_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_draw_cells')
exec(compile(ast.Module(body=[draw_node],type_ignores=[]),str(draw_source),'exec'))

OUT=ROOT/'results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53'
ART=ROOT/'artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53'
DATA=ROOT/'results/cai_agent_v3/new_protocol'
W3=ROOT/'results/cai_agent_v3/w3_pilot/r1_0e11452a'
KEY='cgtnjyggtm:q24-48'
CACHE='d11f57f3d3ba2e96339d1857e31f3f9192074a624682d7bf4faa29cd04345773'

def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def csvout(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def findrow(path):
 with path.open() as f:
  found=[r for r in csv.DictReader(f) if r['specimen_key']==KEY]
 assert len(found)==1
 return found[0]
def image_panel(im,title,layers=(),grid=False):
 fig,ax=plt.subplots(figsize=(7,8));ax.imshow(im,origin='upper',interpolation='none');g=NativeCellGrid.from_shape((im.height,im.width))
 if grid:
  for c in g.cells:
   ax.add_patch(Rectangle((c.col_start-.5,c.row_start-.5),c.col_stop-c.col_start,c.row_stop-c.row_start,fill=False,lw=.45,ec='#66cccc'))
 for cells,color,lw,label in layers:
  for i in cells:
   c=g.cells[i];ax.add_patch(Rectangle((c.col_start-.5,c.row_start-.5),c.col_stop-c.col_start,c.row_stop-c.row_start,fill=False,lw=lw,ec=color))
  ax.plot([],[],color=color,lw=lw,label=label)
 if layers:ax.legend(loc='upper center',bbox_to_anchor=(.5,-.015),fontsize=9)
 ax.set_xlim(-.5,im.width-.5);ax.set_ylim(im.height-.5,-.5);ax.set_title(title,fontsize=11);ax.axis('off');return fig,ax

def save(fig,name):fig.savefig(OUT/name,dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)

def main():
 OUT.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
 # Filter exact authorized VALID case before selecting columns; do not export labels.
 allsource=findrow(DATA/'candidate_queue.csv')
 allowed=['specimen_key','split','impacted_surface_path','surface_sha256','registered_cscan_crop_path','registered_cscan_crop_sha256','registered_cscan_height_px','registered_cscan_width_px']
 source={k:allsource[k] for k in allowed};assert source['split']=='VALID'
 assert findrow(W3/'case_manifest.csv')['scope']=='VALID'
 features=findrow(DATA/'vlm_actor_features_fit.csv');assert features['cache_key']==CACHE
 cache=[json.loads(l) for l in (DATA/'vlm_surface_percepts.jsonl').open() if json.loads(l).get('cache_key')==CACHE];assert len(cache)==1;cache=cache[0]
 assert SurfacePerceptRequest(**cache['request']).cache_key==CACHE
 assert _prompt_sha256()==cache['request']['prompt_sha256']
 external=Path(json.loads((DATA/'feature_bank_manifest.json').read_text())['encoder_execution_root'])
 path=external/source['impacted_surface_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['surface_sha256']
 original=Image.open(path).convert('RGB');render=render_surface_inputs(original,max_edge=1024)
 assert render.clean_sha256==features['clean_image_sha256']==cache['request']['clean_image_sha256']
 assert render.gridded_sha256==features['gridded_image_sha256']==cache['request']['gridded_image_sha256']
 # Actual historical pixel inputs: no added titles/labels inside these three images.
 for name,im in [('01_source_surface.png',original),('02_vlm_clean_input.png',render.clean),('03_vlm_numbered_input.png',render.gridded)]:im.save(OUT/name,optimize=False,compress_level=9)
 parsed=parse_surface_percept(cache['raw_text']);indicator,confidence=_features(parsed)
 fi=np.array(features['region_indicator'].split(';'),dtype=np.float32);fc=np.array(features['confidence'].split(';'),dtype=np.float32)
 assert np.array_equal(indicator,fi) and np.array_equal(confidence,fc)
 raw=json.loads(cache['raw_text'].strip().removeprefix('```json').removesuffix('```'))
 parsed_payload={'no_reliable_cue':parsed.no_reliable_cue,'regions':[{'cells':list(r.cells),'cue':r.cue,'alternative':r.alternative,'confidence':r.confidence} for r in parsed.regions]}
 assert raw==parsed_payload==cache['percept'];rawcells=[c for r in raw['regions'] for c in r['cells']]
 episodes=[]
 with gzip.open(W3/'policy_validation_episodes.csv.gz','rt') as f:
  for r in csv.DictReader(f):
   if r['specimen_key']==KEY and r['method']=='VLM_SPATIAL_FEEDBACK':episodes.append(r)
 assert len(episodes)==1;episode=episodes[0];trace=json.loads(episode['execution_trace']);actions=list(map(int,episode['cells'].split(';')))
 assert actions==[r['cell'] for r in trace]
 eligible=(fi>0)&(fc>=np.float32(2/3));highest=fc[eligible].max();legal=np.array([x=='1' for x in trace[0]['environment_legal']]);c0=np.flatnonzero(legal&eligible&(fc==highest)).tolist()
 assert c0==[i for i,x in enumerate(trace[0]['proposal_legal']) if x=='1'];first=actions[0];assert first in c0
 # Only action/state metadata, not CAI labels or new assessment.
 history=[{k:r[k] for k in ['actor_call_index','action_index','cell','visible_cells_before','environment_legal','proposal_legal','c0_reason','before_cost','after_cost']} for r in trace]
 dump(OUT/'historical_actions.json',history);dump(OUT/'cache_record.json',cache);dump(OUT/'parsed_response.json',parsed_payload)
 (OUT/'raw_response.txt').write_text(cache['raw_text']);(OUT/'prompt.txt').write_text(SURFACE_PERCEPT_PROMPT)
 g=NativeCellGrid.from_shape((render.clean.height,render.clean.width));w,h=render.clean.size;sw,sh=original.size
 levels={c:r['confidence'] for r in raw['regions'] for c in r['cells']};rows=[]
 for c in g.cells:
  u0,u1=c.col_start/w,c.col_stop/w;v0,v1=c.row_start/h,c.row_stop/h
  rows.append(dict(cell_id=c.index,row=c.row,col=c.column,source_width=sw,source_height=sh,render_width=w,render_height=h,x0=c.col_start,y0=c.row_start,x1_exclusive=c.col_stop,y1_exclusive=c.row_stop,u0=u0,v0=v0,u1=u1,v1=v1,display_x0=c.col_start-.5,display_y0=c.row_start-.5,display_x1=c.col_stop-.5,display_y1=c.row_stop-.5,source_x0_approx=v0*sw,source_y0_approx=(1-u1)*sh,source_x1_approx=v1*sw,source_y1_approx=(1-u0)*sh,raw=c.index in rawcells,parsed=bool(indicator[c.index]),feature_indicator=float(fi[c.index]),ordinal_confidence=float(fc[c.index]),confidence_level=levels.get(c.index,'unknown/unselected'),c0=c.index in c0,historical_first_action=c.index==first,historical_action_step=actions.index(c.index)+1 if c.index in actions else ''))
 cg=NativeCellGrid.from_shape((int(source['registered_cscan_height_px']),int(source['registered_cscan_width_px'])))
 for row,cell in zip(rows,cg.cells):
  row.update(cscan_width=int(source['registered_cscan_width_px']),cscan_height=int(source['registered_cscan_height_px']),cscan_x0=cell.col_start,cscan_y0=cell.row_start,cscan_x1_exclusive=cell.col_stop,cscan_y1_exclusive=cell.row_stop)
 csvout(OUT/'coordinate_trace.csv',rows)
 csvout(OUT/'features_64.csv',[{k:r[k] for k in ['cell_id','raw','parsed','feature_indicator','ordinal_confidence','confidence_level','c0','historical_first_action']} for r in rows])
 np.save(OUT/'region_indicator.npy',fi);np.save(OUT/'ordinal_confidence.npy',fc)
 fig,ax=image_panel(render.clean,'Diagnostic reference, NOT historical VLM input\nrow-major IDs; top-left origin; row/col zero-based',grid=True)
 for c in g.cells:ax.text((c.col_start+c.col_stop-1)/2,(c.row_start+c.row_stop-1)/2,f'{c.index}\n({c.row},{c.column})',ha='center',va='center',fontsize=9,color='white',bbox=dict(facecolor='black',alpha=.55,pad=1,edgecolor='none'))
 save(fig,'04_readable_grid_reference.png')
 fig,ax=plt.subplots(figsize=(8,8));cmap=ListedColormap(['#dddddd','#b9d7ed','#4d91bd','#16476c']);norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4)
 discrete=np.rint(fc*3).reshape(8,8);im=ax.imshow(discrete,origin='upper',interpolation='none',cmap=cmap,norm=norm)
 for i in range(64):ax.text(i%8,i//8,f'{i}\n'+(['U','low','medium','high'][int(discrete.flat[i])]),ha='center',va='center',fontsize=7.5,color='white' if discrete.flat[i]>=2 else 'black')
 ax.set_xticks(range(8));ax.set_yticks(range(8));ax.set_xlabel('column (zero-based)');ax.set_ylabel('row (zero-based)');ax.set_title('Cached VLM cell confidence — ordinal\nafter VLM decoding / before Actor selection\nU = unknown / unselected, not confirmed normal',fontsize=11)
 cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04,ticks=range(4));cb.ax.set_yticklabels(['unknown/unselected','low (1/3)','medium (2/3)','high (1)']);save(fig,'05_cached_confidence_8x8.png')
 fig,_=image_panel(render.gridded,'Raw JSON IDs on historical numbered input',[(rawcells,'#ff8c00',2,'Raw JSON: '+str(rawcells))]);save(fig,'06_raw_ids_on_numbered_input.png')
 fig,_=image_panel(render.clean,'Parsed cells and frozen CSV features: identical at all 64 cells',[(rawcells,'#009e73',2,'Parsed = CSV (medium, 2/3)')]);save(fig,'07_parsed_and_feature_cells.png')
 fig,axs=plt.subplots(1,3,figsize=(15,6))
 for ax,cells,title,color in zip(axs,[rawcells,c0,[first]],['All decoded candidates','Highest-reliable C0','Historical Actor first action'],['#dd8800','#0072b2','#cc0077']):
  ax.imshow(render.clean,origin='upper',interpolation='none');_draw_cells(ax,g,cells,color=color,fill=False);ax.set_title(title+'\n'+str(cells),fontsize=10);ax.axis('off')
 fig.suptitle('Cached VLM candidates ≠ attention ≠ historical Actor action',fontsize=12);save(fig,'08_c0_and_historical_action.png')
 # Compare archived overlays with deterministic same-code re-render; do not overwrite archives.
 manifest=json.loads((W3/'figure_manifest.json').read_text());case=next(r for r in manifest['cases'] if r['specimen_key']==KEY)
 reuse=[r for r in csv.DictReader((ROOT/'results/cai_agent_v3/paper_evidence/r1_e2a11154/case_figure_reuse.csv').open()) if r['specimen_key']==KEY];overlay_checks={}
 for mode,cells,title in [('surface_cues',rawcells,'VALID surface + frozen VLM cues'),('first_action',[first],f'VALID real first action: cell {first}')]:
  histpath=next(ROOT/r['source_path'] for r in reuse if r['source_path'].endswith('_'+mode+'.png'));ref=next(r for r in reuse if ROOT/r['source_path']==histpath);assert str(histpath.relative_to(ROOT)) in case['paths'];assert hashlib.sha256(histpath.read_bytes()).hexdigest()==ref['source_sha256']
  fig,ax=plt.subplots(figsize=(5,5));ax.imshow(np.asarray(render.clean))
  for x in sorted({c.col_start for c in g.cells}|{g.cells[-1].col_stop}):ax.axvline(x-.5,color='gray',lw=.4)
  for y in sorted({c.row_start for c in g.cells}|{g.cells[-1].row_stop}):ax.axhline(y-.5,color='gray',lw=.4)
  _draw_cells(ax,g,cells,color='#D55E00',fill=True);ax.set_title(title);ax.axis('off');buf=io.BytesIO();fig.savefig(buf,dpi=150,bbox_inches='tight');plt.close(fig)
  old=np.asarray(Image.open(histpath).convert('RGB'));new=np.asarray(Image.open(buf).convert('RGB'));equal=old.shape==new.shape and np.array_equal(old,new)
  overlay_checks[mode]={'archived_path':str(histpath.relative_to(ROOT)),'archived_hash_matches_manifest':True,'same_code_rgb_equal':equal,'old_shape':list(old.shape),'new_shape':list(new.shape),'max_abs_pixel_difference':int(np.max(np.abs(old.astype(int)-new.astype(int)))) if old.shape==new.shape else None}
  shutil.copyfile(histpath,OUT/f'historical_{mode}.png')
 # Unique IDs on a non-square odd-size synthetic raster; tests real crop and display rectangle paths.
 synth=np.empty((67,83,3),dtype=np.uint8);sg=NativeCellGrid.from_shape((67,83))
 for c in sg.cells:synth[c.row_start:c.row_stop,c.col_start:c.col_stop]=[c.index,c.index*3,c.index*4]
 crops=_cell_crops(synth);synthetic=[];fig,ax=plt.subplots();ax.imshow(synth,origin='upper',interpolation='none');_draw_cells(ax,sg,list(range(64)),color='white',fill=False)
 for c,crop,patch in zip(sg.cells,crops,ax.patches):
  expected=np.array([c.index,c.index*3,c.index*4]);assert np.all(crop==expected)
  center=((c.col_start+c.col_stop-1)/2,(c.row_start+c.row_stop-1)/2);assert patch.contains_point(ax.transData.transform(center));assert tuple(patch.get_xy())==(c.col_start-.5,c.row_start-.5)
  synthetic.append({'cell_id':c.index,'crop_unique_id':int(crop[0,0,0]),'center_x':center[0],'center_y':center[1],'roundtrip_pass':True})
 plt.close(fig);Image.fromarray(synth).resize((664,536),Image.Resampling.NEAREST).save(OUT/'synthetic_64_ids.png');csvout(OUT/'synthetic_roundtrip.csv',synthetic)
 dx=max(abs(int(x)-round(i*w/8)) for i,x in enumerate(np.rint(np.linspace(0,w,9))));dy=max(abs(int(y)-round(i*h/8)) for i,y in enumerate(np.rint(np.linspace(0,h,9))))
 identity={'specimen_key':KEY,'source_identity_note':'已按任务指定q24-48的源图哈希、cache_key、两输入hash、固定案例与历史overlay交叉确认；本轮消息未附新图片，因此未声称与未提供的上传图逐像素匹配。','source':source,'source_path_resolved':str(path),'source_orientation':'source orientation: original file, no rotation','render_operation':'RGB -> ROTATE_270 (clockwise 90 once) -> longest-edge <=1024 LANCZOS','source_size_wh':list(original.size),'render_size_wh':list(render.clean.size),'cache_key':CACHE,'clean_sha256':render.clean_sha256,'numbered_sha256':render.gridded_sha256,'input_identity_exact_png_hash':True,'model_revision':cache['request']['model_revision'],'historical_call_count':cache['call_count'],'repaired':cache['repaired'],'historical_attention':'NOT_RECORDED','original_invalid_answer':'NOT_APPLICABLE_UNREPAIRED','raw_cells':rawcells,'c0':c0,'historical_first_action':first,'source_credit':'Hasebe et al., cgtnjyggtm version 1, CC BY 4.0; diagnostic overlays added.'}
 dump(OUT/'identity.json',identity)
 dump(OUT/'stage_a.json',{'status':'COORDINATE_EXPORT_COMPLETE','input_hashes_match':True,'raw_parsed_cache_csv_all_64_equal':True,'c0_matches_trace':True,'first_action_row_col':[first//8,first%8],'native_vs_render_boundary_max_difference_px':[dx,dy],'synthetic_roundtrip_64_pass':True,'historical_overlay_comparison':overlay_checks,'coordinate_conclusion':'No raw/parsed/CSV cell mismatch; exact row-major surface rectangles; see archived pixel comparison. Surface/C-scan grids use their own dimensions. No permutation mapping called by v3.','new_model_forwards':0})
 print(json.dumps({'case':KEY,'render':render.clean.size,'raw_cells':rawcells,'first':first,'overlay':overlay_checks},ensure_ascii=False))
if __name__=='__main__':main()
