"""Four finite export checks; no model calls, training or statistical recomputation."""
from pathlib import Path
import csv,hashlib,json,re,subprocess
from html.parser import HTMLParser
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53';ART=ROOT/'artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53'
BASE='8778aa53875c75f3ca681915e322a9c02cbc1ac3'
def read(name):return json.loads((OUT/name).read_text())
def main():
 identity=read('identity.json');a=read('stage_a.json');cache=read('cache_record.json');b=read('attention_status.json');meta=read('attention_preflight.json')
 assert identity['specimen_key']=='cgtnjyggtm:q24-48'
 assert identity['cache_key']==cache['cache_key']=='d11f57f3d3ba2e96339d1857e31f3f9192074a624682d7bf4faa29cd04345773'
 for name,key in [('02_vlm_clean_input.png','clean_image_sha256'),('03_vlm_numbered_input.png','gridded_image_sha256')]:assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==cache['request'][key]
 assert a['raw_parsed_cache_csv_all_64_equal'] and a['c0_matches_trace']
 coords=list(csv.DictReader((OUT/'coordinate_trace.csv').open()));assert len(coords)==64
 for i,r in enumerate(coords):
  assert [int(r[k]) for k in ['cell_id','row','col']]==[i,i//8,i%8]
  x0,x1,y0,y1=[int(r[k]) for k in ['x0','x1_exclusive','y0','y1_exclusive']]
  assert 0<=x0<x1<=1024 and 0<=y0<y1<=1024
  assert [float(r[k]) for k in ['display_x0','display_x1','display_y0','display_y1']]==[x0-.5,x1-.5,y0-.5,y1-.5]
  assert int(r['cscan_width'])==675 and int(r['cscan_height'])==674
 assert all(x['same_code_rgb_equal'] and x['max_abs_pixel_difference']==0 for x in a['historical_overlay_comparison'].values())
 synthetic=list(csv.DictReader((OUT/'synthetic_roundtrip.csv').open()));assert len(synthetic)==64 and all(r['roundtrip_pass']=='True' and r['cell_id']==r['crop_unique_id'] for r in synthetic)
 assert b['status']=='ATTENTION_EXPORTED' and b['actual_qwen_forwards']==1 and b['elapsed_seconds']<=900
 prefix=(OUT/'answer_prefix.txt').read_text();raw=(OUT/'raw_response.txt').read_text();target=re.search(r'"cells"\s*:\s*\[\s*(\d+)',raw)
 assert prefix==raw[:target.start(1)] and target.group(1)=='36' and not re.search(r'"cells"\s*:\s*\[\s*\d',prefix)
 assert meta['query_index']==2792 and meta['total_tokens']==2793 and meta['prefix_tokens']==26 and meta['target_first_token_decoded']=='3'
 rows=np.load(OUT/'attention_headmean_last4.npy',allow_pickle=False);mean=np.load(OUT/'attention_mean_query.npy',allow_pickle=False)
 assert rows.shape==(4,2793) and np.isfinite(rows).all() and (rows>=0).all();assert np.allclose(rows.sum(1),1,atol=1e-6) and np.array_equal(rows.mean(0),mean)
 mapping=list(csv.DictReader((OUT/'attention_token_mapping.csv').open()));assert len(mapping)==2450
 used=set()
 for name in ['clean','numbered']:
  part=[r for r in mapping if r['image']==name];pos=[int(r['sequence_position']) for r in part];assert len(pos)==1225 and not (used & set(pos));used.update(pos)
  att=np.load(OUT/f'attention_{name}.npy',allow_pickle=False);assert att.shape==(35,35) and np.array_equal(att.ravel(),mean[pos])
  assert all(int(r['merged_token_index'])==int(r['row'])*35+int(r['col']) for r in part)
 mass=read('attention_mass.json');assert abs(mass['clean']+mass['numbered']+mass['non_visual']-1)<1e-6
 cap=read('attention_capture_checks.json');assert set(cap)=={'24','25','26','27'}
 assert all(r['captured_shape']==[1,28,1,2793] and r['head_sum_max_abs_error']<1e-6 for r in cap.values())
 # Probability-weighted V agrees with unchanged BF16 SDPA output to its observed precision.
 assert all(r['row_output_reconstruction_rms']<.002 and r['row_output_reconstruction_max_abs']/r['sdpa_output_abs_max']<.004 for r in cap.values())
 display=read('attention_display.json');assert display['vmin']==0 and display['conditional_normalization'] is False
 class Links(HTMLParser):
  def __init__(self):super().__init__();self.paths=[];self.images=[]
  def handle_starttag(self,tag,attrs):
   attrs=dict(attrs)
   if tag=='a' and 'href' in attrs:self.paths.append(attrs['href'])
   if tag=='img':self.images.append(attrs['src'])
 html=Links();html.feed((OUT/'index.html').read_text());assert len(html.images)==11 and all((OUT/path).is_file() for path in html.paths)
 for f in OUT.glob('*.png'):
  with Image.open(f) as im:im.verify()
 changed=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
 allowed=('scripts/vlm_cell_diagnostic/','results/cai_agent_v3/vlm_cell_diagnostic/','artifacts/cai_agent_v3/vlm_cell_diagnostic/')
 assert all(n.startswith(allowed) or n=='results/cai_agent_v3/compute_ledger.jsonl' for n in changed),changed
 ledger=ROOT/'results/cai_agent_v3/compute_ledger.jsonl';base=subprocess.check_output(['git','show',f'{BASE}:results/cai_agent_v3/compute_ledger.jsonl'],cwd=ROOT);now=ledger.read_bytes();assert now.startswith(base);new=now[len(base):].splitlines();assert len(new)==1 and json.loads(new[0])['actual_qwen_forward_attempts']==1
 report={'Q1_identity_and_cache':'PASS exact input PNG hashes, named VALID case and raw/parsed/features chain','Q2_coordinate_chain':'PASS 64-cell native/crop/overlay roundtrip; archived cue and first-action PNG RGB exactly reproduced','Q3_attention_definition_and_arrays':'PASS strict prefix; query2792; layers24–27, all28heads; two35x35 segments; probability and SDPA row consistency; 1 forward','Q4_viewable_delivery_and_frozen_scope':'PASS PNG decode, 11 HTML images, relative links; production/paper/history unchanged except one authorized ledger append. Browser and Git evidence in separate records.'}
 (ART/'FINITE_VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
