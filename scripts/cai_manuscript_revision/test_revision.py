"""Eight bounded manuscript-adapter checks; no research model execution."""
from pathlib import Path
import math,re,unittest
from evidence import read_json,read_csv,normalize_order,held,left_area,first_quality,relative_saving,METHODS,SHA,ORDER,EVID

ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'paper_cai_aei/r3_positive_review_r1'
A=ROOT/'artifacts/cai_agent_v3/manuscript/r3_positive_review_r1'

class RevisionChecks(unittest.TestCase):
 def test_model_and_method_binding(self):
  lock=read_json(A/'MODEL_IDENTITY_LOCK.json');job=read_json(ROOT/'results/cai_agent_v3/w3_pilot/r1_0e11452a/jobs/NO_VLM_SPATIAL_FEEDBACK.json')['manifest']
  self.assertEqual(lock['actor']['method'],'NO_VLM_SPATIAL_FEEDBACK');self.assertFalse(lock['actor']['use_vlm']);self.assertTrue(lock['actor']['use_feedback'])
  self.assertEqual((lock['actor']['selected_update'],lock['actor']['seed'],lock['actor']['parameter_count']),(1000,2026091302,378978))
  self.assertEqual(job['checkpoint_sha256'],lock['actor']['file_sha256']);self.assertEqual(job['state_dict_sha256'],lock['actor']['state_dict_sha256'])
  self.assertEqual(lock['predictor']['name'],'MEAN_SC');self.assertEqual(lock['predictor']['selected_update'],1750)
  for role in ['actor','predictor']:self.assertFalse(lock[role]['forward_allowed']);self.assertFalse(lock[role]['loaded'])
  rows=read_csv(P/'tables/table3.csv');self.assertEqual([r['method'] for r in rows[:6]],METHODS)
  self.assertEqual(rows[5]['strategy'],'Proposed');self.assertFalse(any(r['method'].startswith('VLM_') for r in rows))
  source=(ROOT/'src/cmc_bbdm/cai_agent_v3/models.py').read_text();self.assertIn('nn.Linear(48 + 48 + 6, self.token_width)',source);self.assertIn('observed = measured_weight.unsqueeze(-1) * cscan',source)
  self.assertTrue(all(v==0 for v in read_json(A/'SOURCE_BINDINGS.json')['activities'].values()))

 def test_stage_sign_normalization(self):
  raw=read_csv(ROOT/ORDER/'order_paired_contrasts.csv');norm=normalize_order(raw,'23a65f8d481797298b25bb228d1b2fd91181c65f')
  self.assertEqual(norm,normalize_order(norm,'23a65f8d481797298b25bb228d1b2fd91181c65f'))
  changed=0
  for r,n in zip(raw,norm):
   if r['metric'].startswith('weighted_stage_'):
    changed+=1
    self.assertEqual(float(n['estimate_mpa']),-float(r['estimate_mpa']));self.assertEqual(float(n['ci_low_mpa']),-float(r['ci_high_mpa']));self.assertEqual(float(n['ci_high_mpa']),-float(r['ci_low_mpa']))
   else:
    for k in ['estimate_mpa','ci_low_mpa','ci_high_mpa','difference_definition']:self.assertEqual(n[k],r[k])
   self.assertLessEqual(float(n['ci_low_mpa']),float(n['ci_high_mpa']))
  self.assertEqual(changed,8)

 def test_stage_area_identity(self):
  rows=read_csv(P/'tables/order_paired_contrasts_normalized.csv')
  for contrast in set(r['contrast'] for r in rows):
   rr=[r for r in rows if r['contrast']==contrast];area=next(float(r['estimate_mpa']) for r in rr if r['metric']=='area_mpa')
   self.assertLess(abs(area-sum(float(r['estimate_mpa']) for r in rr if r['metric'].startswith('weighted_stage'))),1e-8)
   objective=next(float(r['estimate_mpa']) for r in rr if r['metric']=='objective_mpa');self.assertLess(abs(objective-area),1e-8)
  main=read_json(P/'figure_data/frozen_data.json');geo=[r for r in main['stage_paired_contrasts'] if r['comparator']=='GEOMETRY_SPREAD' and r['metric']=='weighted_net_mpa']
  gap=float(main['method_summary'][1]['area_mpa'])-float(main['method_summary'][5]['area_mpa']);self.assertLess(abs(sum(float(r['estimate_mpa']) for r in geo)-gap),1e-8)

 def test_random_loss_before_pooling(self):
  predictions=[-2.,2.];target=0.
  loss=sum(abs(p-target) for p in predictions)/2;wrong=abs(sum(predictions)/2-target)
  self.assertEqual(loss,2.);self.assertEqual(wrong,0.)
  self.assertEqual(math.sqrt(sum((p-target)**2 for p in predictions)/2),2.)

 def test_domain_equal_vs_specimen_pooling(self):
  groups=[[0.],[4.,4.,4.]]
  domain=sum(sum(g)/len(g) for g in groups)/len(groups);pool=sum(map(sum,groups))/sum(map(len,groups))
  self.assertEqual(domain,2.);self.assertEqual(pool,3.)
  text=(P/'source/manuscript_en.md').read_text();self.assertIn('six-source-group-equal',text);self.assertIn('specimen-pooled endpoint MAE',text)

 def test_prefix_and_carry(self):
  costs=[0.,.04,.09];pred=[10.,6.,4.]
  self.assertEqual(held(costs,pred,.039),10.);self.assertEqual(held(costs,pred,.04),6.);self.assertEqual(held(costs,pred,.1),4.)
  self.assertAlmostEqual(left_area(costs,pred,3.,.1),4.4)

 def test_quality_first_crossing(self):
  costs=[0.,.1,.2,.3];errors=[10.,4.,8.,3.]
  self.assertEqual(first_quality(costs,errors,5.),{'cost':.1,'recrosses':True})
  self.assertEqual(first_quality(costs,errors,2.)['cost'],None);self.assertEqual(first_quality(costs,errors,12.)['cost'],0.)
  self.assertIsNone(relative_saving(0.,0.));self.assertIsNone(relative_saving(None,.1));self.assertEqual(relative_saving(.1,.2),.5)

 def test_generated_numbers_and_links(self):
  from evidence import display_names
  for source,label in [('74t7kcdgkr','8-layer cross-ply'),('cgtnjyggtm','24-layer quasi-isotropic'),('w68dtmpfyf','16-layer quasi-isotropic'),('xcmzfsbd9t','24-layer cross-ply'),('yfxyg8jm46','16-layer cross-ply'),('ykhs7s2dck','8-layer quasi-isotropic')]:
   self.assertEqual(display_names(source),label)
  self.assertEqual(display_names('74t7kcdgkr:c8-16'),'8-layer cross-ply / c8-16')
  self.assertEqual(display_names('https://data.mendeley.com/datasets/74t7kcdgkr/1.'),'[8-layer cross-ply dataset, version 1](https://data.mendeley.com/datasets/74t7kcdgkr/1).')
  self.assertEqual(display_names('[Data](https://data.mendeley.com/datasets/74t7kcdgkr/1)'),'[Data](https://data.mendeley.com/datasets/74t7kcdgkr/1)')
  from tempfile import TemporaryDirectory
  from verify import inventory
  with TemporaryDirectory() as directory:
   fixture=Path(directory);docs='docs/cai/positive_review_r1'
   for name in ['inputs/current_manuscript.pdf','inputs/current_manuscript.docx','original_extracted.md','original_tables.json','original_media/media/rId26.png','README.md','.gitignore']:
    target=fixture/docs/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b'fixture')
   self.assertEqual({r['path'] for r in inventory(fixture)},{docs+'/README.md',docs+'/.gitignore'})
  from build import _load_sources
  sources=_load_sources(P);self.assertEqual(len(sources.manuscript_en),len(sources.manuscript_zh))
  for source in (P/'source').glob('*.md'):
   visible=re.sub(r'\]\([^)]*\)',']',source.read_text())
   self.assertIsNone(re.search(r'74t7kcdgkr|cgtnjyggtm|w68dtmpfyf|xcmzfsbd9t|yfxyg8jm46|ykhs7s2dck|source\s*id',visible,re.I),source.name)
  table=read_csv(P/'tables/table3.csv');raw=read_csv(ROOT/EVID/'method_summary.csv')
  for r in table[:6]:
   original=next(t for t in raw if t['method']==r['method'])
   for k,key in [('A','area_mpa'),('early_A','early_area_mpa'),('MAE','endpoint_mae_mpa'),('RMSE','endpoint_rmse_mpa'),('R2','endpoint_r2')]:self.assertEqual(float(r[k]),float(original[key]))
  self.assertEqual(f"{float(table[5]['MAE']):.3f}",'42.385');self.assertEqual(f"{float(table[5]['A']):.3f}",'43.597')
  contrasts=read_csv(P/'tables/order_paired_contrasts_normalized.csv')
  for c in ['PRIMARY','SECONDARY']:
   r=next(r for r in contrasts if r['contrast'].startswith(c) and r['metric']=='area_mpa')
   self.assertEqual(float(r['ci_low_mpa'])<0,c=='PRIMARY');self.assertGreater(float(r['ci_high_mpa']),0)
   r=next(r for r in contrasts if r['contrast'].startswith(c) and r['metric']=='early_area_mpa');self.assertLess(float(r['ci_low_mpa']),0)
  main=(P/'source/manuscript_en.md').read_text();self.assertEqual(len(re.findall(r'^\$\$',main,re.M)),13)
  for m in re.findall(r'!\[[^]]*\]\(([^)]+)\)',main+(P/'source/supplementary_en.md').read_text()):self.assertTrue((P/m).is_file(),m)
  for f in range(1,7):
   for ext in ['svg','pdf','png']:self.assertGreater((P/f'figures/F{f}.{ext}').stat().st_size,100)
  for name in ['manuscript','supplementary']:
   for language in ['en','bilingual']:
    for ext in ['docx','pdf']:self.assertGreater((P/f'build/{name}_{language}.{ext}').stat().st_size,1000)
  self.assertEqual(len(read_csv(P/'reference_map.csv')),15);self.assertEqual(len(re.findall(r'^@', (P/'source/references.bib').read_text(),re.M)),15)
  self.assertTrue(all(r['current_citation_context'] and r['current_citation_section'] for r in read_csv(P/'reference_map.csv')))
  splits=read_csv(P/'tables/table1.csv')
  for column,total in [('Train / 训练',161),('Dev. / 开发验证',50),('Reserved / 预留',65),('Total / 合计',276)]:
   self.assertEqual(sum(int(r[column]) for r in splits[:-1]),total);self.assertEqual(int(splits[-1][column]),total)
  domains=read_csv(P/'tables/table5.csv');self.assertEqual(sum(int(r['n']) for r in domains),50)
  self.assertLess(abs(sum(float(r['Proposed A']) for r in domains)/6-float(table[5]['A'])),.001)
  self.assertLess(abs(sum(int(r['n'])*float(r['Proposed MAE']) for r in domains)/50-float(table[5]['MAE'])),.001)
  edit=read_json(A/'TEXT_CHANGE_MAP.json');self.assertTrue(200<=edit['abstract_word_count']<=250)

if __name__=='__main__':unittest.main(verbosity=2)
