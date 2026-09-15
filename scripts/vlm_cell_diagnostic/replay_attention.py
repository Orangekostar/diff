"""One-case frozen Qwen diagnostic. Never calls generate or writes the cache.

Only last-query attention rows are explicitly computed from the live SDPA Q/K;
the production SDPA result is returned unchanged. No NxN attention is retained.
"""
from pathlib import Path
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse,ast,csv,json,math,re,subprocess,time,traceback
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import transformers
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53'
ART=ROOT/'artifacts/cai_agent_v3/vlm_cell_diagnostic/r1_8778aa53'
# Read the production model-path configuration without importing a training entry.
_config_ast=ast.parse((ROOT/'src/cmc_bbdm/cai_agent_v3/vlm_perception.py').read_text())
_path_node=next(n.value for n in _config_ast.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODEL_PATH' for t in n.targets))
assert isinstance(_path_node,ast.Call) and isinstance(_path_node.func,ast.Name) and _path_node.func.id=='Path'
MODEL=Path(ast.literal_eval(_path_node.args[0]))

def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def table(path,rows):
 with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def prepare():
 identity=json.loads((OUT/'identity.json').read_text());cache=json.loads((OUT/'cache_record.json').read_text())
 assert identity['specimen_key']=='cgtnjyggtm:q24-48' and identity['input_identity_exact_png_hash']
 assert cache['repaired'] is False and cache['call_count']==1
 assert MODEL.name==cache['request']['model_revision']
 processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True,use_fast=False,min_pixels=256*28*28,max_pixels=1280*28*28)
 images=[Image.open(OUT/f).convert('RGB') for f in ['02_vlm_clean_input.png','03_vlm_numbered_input.png']]
 prompt=(OUT/'prompt.txt').read_text();raw=(OUT/'raw_response.txt').read_text()
 match=re.search(r'"cells"\s*:\s*\[\s*(\d+)',raw);assert match
 prefix=raw[:match.start(1)];first=int(match.group(1));assert not re.search(r'"cells"\s*:\s*\[\s*\d',prefix)
 messages=[{'role':'user','content':[{'type':'image'},{'type':'image'},{'type':'text','text':prompt}]}]
 text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
 base=processor(text=[text],images=images,padding=True,return_tensors='pt')
 assert int(base['attention_mask'].sum())==cache['input_tokens'], 'Historical processor input-token count differs'
 prefix_ids=processor.tokenizer(prefix,add_special_tokens=False)['input_ids']
 full_encoded=processor.tokenizer(raw,add_special_tokens=False)
 # Require token boundary exactly at the target numeral, not merely a substring cut.
 assert full_encoded['input_ids'][:len(prefix_ids)]==prefix_ids
 decoded_before=processor.tokenizer.decode(full_encoded['input_ids'][:len(prefix_ids)],clean_up_tokenization_spaces=False)
 assert decoded_before==prefix
 decoded_through=processor.tokenizer.decode(full_encoded['input_ids'][:len(prefix_ids)+1],clean_up_tokenization_spaces=False)
 assert decoded_through.startswith(prefix) and raw.startswith(decoded_through)
 offset=(len(prefix),len(decoded_through));assert offset[0]==match.start(1)
 assert re.fullmatch(r'\d+',decoded_through[len(prefix):])
 assert processor.tokenizer.decode(prefix_ids,clean_up_tokenization_spaces=False)==prefix
 target_token=full_encoded['input_ids'][len(prefix_ids)]
 total_ids=torch.cat([base['input_ids'],torch.tensor([prefix_ids],dtype=base['input_ids'].dtype)],dim=1)
 base['input_ids']=total_ids;base['attention_mask']=torch.ones_like(total_ids)
 grids=base['image_grid_thw'].tolist();config=json.loads((MODEL/'config.json').read_text());merge=config['vision_config']['spatial_merge_size'];patch_size=config['vision_config']['patch_size']
 positions=(total_ids[0]==config['image_token_id']).nonzero().flatten().tolist();segments=[]
 for pos in positions:
  if not segments or pos!=segments[-1][-1]+1:segments.append([])
  segments[-1].append(pos)
 assert len(segments)==2
 mapping=[]
 for image_index,(grid,segment) in enumerate(zip(grids,segments)):
  t,h,w=grid;assert t==1 and h%merge==0 and w%merge==0;mh,mw=h//merge,w//merge;assert len(segment)==t*mh*mw
  for local,pos in enumerate(segment):
   row,col=divmod(local,mw)
   mapping.append({'image':'clean' if image_index==0 else 'numbered','image_index':image_index,'sequence_position':pos,'merged_token_index':local,'row':row,'col':col,'grid_rows':mh,'grid_cols':mw,'processor_width':w*patch_size,'processor_height':h*patch_size,'processor_x0':col*merge*patch_size,'processor_y0':row*merge*patch_size,'processor_x1':(col+1)*merge*patch_size,'processor_y1':(row+1)*merge*patch_size,'render_x0':col*images[image_index].width/mw,'render_y0':row*images[image_index].height/mh,'render_x1':(col+1)*images[image_index].width/mw,'render_y1':(row+1)*images[image_index].height/mh})
 table(OUT/'attention_token_mapping.csv',mapping)
 tokens=[{'position':i,'token_id':int(token),'token':processor.tokenizer.convert_ids_to_tokens(int(token)),'image_token':int(token)==config['image_token_id'],'query':i==total_ids.shape[1]-1} for i,token in enumerate(total_ids[0])]
 table(OUT/'input_token_sequence.csv',tokens)
 (OUT/'answer_prefix.txt').write_text(prefix);(OUT/'chat_prompt.txt').write_text(text)
 meta={'status':'PREFIX_PREPARED_NO_FORWARD','specimen_key':identity['specimen_key'],'model_path':str(MODEL),'revision':MODEL.name,'transformers':transformers.__version__,'torch':torch.__version__,'processor':type(processor.image_processor).__name__,'min_pixels':256*784,'max_pixels':1280*784,'historical_input_tokens':cache['input_tokens'],'reconstructed_prompt_tokens':cache['input_tokens'],'prefix_tokens':len(prefix_ids),'total_tokens':total_ids.shape[1],'query_index':total_ids.shape[1]-1,'first_cell':first,'target_first_token_id':target_token,'target_first_token_decoded':processor.tokenizer.decode([target_token]),'target_token_offsets_in_raw_text':list(offset),'prefix_excludes_first_number':True,'full_recorded_generation_ids':'NOT_RECORDED; prefix re-tokenization boundary verified, not archived token-ID equality','image_grid_thw':grids,'spatial_merge_size':merge,'patch_size':patch_size,'layers_zero_based':[24,25,26,27],'heads_per_layer':28,'attention_definition':'Explicit FP32 last-query softmax of live post-RoPE repeated-head Q/K; actual SDPA output unchanged; mean all heads then final four decoder layers. No gradients, rollout or candidate interpolation.','mapping_definition':'Qwen2VLImageProcessor grouped 2x2 patches in merged row-major order; visual forward reverses window_index after merger using argsort, restoring that order before image-token masked_scatter. Two image segments handled separately.','estimated_extra_gpu_gib':21.0,'reserve_other_processes_gib':8.0,'use_cache':False,'full_prefix_teacher_forcing':True,'historical_generation':'SDPA greedy generate with cache; this is a new one-shot forward of the truncated recorded prefix, not exact historical cached execution.'}
 dump(OUT/'attention_preflight.json',meta)
 return processor,base,meta,segments

def run(gpu):
 start=time.monotonic();torch.set_num_threads(4)
 status={'status':'ATTENTION_NOT_EXPORTED','actual_qwen_forwards':0,'reason':'Preflight incomplete','new_other_model_forwards':0}
 attempts_path=OUT/'attention_attempts.json'
 previous=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
 assert not previous, 'This single-run entry is resume-safe: inspect existing attempt record; never automatically replay'
 try:
  processor,inputs,meta,segments=prepare()
  gpu_text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.free,utilization.gpu','--format=csv,noheader'],text=True)
  procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory,gpu_uuid','--format=csv,noheader'],text=True)
  processes=[]
  for line in procs.splitlines():
   pid=line.split(',')[0].strip()
   if pid.isdigit():
    try:processes.append(subprocess.check_output(['ps','-p',pid,'-o','pid=,args='],text=True).strip())
    except subprocess.CalledProcessError:pass
  free_mib=float(next(l for l in gpu_text.splitlines() if l.split(',')[0].strip()==str(gpu)).split(',')[2].strip().split()[0])
  # Existing evaluation jobs are retained; do not touch or co-run with training.
  assert not any(re.search(r'(^|[/ ._-])train(ing)?([/ ._-]|$)',p.split(' --checkpoint')[0]) for p in processes),'User training process detected'
  assert free_mib/1024 >= meta['estimated_extra_gpu_gib']+meta['reserve_other_processes_gib'],'Insufficient free memory including reserve'
  dump(ART/'GPU_PREFLIGHT.json',{'gpu':gpu,'gpu_snapshot':gpu_text,'existing_processes':processes,'free_mib':free_mib,'chosen_extra_estimate_gib':21,'reserve_gib':8,'existing_jobs_are_evaluation_not_training':True})
  device=f'cuda:{gpu}'
  model=Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to(device)
  model.eval();model.requires_grad_(False)
  assert len(model.model.layers)==28
  assert not model.config.use_sliding_window
  # Verify vision-window reversal against a synthetic merged-index sequence, no vision forward.
  window,_=model.visual.get_window_index(inputs['image_grid_thw']);indices=torch.arange(len(window));assert torch.equal(indices[window][torch.argsort(window)],indices)
  rows={};diagnostics={};sdpa=torch.nn.functional.scaled_dot_product_attention
  for idx in [24,25,26,27]:
   attn=model.model.layers[idx].self_attn;original=attn.forward
   assert type(attn).__name__=='Qwen2_5_VLSdpaAttention'
   def wrapped(*args,_original=original,_idx=idx,**kwargs):
    def capture(q,k,v,attn_mask=None,dropout_p=0.0,is_causal=False,**options):
     assert q.shape[0]==1 and q.shape[1]==28 and q.shape[-2]==meta['total_tokens'] and k.shape[-2]==q.shape[-2]
     assert dropout_p==0.0 and not options
     score=torch.matmul(q[:,:,-1:,:].float(),k.float().transpose(-2,-1))/math.sqrt(q.shape[-1])
     if attn_mask is not None:
      mask=attn_mask[...,-1:,:]
      score=score.masked_fill(~mask,float('-inf')) if mask.dtype==torch.bool else score+mask.float()
     # Last query in an unpadded square causal sequence may see every key.
     weights=torch.softmax(score,dim=-1)
     result=sdpa(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
     reconstructed=torch.matmul(weights,v.float());err=(reconstructed-result[:,:,-1:,:].float()).abs()
     rows[_idx]=weights[0,:,0,:].mean(0).cpu().numpy()
     diagnostics[_idx]={'heads':q.shape[1],'query_index':q.shape[-2]-1,'key_count':k.shape[-2],'head_sum_max_abs_error':float((weights.sum(-1)-1).abs().max()),'row_output_reconstruction_max_abs':float(err.max()),'row_output_reconstruction_rms':float(err.square().mean().sqrt()),'sdpa_output_abs_max':float(result[:,:,-1:,:].abs().max()),'sdpa_is_causal':is_causal,'mask_present':attn_mask is not None,'captured_shape':[1,28,1,k.shape[-2]]}
     return result
    with patch('torch.nn.functional.scaled_dot_product_attention',capture):return _original(*args,**kwargs)
   attn.forward=wrapped
  inputs=inputs.to(device)
  assert time.monotonic()-start<840,'Preflight/load consumed diagnostic time limit'
  attempts=[{'attempt':1,'status':'STARTED','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'forward_type':'full prefix, no generate','gpu':gpu}]
  dump(attempts_path,attempts);status['actual_qwen_forwards']=1
  torch.cuda.reset_peak_memory_stats(device)
  with torch.inference_mode():output=model(**inputs,use_cache=False,output_attentions=False,return_dict=True)
  torch.cuda.synchronize(device)
  assert set(rows)=={24,25,26,27}
  mean=np.stack([rows[i] for i in [24,25,26,27]]).mean(0);assert np.isfinite(mean).all() and abs(float(mean.sum())-1)<1e-5
  np.save(OUT/'attention_headmean_last4.npy',np.stack([rows[i] for i in [24,25,26,27]]));np.save(OUT/'attention_mean_query.npy',mean)
  for name,grid,segment in zip(['clean','numbered'],meta['image_grid_thw'],segments):
   attention=mean[segment].reshape(grid[1]//2,grid[2]//2);np.save(OUT/f'attention_{name}.npy',attention)
  masses={name:float(mean[seg].sum()) for name,seg in zip(['clean','numbered'],segments)};masses['non_visual']=float(mean.sum()-sum(masses.values()));masses['total']=float(mean.sum());masses['scale']='raw probability mass, summed after head/layer mean; images NOT individually normalized';dump(OUT/'attention_mass.json',masses)
  logits=output.logits[0,-1].float();prob=torch.softmax(logits,dim=-1);top=prob.topk(10);topk=[{'token_id':int(i),'decoded':processor.tokenizer.decode([int(i)],clean_up_tokenization_spaces=False),'probability':float(v)} for i,v in zip(top.indices,top.values)]
  target=meta['target_first_token_id'];match=topk[0]['token_id']==target
  dump(OUT/'next_token.json',{'historical_first_cell':meta['first_cell'],'historical_first_digit_token_id':target,'historical_token_probability':float(prob[target]),'historical_token_rank':int((prob>prob[target]).sum())+1,'top10':topk,'status':'REPLAY_FIRST_TOKEN_MATCH' if match else 'REPLAY_TOKEN_MISMATCH','not_a_new_complete_response':True})
  dump(OUT/'attention_capture_checks.json',diagnostics)
  attempts[0]['status']='COMPLETED';dump(attempts_path,attempts)
  status.update(status='ATTENTION_EXPORTED',reason='True diagnostic query-row attention extracted from one new frozen Qwen forward; not historical saved attention.',replay_token_status='REPLAY_FIRST_TOKEN_MATCH' if match else 'REPLAY_TOKEN_MISMATCH',gpu=gpu,elapsed_seconds=time.monotonic()-start,max_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,implementation=meta['attention_definition'])
 except Exception as error:
  status.update(reason=f'{type(error).__name__}: {error}',elapsed_seconds=time.monotonic()-start)
  (ART/'attention_error.txt').write_text(traceback.format_exc())
  if attempts_path.exists():
   attempts=json.loads(attempts_path.read_text());attempts[-1]['status']='FAILED';attempts[-1]['error']=status['reason'];dump(attempts_path,attempts)
 finally:
  dump(OUT/'attention_status.json',status);print(json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--gpu',type=int,default=0);args=ap.parse_args()
 if args.prepare_only:print(json.dumps(prepare()[2],ensure_ascii=False))
 else:run(args.gpu)
