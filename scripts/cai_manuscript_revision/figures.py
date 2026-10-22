"""Publication displays from frozen CSV/JSON and already saved images only."""
from pathlib import Path
import json, sys, subprocess, string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image
from evidence import read_json, write_json, write_csv, display_names, METHODS, LABELS

SKILL=Path('/home/ww/.codex/skills/nature-figure/scripts')
if SKILL.is_dir(): sys.path.insert(0,str(SKILL))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Liberation Sans','DejaVu Sans'],'font.size':9.5,'axes.labelsize':9.5,'axes.titlesize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'legend.frameon':False,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'savefig.dpi':300})
COLORS=['#7C8792','#C07A20','#9275AB','#4B9A89','#6579B5','#0072B2']
STYLES=['--','-.',':','--','-.','-']
CONTRACTS={
'F1':('A fixed-weight observation loop connects regional requests to CAI estimates.','schematic-led composite','surface context / policy / observation-state / assessment; training-only truth; illustrative update'),
'F2':('Unknown input is represented explicitly in actor scoring and independently masked in prediction.','schematic-led composite','observable representation / token context / feasibility / independent set readout'),
'F3':('The policy reaches useful pooled prediction quality early, with local recrossings.','quantitative grid','a: six-method full event curves; b: proposed local nonmonotonicity'),
'F4':('First attainment depends on quality target and reporting resolution.','quantitative grid','a: five caps; b: event union; all 41–61 targets and NR/zero retained'),
'F5':('Early realized reductions support process gains, while primary order uncertainty remains.','quantitative grid','a: raw early paired effects; b: weighted stage closure; c: fixed-set paths; d: primary/secondary order intervals'),
'F6':('Real-image regional requests produce specimen-specific updates and retain a large residual case.','image plate + quant','a/d: known surface; b/e: retrospective C-scan and selected ordinals; c/f: actual saved strength trajectories')}


def label(ax,letter,title):
    ax.set_title(title,loc='left',pad=12)
    ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-18,12),textcoords='offset points',fontweight='bold',fontsize=11,annotation_clip=False)


def save(fig,name,paper,art,exclude=()):
    out=paper/'figures'; out.mkdir(exist_ok=True)
    qa=art/'figure_qa'; qa.mkdir(exist_ok=True)
    fig.canvas.draw()
    if SKILL.is_dir():
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig,json_out=qa/f'{name}.alignment.json',strict=True,exclude_axes=exclude)
    else:
        write_json(qa/f'{name}.alignment.json',{'status':'AUTOMATED_ALIGNMENT_TOOL_UNAVAILABLE','manual_required':True})
    fig.savefig(out/f'{name}.svg')
    svg=out/f'{name}.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig.savefig(out/f'{name}.pdf')
    fig.savefig(out/f'{name}.png',dpi=300)
    plt.close(fig)
    for script,args in [('audit_pdf_text.py',['--min-pt','8','--json']),('audit_figure_collisions.py',['--json-out',str(qa/f'{name}.collision.json')])]:
        if not (SKILL/script).exists(): continue
        result=subprocess.run([sys.executable,str(SKILL/script),str(out/f'{name}.pdf'),*args],capture_output=True,text=True)
        (qa/f'{name}.{script}.log').write_text((result.stdout+'\n'+result.stderr).rstrip()+'\n')
        if script=='audit_pdf_text.py':
            try: write_json(qa/f'{name}.text.json',json.loads(result.stdout))
            except ValueError: pass
        if result.returncode not in [0,1,2]: raise RuntimeError(f'{name}: {script} failed')


def box(ax,xy,w,h,text,color='#EDF3F7'):
    x,y=xy; ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.012',fc=color,ec='#6B7C86',lw=.8))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',linespacing=1.5)


def arrow(ax,start,end,**kw):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=11,color='#465B66',lw=1,**kw))


def schematic1(paper,art,cases):
    fig,ax=plt.subplots(figsize=(7.2,4.3)); fig.subplots_adjust(left=.02,right=.98,bottom=.04,top=.95)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    xs=[.025,.275,.525,.775]
    texts=['Surface context\nLocal descriptors\nFrozen encoder','Regional policy\nVisible state\nFeasible request','Observe / update\nReveal one region\nMask and history','CAI predictor\nFrozen readout\nUpdated estimate']
    for x,t in zip(xs,texts): box(ax,(x,.59),.20,.30,t)
    for x in xs[:-1]: arrow(ax,(x+.21,.74),(x+.245,.74))
    arrow(ax,(.87,.57),(.87,.48));arrow(ax,(.87,.48),(.37,.48));arrow(ax,(.37,.48),(.37,.57))
    ax.text(.63,.425,'Next decision: new evidence, progress and prediction',ha='center')
    ax.text(.5,.955,'Execution: fixed weights; measured strength is unavailable',ha='center',fontweight='bold')
    # One update is illustrative, with actual saved predictions for c8-16.
    c=cases[0]
    box(ax,(.03,.08),.43,.25,f"One saved update: c8-16\nRequest cell {c['actions'][0]} (zero-based index)\nCAI: {c['predictions_mpa'][0]:.3f} → {c['predictions_mpa'][1]:.3f} MPa",'#F3F6F8')
    box(ax,(.57,.08),.39,.25,'Training only\nMeasured strength → trajectory cost\nProcess A + 0.25 × terminal error','#FFF3DD')
    arrow(ax,(.77,.35),(.77,.395),linestyle='--')
    save(fig,'F1',paper,art)


def schematic2(paper,art):
    fig,ax=plt.subplots(figsize=(7.2,4.7));fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98)
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,(.03,.70),.25,.23,'Known surface descriptors\nInternal: measured x or\nlearned unknown u\nMask / position / ordinal')
    box(ax,(.39,.70),.25,.23,'64 regional tokens\n102 inputs → width 128\n+ one global token\n6 inputs → width 128')
    box(ax,(.75,.70),.22,.23,'Transformer\n2 layers, 4 heads\nFeed-forward 256\n65 contextual tokens')
    arrow(ax,(.295,.815),(.375,.815));arrow(ax,(.655,.815),(.735,.815))
    box(ax,(.75,.36),.22,.21,'64 regional scores\n+ feasible-cell mask\nArgmax → next request')
    arrow(ax,(.86,.68),(.86,.59))
    box(ax,(.03,.34),.35,.23,'Independent strength predictor\nMask internal x before affine map\nMean-all + mean-measured\nHistory excluded from readout','#F5F5F5')
    box(ax,(.42,.08),.40,.17,'Global state: area / remaining area\nCurrent CAI / measured-cell fraction\nArea fraction ≠ cell fraction','#FFF3DD')
    arrow(ax,(.39,.45),(.40,.26));arrow(ax,(.40,.26),(.43,.20));arrow(ax,(.51,.27),(.51,.68))
    ax.text(.03,.19,'Unknown ≠ observed zero.\nTwo zero VLM channels per block.\nSee S2 for the exact input layout.',va='center')
    save(fig,'F2',paper,art)


def curves(data,family,series):
    rows=[r for r in data['cost_error_curves'] if r['identity_family']==family and r['series']==series]
    rows.sort(key=lambda r:float(r['requested_cost']))
    x=[float(r['requested_cost']) for r in rows];y=[float(r['mae_mpa']) for r in rows]
    assert all(a<b for a,b in zip(x,x[1:]))
    return x,y


def f3(paper,art,data):
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.65));fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.75,wspace=.32)
    for i,(m,label_) in enumerate(zip(METHODS,LABELS)):
        x,y=curves(data,'ARCHIVED_METHODS',m)
        axs[0].step(x,y,where='post',color=COLORS[i],ls=STYLES[i],lw=1.7 if i==5 else .9,label=label_)
        r=[r for r in data['same_cap_metrics'] if r['identity_family']=='ARCHIVED_METHODS' and r['series']==m]
        axs[0].plot([float(a['requested_cost']) for a in r],[float(a['mae_mpa']) for a in r],ls='none',marker='o',ms=3,color=COLORS[i])
    x,y=curves(data,'ARCHIVED_METHODS',METHODS[-1]); local=[(a,b) for a,b in zip(x,y) if .06<a<.14]; from evidence import held
    zx=[.06,*[a for a,b in local],.14];zy=[held(x,y,.06),*[b for a,b in local],held(x,y,.14)];axs[1].step(zx,zy,where='post',lw=1.4,color=COLORS[-1])
    for ax in axs:
        ax.axhline(data['full_reference']['mae'],color='#777777',ls=':',lw=1)
        ax.axvline(.0625,color='#AAAAAA',ls=':',lw=.7);ax.axvline(.25,color='#AAAAAA',ls=':',lw=.7)
        ax.set_xlabel('Common area cap');ax.set_ylabel('Cohort MAE (MPa)')
        ax.grid(axis='y',alpha=.15)
    axs[0].set(xlim=(0,.25),ylim=(40,61),xticks=[0,.0625,.125,.25],xticklabels=['0','.0625','.125','.25'])
    axs[1].axhline(42,color='#C07A20',ls='--',lw=.8)
    axs[1].set(xlim=(.06,.14),ylim=(40.8,45.5),xticks=[.0625,.10,.14],xticklabels=['.0625','.10','.14'])
    label(axs[0],'a','Saved event curves');label(axs[1],'b','Proposed: local recrossings')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1.01),columnspacing=1.3,handlelength=2)
    fig.text(.52,.025,'Dotted horizontal line: full input at cost 1.0 (41.690 MPa)',ha='center',fontsize=9)
    save(fig,'F3',paper,art)


def f4(paper,art,data):
    fig,axs=plt.subplots(1,2,figsize=(7.2,4.1));fig.subplots_adjust(left=.09,right=.98,bottom=.17,top=.77,wspace=.27)
    src=[]
    for ax,grid,letter,title in zip(axs,['FIVE_FIXED_CAPS','FULL_EVENT_UNION'],'ab',['Five-cap first attainment','Event first attainment']):
        for i,m in enumerate(METHODS):
            local=[]
            for q in range(41,62):
                r=next(r for r in data['matched_quality'] if r['identity_family']=='ARCHIVED_METHODS' and r['grid_type']==grid and float(r['target_mpa'])==q and (r['main_series']==m or r['comparator_series']==m))
                v=r['main_cost'] if r['main_series']==m else r['comparator_cost']; v=float(v) if v else None
                local.append(v);src.append(dict(grid=grid,method=m,target_mpa=q,first_cost=v,status='NR' if v is None else 'ZERO' if v==0 else 'REACHED'))
            x=np.arange(41,62);y=[v if v is not None else np.nan for v in local]
            ax.plot(x,y,color=COLORS[i],ls=STYLES[i],lw=1.5 if i==5 else .8,marker='o',ms=3,label=LABELS[i])
            # A dedicated NR band preserves nonattainment without false numeric caps.
            ax.scatter([q for q,v in zip(x,local) if v is None],[.28+i*.009 for v in local if v is None],s=13,marker='x',color=COLORS[i],linewidths=.7)
        ax.axhline(.265,color='#BBBBBB',lw=.7);ax.set(xlim=(40.5,61.5),ylim=(-.015,.342),xticks=[41,45,47,50,55,58,61],yticks=[0,.0625,.125,.1875,.25,.305],yticklabels=['0','.0625','.125','.1875','.25','NR'])
        ax.set_xlabel('Target cohort MAE (MPa)');ax.set_ylabel('First common cap');ax.grid(axis='y',alpha=.15);label(ax,letter,title)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=3,bbox_to_anchor=(.5,1),columnspacing=1.3)
    write_csv(paper/'figure_data/F4_quality_display.csv',src)
    save(fig,'F4',paper,art)


def effect(ax,rows,labels_,color):
    e=np.array([float(r['estimate_mpa']) for r in rows]);lo=np.array([float(r['ci_low_mpa']) for r in rows]);hi=np.array([float(r['ci_high_mpa']) for r in rows]);yy=np.arange(len(rows))
    ax.errorbar(e,yy,xerr=np.vstack((e-lo,hi-e)),fmt='o',color=color,capsize=3,lw=1,ms=4)
    ax.axvline(0,color='#999999',ls=':',lw=.8);ax.set_yticks(yy,labels_);ax.invert_yaxis();ax.margins(y=.4)


def f5(paper,art,data):
    fig,axs=plt.subplots(2,2,figsize=(7.2,6.6));fig.subplots_adjust(left=.13,right=.98,bottom=.18,top=.92,hspace=.65,wspace=.47)
    a,b,c,d=axs.flat
    rows=[next(r for r in data['stage_paired_contrasts'] if r['comparator']==m and r['metric']=='raw_net_mpa' and r['stage']=='1') for m in ['GEOMETRY_SPREAD','RANDOM']]
    effect(a,rows,['Geometry','Random'],'#0072B2');a.set_xlabel('Proposed − control gain (MPa)');label(a,'a','Early raw net decrease')
    rows=[next(r for r in data['stage_paired_contrasts'] if r['comparator']=='GEOMETRY_SPREAD' and r['metric']=='weighted_net_mpa' and r['stage']==str(s)) for s in range(1,5)]
    effect(b,rows,['Stage 1','Stage 2','Stage 3','Stage 4'],'#0072B2');b.set_xlabel('Proposed − geometry (MPa)');label(b,'b','Weighted stage contributions')
    for series,lab,col,ls in [('NATIVE_REPLAY','Original','#0072B2','-'),('PERMUTED_MEAN','Permutation mean','#9275AB','--'),('REVERSE','Reverse','#C07A20','-.')]:
        x,y=curves(data,'FIXED_SET_ORDERS',series);c.step(x,y,where='post',label=lab,color=col,ls=ls,lw=1.2)
    c.set(xlim=(0,.25),ylim=(40,62),xticks=[0,.125,.25]);c.set_xlabel('Common area cap');c.set_ylabel('Cohort MAE (MPa)');c.legend(fontsize=9,loc='upper right',handlelength=1.8);label(c,'c','Same final sets; changed arrival')
    rows=[next(r for r in data['order_contrasts'] if r['contrast'].startswith(prefix) and r['metric']=='area_mpa') for prefix in ['PRIMARY','SECONDARY']]
    effect(d,rows,['Primary:\npermutation','Secondary:\nreverse'],'#0072B2');d.set_xlabel('Comparator − original A (MPa)');label(d,'d','Order: primary and secondary')
    fig.text(.52,.025,'Bars: saved paired 95% intervals; 50 specimens / 48 capture groups\na,b,d: six-source-group equal; c: specimen-pooled MAE',ha='center',fontsize=9,linespacing=1.5)
    save(fig,'F5',paper,art)


def f6(paper,art,data,cases,assets):
    fig,axs=plt.subplots(2,3,figsize=(7.2,5.5));fig.subplots_adjust(left=.05,right=.98,bottom=.20,top=.92,wspace=.30,hspace=.66)
    for ri,(case,asset) in enumerate(zip(cases[:2],assets)):
        key=case['specimen_key'];sid=key.split(':')[1]
        surf,scan,trace=axs[ri]
        surf.imshow(Image.open(paper/asset['surface']['output_path']));surf.axis('off');surf.set_box_aspect(1)
        label(surf,string.ascii_lowercase[ri*3],f'{sid}: surface')
        image=np.asarray(Image.open(paper/asset['cscan']['output_path']).convert('RGB'));h,w=image.shape[:2];size=max(h,w)
        canvas=np.full((size,size,3),255,dtype=np.uint8);canvas[:h,:w]=image
        scan.imshow(canvas);scan.set(xlim=(0,size),ylim=(size,0));scan.set_box_aspect(1);scan.axis('off')
        # Match the source crop's integer grid exactly; preserve unmodified pixels beneath overlays.
        yy=np.rint(np.linspace(0,h,9)).astype(int);xx=np.rint(np.linspace(0,w,9)).astype(int)
        cohort=next(r for r in data['cohort_manifest'] if r['specimen_key']==key)
        counts=[(yy[r+1]-yy[r])*(xx[c+1]-xx[c]) for r in range(8) for c in range(8)]
        assert counts==list(map(int,cohort['cell_pixel_counts'].split(';')))
        order={v:i+1 for i,v in enumerate(case['actions'])}
        for cell in range(64):
            r,c=divmod(cell,8);x,y=xx[c],yy[r];cw,ch=xx[c+1]-x,yy[r+1]-y
            if cell not in order: scan.add_patch(Rectangle((x,y),cw,ch,fc='#E4E8EB',alpha=.84,ec='none'))
            else:
                scan.add_patch(Rectangle((x,y),cw,ch,fc='white',alpha=.50,ec='none'))
                scan.text(x+cw/2,y+ch/2,str(order[cell]),ha='center',va='center',fontsize=9,color='#003C61',fontweight='bold')
        label(scan,string.ascii_lowercase[ri*3+1],f'{sid}: C-scan + order')
        costs=case['costs'];pred=case['predictions_mpa'];trace.step([*costs,.25],[*pred,pred[-1]],where='post',color='#0072B2',lw=1.5)
        trace.axhline(case['target_mpa'],ls='--',color='#777777',lw=1)
        trace.set(xlim=(0,.25),xticks=[0,.125,.25],ylim=(110,325));trace.set_xlabel('Acquired area fraction');trace.set_ylabel('CAI (MPa)');trace.set_box_aspect(1)
        label(trace,string.ascii_lowercase[ri*3+2],f'{sid}: strength')
    fig.text(.52,.025,'Blue: saved prediction; dashed: measured CAI (retrospective scoring only)\nC-scan backdrop: retrospective full-image reference; never supplied to policy',ha='center',fontsize=9,linespacing=1.5)
    save(fig,'F6',paper,art)


def supplementary_case_figures(paper,art,cases):
    for j,case in enumerate(cases,1):
        fig,axs=plt.subplots(1,4,figsize=(7.2,2.65));fig.subplots_adjust(left=.05,right=.99,bottom=.29,top=.73,wspace=.35)
        logits=np.array(case['initial_logits']);prob=np.exp(logits-logits.max());prob/=prob.sum()
        att=np.array(case['diagnostics']['attention.npz']['query_to_cells'])[0]
        attr=np.array(case['diagnostics']['surface_attribution.npz']['total_l1'])[0]
        arrays=[prob,attr,att]
        for ax,arr,title,letter in zip(axs[:3],arrays,['Action preference','Surface sensitivity','Token attention'],'abc'):
            ax.imshow(arr.reshape(8,8),cmap='cividis',interpolation='nearest',aspect='auto');ax.set(xticks=[],yticks=[]);ax.set_box_aspect(1);label(ax,letter,title)
        axs[3].step([*case['costs'],.25],[*case['predictions_mpa'],case['predictions_mpa'][-1]],where='post',color='#0072B2',lw=1.2);axs[3].axhline(case['target_mpa'],ls='--',color='#777777',lw=.8)
        axs[3].set(xlim=(0,.25),xticks=[0,.25]);axs[3].set_box_aspect(1);label(axs[3],'d','CAI updates')
        fig.text(.5,.025,display_names(case['specimen_key'])+'\nEach map uses its own range; no cross-map quantitative scale.',ha='center',fontsize=9,linespacing=1.4)
        save(fig,f'S5_{j}',paper,art)


def render_figures(paper,art):
    paper=Path(paper);art=Path(art);data=read_json(paper/'figure_data/frozen_data.json');cases=read_json(paper/'figure_data/cases.json');assets=read_json(paper/'figure_data/assets.json')
    write_json(art/'FIGURE_CONTRACTS.json',{k:{'conclusion':v[0],'archetype':v[1],'panels':v[2],'backend':'Python/matplotlib','size_width_in':7.2,'minimum_rendered_text_pt_target':8,'formats':['svg','pdf','png']} for k,v in CONTRACTS.items()})
    schematic1(paper,art,cases);schematic2(paper,art);f3(paper,art,data);f4(paper,art,data);f5(paper,art,data);f6(paper,art,data,cases,assets);supplementary_case_figures(paper,art,cases)
    return {'figures':6,'supplementary_case_plates':6}
