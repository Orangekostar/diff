"""Finite local validation and nonrecursive delivery inventory."""
from pathlib import Path
import json,subprocess,sys
from evidence import digest,read_json,read_csv,write_json,DOCS,TASK,SHA

ROOTS=['paper_cai_aei/r3_positive_review_r1','artifacts/cai_agent_v3/manuscript/r3_positive_review_r1',DOCS,'scripts/cai_manuscript_revision']

def inventory(repo):
    rows=[]
    for root in ROOTS:
        for p in sorted((repo/root).rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name in ['DELIVERY_MANIFEST.json','PUBLICATION_RECEIPT.json'] or 'page_renders' in p.parts:continue
            if root==DOCS and p.relative_to(repo/root).parts[0] in {'inputs','original_extracted.md','original_media','original_tables.json'}:continue
            rows.append(dict(path=str(p.relative_to(repo)),bytes=p.stat().st_size,sha256=digest(p)))
    return rows


def verify(paper,art):
    paper=Path(paper);art=Path(art);repo=paper.parents[1]
    result=subprocess.run([sys.executable,str(repo/'scripts/cai_manuscript_revision/test_revision.py')],cwd=repo,capture_output=True,text=True)
    (art/'TEST_RESULTS.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Eight bounded checks failed; see TEST_RESULTS.log')
    bindings=read_json(art/'SOURCE_BINDINGS.json')
    for row in bindings['records']:
        path=Path(row['path']);path=path if path.is_absolute() else repo/path
        if digest(path)!=row['sha256']:raise ValueError(f'Frozen source hash changed: {path}')
    report=read_json(art/'BUILD_REPORT.json')
    if report['status']!='success' or len(report['outputs'])!=8:raise ValueError('Missing build output')
    visual=read_json(art/'DOCUMENT_VISUAL_REVIEW.json')
    if len(visual['documents'])!=4:raise ValueError('Missing PDF visual review')
    for document in visual['documents']:
        if digest(paper/document['file'])!=document['sha256']:raise ValueError('PDF changed after visual review')
        if not document['pages'] or any(p['review_status']!='PASS' or p['outside_page_blocks'] for p in document['pages']):raise ValueError('Unresolved PDF page review')
    if len(visual['docx_structure'])!=4 or any(d['zip_crc_status']!='PASS' for d in visual['docx_structure']):raise ValueError('Invalid editable DOCX')
    for document in visual['docx_structure']:
        if document['file'].startswith('manuscript') and (document['display_equations'],document['tables'],document['images'])!=(13,5,6):raise ValueError('Main document structure changed')
    panels=read_csv(art/'FIGURE_PANEL_REVIEW.csv')
    if len(panels)!=40 or any(p['manual_status']!='PASS' or int(p['automatic_collision_failures']) for p in panels):raise ValueError('Unresolved figure panel review')
    for name in ['REVIEW.md','CODEX_HANDOFF_CAI_POSITIVE_REVIEW.md']:
        if not (art/name).is_file():raise ValueError(f'Missing delivery review: {name}')
    for f in (art/'figure_qa').glob('*.collision.json'):
        data=read_json(f)
        if data['summary']['fail']:raise ValueError(f'Unresolved collision: {f}')
    for f in (art/'figure_qa').glob('*.text.json'):
        if read_json(f)['below_minimum_count']:raise ValueError(f'Figure glyph below requested floor: {f}')
    for f in (paper/'build').glob('*.log'):
        if 'Missing character:' in f.read_text():raise ValueError(f'Missing glyph: {f}')
    outputs={k:{'bytes':v['bytes'],**({'pages':v['pdf_pages']} if 'pdf_pages' in v else {'omath_count':v['omath_count']})} for k,v in report['outputs'].items()}
    write_json(art/'LOCAL_VALIDATION.json',{'task_id':TASK,'tests':8,'test_exit_code':result.returncode,'test_scope':'test_revision.py only','build_status':report['status'],'outputs':outputs,'frozen_sources_unchanged':True,'reviewed_pdf_pages':sum(len(d['pages']) for d in visual['documents']),'reviewed_figure_panels':len(panels),'qa':{**{f'Q{i:02}':'PASS' for i in range(1,8)},'Q05':'PASS_WITH_REVIEWED_WARNINGS','Q08':'LOCAL_SCOPE_PASS; REMOTE_RECEIPT_PENDING'},'research_activity_counts':bindings['activities'],'publication_status':'LOCAL_VERIFIED_PENDING_PUSH'})
    manifest={'task_id':TASK,'evidence_sha':SHA,'execution_base_sha':bindings['execution_base_sha'],'model_lock':'MODEL_IDENTITY_LOCK.json','artifact_status':'REVISION_COMPLETE','scientific_scope':'selected_development_configuration_only','scientific_status':'SELECTED_DEVELOPMENT_CONFIGURATION_ONLY','publication_status':'LOCAL_VERIFIED_PENDING_PUSH','build_status':report['status'],'qa':'LOCAL_VALIDATION.json and REVIEW.md; remote status only in later publication receipt','files':inventory(repo)}
    write_json(art/'DELIVERY_MANIFEST.json',manifest)
    return manifest
