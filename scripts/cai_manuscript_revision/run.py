"""Isolated frozen-evidence manuscript revision CLI. No research execution."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.cai_manuscript_revision.evidence import bind,prepare_evidence,validate_roots

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('command',choices=['bind','prepare-evidence','render-figures','build','verify'])
 parser.add_argument('--repo',type=Path); parser.add_argument('--packet',type=Path)
 parser.add_argument('--paper-root',type=Path,required=True); parser.add_argument('--artifact-root',type=Path,required=True)
 parser.add_argument('--languages',default='en,bilingual')
 args=parser.parse_args()
 for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='4'
 repo,paper,art=validate_roots(args.paper_root,args.artifact_root)
 if args.command=='bind':
  if not args.repo or not args.packet: parser.error('bind requires --repo and --packet')
  bind(args.repo,args.packet,paper,art)
 elif args.command=='prepare-evidence':
  prepare_evidence(paper,art)
  from text import prepare_text
  from supplement import prepare_supplement
  prepare_supplement(paper,art,prepare_text(paper,art))
 elif args.command=='render-figures':
  from scripts.cai_manuscript_revision.figures import render_figures
  render_figures(paper,art)
 elif args.command=='build':
  from scripts.cai_manuscript_revision.build import build_documents
  build_documents(paper,art,args.languages)
 elif args.command=='verify':
  from scripts.cai_manuscript_revision.verify import verify
  verify(paper,art)
 print(f'{args.command}: completed')

if __name__=='__main__': main()
