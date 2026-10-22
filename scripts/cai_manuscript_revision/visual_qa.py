"""Render final PDFs for page-by-page human review; inspect editable DOCX structure."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET
import fitz
from PIL import Image,ImageDraw
from evidence import read_json,write_json,digest


def render_pages(paper,art):
    paper=Path(paper);art=Path(art);out=art/'page_renders';out.mkdir(exist_ok=True)
    reports=[]
    for old in out.glob('*.jpg'):old.unlink()
    for pdf in sorted((paper/'build').glob('*.pdf')):
        with fitz.open(pdf) as doc:
            thumbs=[];rows=[]
            for i,page in enumerate(doc):
                pix=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
                im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
                image=Image.new('RGB',(im.width,im.height+25),'#e6e9ec');image.paste(im,(0,25));ImageDraw.Draw(image).text((8,6),f'{pdf.stem}  page {i+1}',fill='black')
                thumbs.append(image)
                blocks=page.get_text('blocks');overflow=[b[:4] for b in blocks if b[0]<0 or b[1]<0 or b[2]>page.rect.width+.5 or b[3]>page.rect.height+.5]
                rows.append(dict(page=i+1,text_characters=len(page.get_text()),outside_page_blocks=overflow,review_status='PENDING_MANUAL'))
            for start in range(0,len(thumbs),4):
                sheet=Image.new('RGB',(thumbs[0].width*2,thumbs[0].height*2),'#d5dade')
                for j,img in enumerate(thumbs[start:start+4]):sheet.paste(img,((j%2)*img.width,(j//2)*img.height))
                sheet.save(out/f'{pdf.stem}_{start+1:02d}-{min(start+4,len(thumbs)):02d}.jpg',quality=90)
            reports.append(dict(file=str(pdf.relative_to(paper)),sha256=digest(pdf),pages=rows))
    structure=[]
    for docx in sorted((paper/'build').glob('*.docx')):
        with ZipFile(docx) as z:
            xml=ET.fromstring(z.read('word/document.xml'));ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
            structure.append(dict(file=docx.name,zip_crc_status='PASS' if z.testzip() is None else 'FAIL',tables=len(xml.findall('.//w:tbl',ns)),images=len(xml.findall('.//w:drawing',ns)),equations=len(xml.findall('.//m:oMath',ns)),display_equations=len(xml.findall('.//m:oMathPara',ns))))
    write_json(art/'DOCUMENT_VISUAL_REVIEW.json',{'documents':reports,'docx_structure':structure,'note':'Contact sheets are temporary QA renders, excluded from delivery inventory. Manual status is assigned only after actual inspection.'})
    return reports

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    render_pages(root/'paper_cai_aei/r3_positive_review_r1',root/'artifacts/cai_agent_v3/manuscript/r3_positive_review_r1')
