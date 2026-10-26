"""Export the TA2 draft and plots from preserved pilot scores; no model inference."""
from pathlib import Path
import csv, re, html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Spacer, Preformatted
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
pdfmetrics.registerFont(TTFont("Times-Roman", "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"))
pdfmetrics.registerFont(TTFont("Times-Bold", "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"))
pdfmetrics.registerFontFamily("Times-Roman", normal="Times-Roman", bold="Times-Bold", italic="Times-Roman", boldItalic="Times-Bold")
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'docs/ta2'
rows=list(csv.DictReader((ROOT/'research/results/pilot_scores.csv').open()))
y=np.array([r['label']=='ai' for r in rows]); scores=np.array([float(r['score']) for r in rows])
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8,2.4));ax.axis('off')
labels=['Browser upload','FastAPI validation','FFmpeg: mono 16 kHz','AASIST windows','Aggregate + abstain','JSON report + cleanup']
for i,label in enumerate(labels):
 x=.17+(i%3)*.33; yy=.77-(i//3)*.53
 ax.text(x,yy,label,ha='center',va='center',bbox=dict(boxstyle='round,pad=.55',fc='#e5f1f2',ec='#087e83'))
 if i in [0,1,3,4]:ax.annotate('',xy=(x+.20,yy),xytext=(x+.13,yy),arrowprops={'arrowstyle':'->'})
ax.annotate('',xy=(.17,.38),xytext=(.83,.63),arrowprops={'arrowstyle':'->','connectionstyle':'angle,angleA=-90,angleB=180,rad=12'})
fig.savefig(OUT/'architecture.png',dpi=180,bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(8,2.5));ax.bar(['Human 1','Human 2','Human 3','Flite slt','Flite awb','Flite rms'],scores,color=['#277a9a']*3+['#bd6939']*3);ax.set_ylim(0,1.08);ax.set_ylabel('Uncalibrated model score');ax.set_title('Six-recording convenience pilot — not a representative benchmark');fig.tight_layout();fig.savefig(OUT/'pilot_scores.png',dpi=180);plt.close(fig)
order=np.argsort(-scores); tp=np.cumsum(y[order]);fp=np.cumsum(~y[order]);tpr=np.r_[0,tp/y.sum()];fpr=np.r_[0,fp/(~y).sum()];rec=np.r_[0,tp/y.sum()];prec=np.r_[1,tp/(tp+fp)]
fig,axs=plt.subplots(1,2,figsize=(8,2.5));axs[0].step(fpr,tpr,where='post');axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set(xlabel='False-positive rate',ylabel='True-positive rate',title='Exploratory ROC, n=6');axs[1].step(rec,prec,where='post');axs[1].set(xlabel='Recall',ylabel='Precision',title='Exploratory precision–recall, n=6');
for ax in axs:ax.set_xlim(-.03,1.03);ax.set_ylim(-.03,1.06)
fig.tight_layout();fig.savefig(OUT/'pilot_curves.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,1.9));matrix=np.array([[0,0,3],[0,0,3]]);ax.imshow(matrix,cmap='Blues',vmin=0,vmax=3,aspect='auto');ax.set_xticks([0,1,2],['Human decision','AI decision','Uncertain']);ax.set_yticks([0,1],['Human label','AI label']);ax.set_title('Default-policy derivation: all six abstain (coverage = 0)');
for i in range(2):
 for j in range(3):ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if j==2 else 'black')
fig.tight_layout();fig.savefig(OUT/'abstention.png',dpi=180);plt.close(fig)

def plain(s):return re.sub(r'\*\*(.*?)\*\*|`([^`]+)`',lambda m:m.group(1) or m.group(2),s)
def fmt(s):
 s=html.escape(s);s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s);s=re.sub(r'`([^`]+)`',r'\1',s);return s.replace('\n','<br/>')
styles={
 'body':ParagraphStyle('body',fontName='Times-Roman',fontSize=10,leading=11.5,spaceAfter=7),
 'heading':ParagraphStyle('heading',fontName='Times-Bold',fontSize=11,leading=13,spaceBefore=7,spaceAfter=6,keepWithNext=True),
 'title':ParagraphStyle('title',fontName='Times-Bold',fontSize=16,leading=18,spaceAfter=12),
 'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=9,leading=10),
 'code':ParagraphStyle('code',fontName='Courier',fontSize=8,leading=10,spaceAfter=8)}
doc=Document();section=doc.sections[0];section.page_width=Inches(8.2677);section.page_height=Inches(11.6929);section.top_margin=section.bottom_margin=Inches(.65);section.left_margin=section.right_margin=Inches(.65)
for name in ['Normal','Heading 1','Heading 2','Title']:
 st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(10 if name=='Normal' else 12);st.paragraph_format.line_spacing=1;st.paragraph_format.space_after=Pt(6)
doc.styles['Title'].font.size=Pt(16)
footer=section.footer.paragraphs[0];footer.text='VoxGuard AI | Mohit Mohatkar | TA2 Draft | '
f=OxmlElement('w:fldSimple');f.set(qn('w:instr'),'PAGE');footer._p.append(f)
story=[]
for pi,page in enumerate((OUT/'TA2_REPORT.md').read_text().split('<!-- PAGE -->')):
 if pi:doc.add_page_break();story.append(PageBreak())
 lines=page.strip().splitlines();i=0
 while i<len(lines):
  line=lines[i]
  if not line.strip():i+=1;continue
  if line.startswith('#'):
   level=len(line)-len(line.lstrip('#'));title=line[level:].strip();doc.add_heading(plain(title),0 if level==1 else 1);story.append(Paragraph(fmt(title),styles['title' if level==1 else 'heading']))
  elif line.startswith('!['):
   filename=re.search(r'\]\(([^)]+)\)',line).group(1);path=OUT/filename
   doc.add_picture(str(path),width=Inches(6.7))
   from PIL import Image as PILImage
   w,h=PILImage.open(path).size;story.append(Image(str(path),width=480,height=480*h/w));story.append(Spacer(1,5))
  elif line.startswith('|'):
   table=[]
   while i<len(lines) and lines[i].startswith('|'):
    row=[x.strip() for x in lines[i].strip('|').split('|')]
    if not all(re.fullmatch(r':?-+:?',v) for v in row):table.append(row)
    i+=1
   t=doc.add_table(rows=0,cols=len(table[0]));t.style='Table Grid'
   for row in table:
    for cell,value in zip(t.add_row().cells,row):
     cell.text=plain(value)
     for p in cell.paragraphs:
      for r in p.runs:r.font.size=Pt(9)
   rpt=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rpt)
   rt=Table([[Paragraph(fmt(x),styles['cell']) for x in row] for row in table],colWidths=[480/len(table[0])]*len(table[0]),repeatRows=1)
   rt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf1f3')),('GRID',(0,0),(-1,-1),.3,colors.gray),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.extend([rt,Spacer(1,8)]);continue
  elif line.startswith('EQUATION:'):
   eq=line.split(':',1)[1].strip();p=doc.add_paragraph();math=OxmlElement('m:oMath');r=OxmlElement('m:r');txt=OxmlElement('m:t');txt.text=eq;r.append(txt);math.append(r);p._p.append(math);story.append(Paragraph(html.escape(eq),styles['body']))
  elif line.startswith('```'):
   i+=1;buf=[]
   while i<len(lines) and not lines[i].startswith('```'):buf.append(lines[i]);i+=1
   p=doc.add_paragraph();r=p.add_run('\n'.join(buf));r.font.name='Courier New';r.font.size=Pt(8);story.append(Preformatted('\n'.join(buf),styles['code']))
  else:
   buf=[line];i+=1
   while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![','EQUATION:','```')):buf.append(lines[i]);i+=1
   content='\n'.join(buf);p=doc.add_paragraph()
   for part in re.split(r'(\*\*.*?\*\*|`[^`]+`)',content):
    r=p.add_run(plain(part));r.bold=part.startswith('**')
   story.append(Paragraph(fmt(content),styles['body']));continue
  i+=1
doc.save(OUT/'VoxGuard_AI_TA2_Report.docx')
def footer(c,d):
 c.setFont('Times-Roman',8);c.drawString(47,25,'VoxGuard AI | Mohit Mohatkar | TA2 Draft');c.drawRightString(548,25,str(d.page))
SimpleDocTemplate(str(OUT/'VoxGuard_AI_TA2_Report.pdf'),pagesize=(595.28,841.89),leftMargin=47,rightMargin=47,topMargin=42,bottomMargin=42,title='VoxGuard AI — Deep Learning TA2 Report',author='Mohit Mohatkar').build(story,onFirstPage=footer,onLaterPages=footer)
print('Exported TA2 report and four evidence-based figures.')
