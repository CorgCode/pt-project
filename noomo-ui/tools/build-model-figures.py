"""Generate model diagrams and research charts from the checked-in scoreboard.
Requires matplotlib. Outputs browser-ready SVG and desktop PNG previews.
"""
from pathlib import Path
import hashlib, json, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'noomo-ui/model/assets';OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'scoreboard/README.md'
FONT=FontProperties(fname=str(ROOT/'noomo-ui/fonts/onest-400.ttf'))
BOLD=FontProperties(fname=str(ROOT/'noomo-ui/fonts/onest-600.ttf'))
INK='#181520';MUTED='#626671';LINE='#9da5b8';PAPER='#c9d2e7';ACCENT='#8231fe';SOFT='#e5e8f1'
COLORS=[MUTED,INK,ACCENT]
IDS=['temporal_encoder','multiday_codebook','multiday_codebook_plus']
NAMES=['Временной кодировщик','Многодневная история','История + контакты + граф']
plt.rcParams.update({'svg.fonttype':'path','svg.hashsalt':'pt-model-figures','axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.spines.bottom':False,'axes.facecolor':'none','figure.facecolor':'none','axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,'text.color':INK})
text=SOURCE.read_text();records=[]
def number(cell):return float(re.search(r'\d+(?:\.\d+)?',cell).group())
for key,name in zip(IDS,NAMES):
 line=next(l for l in text.splitlines() if l.startswith('| ') and f'[`{key}`]' in l)
 row=[x.strip() for x in line.strip('|').split('|')]
 score=row[3].replace('**','');ci=re.search(r'\[([\d.]+)–([\d.]+)\]',score)
 dayline=next(l for l in text.splitlines() if l.startswith(f'| `{key}` |'))
 days=[number(x) for x in dayline.strip('|').split('|')[1:4]]
 records.append({'id':key,'name':name,'score':number(score),'interval':[float(ci[1]),float(ci[2])],'days':days,'dimensions':int(row[4]),'neighbors':number(row[5]),'links':number(row[6]),'partners':number(row[7]),'identity':number(row[9])})
evidence={'source':'scoreboard/README.md','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'models':records,'vector_parts':[896,64,64],'train_days':[2,12],'validation_days':[13,13],'test_days':[14,16]}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))

def txt(ax,x,y,s,size=12,bold=False,ha='center',color=INK,**kw):
 return ax.text(x,y,s,fontproperties=BOLD if bold else FONT,fontsize=size,ha=ha,va='center',color=color,**kw)
def canvas(w=9.2,h=5):
 fig,ax=plt.subplots(figsize=(w,h));ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off');fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.02);return fig,ax

def box(ax,x,y,w,h,title,sub='',accent=False,size=12):
 patch=FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.5,rounding_size=2',linewidth=1,edgecolor=ACCENT if accent else LINE,facecolor=SOFT if accent else 'none');ax.add_patch(patch)
 txt(ax,x,y+(3 if sub else 0),title,size,True,color=ACCENT if accent else INK)
 if sub:txt(ax,x,y-6,sub,max(10,size-2),color=MUTED)
 return patch

def arrow(ax,a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=INK,linewidth=1,connectionstyle='arc3,rad=0'))
def save(fig,name,mobile,title,description):
 suffix='-mobile' if mobile else ''
 fig.savefig(OUT/f'{name}{suffix}.svg',transparent=True,metadata={'Title':title,'Description':description,'Date':None})
 if not mobile:fig.savefig(OUT/f'{name}.png',dpi=140,transparent=True)
 plt.close(fig)

def architecture(mobile):
 fig,ax=canvas(3.2 if mobile else 10.4,7 if mobile else 5.1)
 if mobile:
  box(ax,50,91,84,13,'Сетевые записи','LANL: кто, с кем, когда',size=11)
  arrow(ax,(50,83),(50,76));ax.plot([18,82],[75,75],color=LINE,lw=1)
  for x,title,count in [(18,'Профиль','896'),(50,'Контакты','64'),(82,'Граф','64')]:
   arrow(ax,(x,75),(x,65));box(ax,x,52,26,23,title,count,size=10);arrow(ax,(x,39),(x,31))
  ax.plot([18,82],[30,30],color=LINE,lw=1);arrow(ax,(50,30),(50,23));box(ax,50,13,84,13,'1024 числа','нормируем и объединяем',accent=True,size=12)
 else:
  box(ax,50,87,65,17,'Сетевые записи LANL','кто, с кем, когда и сколько',size=13)
  arrow(ax,(50,77),(50,68));ax.plot([18,82],[67,67],color=LINE,lw=1)
  for x,title,sub in [(18,'Кодировщик + история','896 чисел'),(50,'Новые контакты','64 числа'),(82,'Структура сети','64 числа')]:
   arrow(ax,(x,67),(x,58));box(ax,x,44,28,23,title,sub,size=11);arrow(ax,(x,31),(x,25))
  ax.plot([18,82],[23,23],color=LINE,lw=1);arrow(ax,(50,23),(50,17));box(ax,50,8,65,13,'Итоговый профиль · 1024 числа',accent=True,size=13)
 save(fig,'architecture',mobile,'Как собирается multiday_codebook_plus','896 чисел многодневного профиля дополняются 64 числами новых контактов и 64 числами графа. Схема сборки, не точная топология нейросети.')

def features(mobile):
 fig,ax=canvas(3.2 if mobile else 7.6,6.3 if mobile else 4.1)
 fields=['src, dst','port, proto','bytes, packets','ts, duration'];names=['С кем общается','Какие сервисы','Сколько передаёт','Когда активен']
 for i,(field,label) in enumerate(zip(fields,names)):
  y=87-i*22;box(ax,24,y,43,16,field,size=10 if mobile else 12);arrow(ax,(47,y),(53,y));txt(ax,77,y,label,10 if mobile else 12)
 save(fig,'features',mobile,'От полей записи к признакам','Названия сетевых полей из описания эксперимента. Это схема признаков, не пример реального трафика.')

def composition(mobile):
 fig,ax=canvas(3.2 if mobile else 7.6,4 if mobile else 2.7)
 left=3;ys=69
 for value,color in zip([896,64,64],[INK,MUTED,ACCENT]):
  width=value/1024*94;ax.add_patch(plt.Rectangle((left,ys),width,16,color=color));left+=width
 txt(ax,44,77,'896',17,bold=True,color=SOFT)
 for y,label in [(49,'896 · история устройства'),(32,'64 · новые контакты'),(15,'64 · структура сети')]:txt(ax,5,y,label,11 if mobile else 13,ha='left')
 save(fig,'composition',mobile,'896 + 64 + 64 = 1024','Пропорции показывают число координат. Они не показывают веса блоков при сравнении.')

def timeline(mobile):
 fig,ax=canvas(3.2 if mobile else 7.6,4.8 if mobile else 2.9)
 if mobile:
  for y,label,days,color in [(85,'Обучение',[*range(2,13)],INK),(51,'Настройки',[13],MUTED),(18,'Проверка',[14,15,16],ACCENT)]:
   txt(ax,3,y,label,11,True,ha='left',color=color)
   for j,day in enumerate(days):txt(ax,5+j*8.4,y-14,f'{day:02}',10,color=color)
 else:
  for i,day in enumerate(range(2,17)):
   x=3+i*6.5;color=INK if day<=12 else MUTED if day==13 else ACCENT
   ax.plot([x,x+5],[50,50],color=color,lw=8,solid_capstyle='butt');txt(ax,x+2.5,33,f'{day:02}',11)
  txt(ax,37,77,'Обучение · 02–12',13,True);txt(ax,76,77,'13',13,True);txt(ax,90,77,'Тест · 14–16',12,True,color=ACCENT)
 save(fig,'timeline',mobile,'Обучение, настройки и будущие тестовые дни','Дни 02–12 — обучение; 13 — выбор настроек; 14–16 — тест. Тестовые окна не участвуют в обучении.')

def adapter(mobile):
 fig,ax=canvas(3.2 if mobile else 7.6,6.7 if mobile else 4.4)
 if mobile:
  box(ax,50,90,85,12,'Свежее окно','кодировщик → 896',size=11);arrow(ax,(50,82),(50,75));box(ax,50,68,85,11,'Ridge → 896',size=12)
  box(ax,50,46,85,14,'20 ближайших профилей','среднее блоков 64 + 64',size=10);arrow(ax,(50,37),(50,28));arrow(ax,(7,66),(7,19));arrow(ax,(7,19),(15,19));box(ax,55,17,75,13,'1024 числа',accent=True,size=13)
 else:
  box(ax,19,80,33,18,'Свежее окно','кодировщик → 896',size=12);arrow(ax,(37,80),(43,80));box(ax,63,80,35,18,'Ridge → 896',size=13)
  box(ax,26,40,47,22,'20 ближайших профилей','среднее дополнительных блоков',size=11);arrow(ax,(51,40),(61,40));box(ax,80,40,30,18,'64 + 64',size=13)
  arrow(ax,(63,69),(63,17));arrow(ax,(80,30),(80,17));box(ax,71,8,48,14,'1024 числа',accent=True,size=13)
 save(fig,'adapter',mobile,'Как кодируется позднее окно','Ridge преобразует основную 896-мерную часть. Новые блоки — взвешенное среднее блоков 20 ближайших обучающих профилей; температура softmax 0,02.')

def chart_font(ax,size=11):
 for item in [*ax.get_xticklabels(),*ax.get_yticklabels()]:item.set_fontproperties(FONT);item.set_fontsize(size)
 ax.tick_params(length=0,pad=8)
 ax.grid(axis='x',color=LINE,alpha=.3,lw=.7);ax.set_axisbelow(True)

def scores(mobile):
 fig,ax=plt.subplots(figsize=(3.2 if mobile else 7.6,4.6 if mobile else 3.6));fig.subplots_adjust(left=.08 if mobile else .35,right=.9,top=.96,bottom=.14)
 ax.set_xlim(0,100);ax.set_ylim(-.7,2.6);ax.set_yticks([]);ax.set_xticks([0,25,50,75,100]);chart_font(ax,10 if mobile else 11)
 for i,(r,c) in enumerate(zip(records,COLORS)):
  y=2-i;ax.barh(y,r['score'],height=.2 if mobile else .32,color=c)
  low,high=r['interval'];ax.errorbar(r['score'],y,xerr=[[r['score']-low],[high-r['score']]],fmt='none',ecolor=INK,capsize=4,lw=1.1)
  label=r['name'] if not mobile else ['Кодировщик','История','История + контакты + граф'][i]
  if mobile:ax.text(0,y+.25,label,fontproperties=FONT,fontsize=10,ha='left',color=INK)
  else:ax.text(-3,y,label,fontproperties=FONT,fontsize=11,ha='right',va='center')
  ax.text(101,y,f"{r['score']:.1f}".replace('.',','),fontproperties=BOLD,fontsize=11,va='center',ha='left',clip_on=False)
 save(fig,'score-comparison',mobile,'Общий балл и 95% интервалы','Шкала 0–100. Источник: scoreboard/README.md. Временной кодировщик 56,5; история 64,9; plus 70,0. Усики — указанные в таблице 95% интервалы.')

def days(mobile):
 fig,ax=plt.subplots(figsize=(3.2 if mobile else 7.6,4.8 if mobile else 4.2));fig.subplots_adjust(left=.16 if mobile else .09,right=.95,top=.92,bottom=.37 if mobile else .23)
 for r,c in zip(records,COLORS):ax.plot([14,15,16],r['days'],color=c,marker='o',lw=2,markersize=4,label=r['name'])
 ax.set(xlim=(13.85,16.15),ylim=(50,75),xticks=[14,15,16],yticks=[50,55,60,65,70,75]);chart_font(ax,10 if mobile else 11);ax.grid(axis='y',color=LINE,alpha=.3,lw=.7)
 handles,_=ax.get_legend_handles_labels();fig.legend(handles,['Кодировщик','История','История + контакты + граф'],loc='lower left',bbox_to_anchor=(.07,.015),frameon=False,ncol=1 if mobile else 3,prop=FontProperties(fname=FONT.get_file(),size=9 if mobile else 10))
 ax.set_xlabel('Тестовый день',fontproperties=FONT,fontsize=10,labelpad=12)
 save(fig,'test-days',mobile,'Результаты на днях 14–16','Каждая точка — одно тестовое окно. Линии соединяют три наблюдения, это не кривая обучения. Вертикальная шкала начинается с 50 баллов.')

def tasks(mobile):
 fig,axes=plt.subplots(4 if mobile else 2,1 if mobile else 2,figsize=(3.2 if mobile else 7.6,9 if mobile else 5.6));fig.subplots_adjust(left=.13 if mobile else .12,right=.81 if mobile else .90,top=.92 if mobile else .87,bottom=.1,hspace=.85 if mobile else .85,wspace=.4)
 settings=[('identity','Узнать устройство','Top-1',1),('links','Предсказать связь','ROC-AUC',1),('partners','Новые собеседники','MRR',.4),('neighbors','Похожее поведение','Neighbors@10',.35)]
 for ax,(key,label,metric,maximum) in zip(axes.flat,settings):
  ax.set_xlim(0,maximum);ax.set_ylim(-.6,2.8);ax.set_yticks([]);ax.set_xticks([0,maximum]);chart_font(ax,9)
  ax.set_title(label+'\n'+metric,fontproperties=BOLD,fontsize=11,pad=10,loc='left')
  for i,(r,c) in enumerate(zip(records,COLORS)):
   y=2-i;ax.barh(y,r[key],height=.4,color=c);ax.text(maximum*1.02,y,f"{r[key]:.3f}".replace('.',','),fontproperties=FONT,fontsize=10,va='center',clip_on=False)
 from matplotlib.lines import Line2D
 handles=[Line2D([0],[0],color=c,lw=4) for c in COLORS]
 fig.legend(handles,['Кодировщик','История','Plus'],loc='lower center',bbox_to_anchor=(.5,0),frameon=False,ncol=3,prop=FontProperties(fname=FONT.get_file(),size=9))
 save(fig,'task-comparison',mobile,'Что меняется в отдельных задачах','Сравнивать нужно модели внутри одной панели. Шкалы задач различаются. Plus улучшает общий балл, но Neighbors@10 уменьшается с 0,222 до 0,208.')

if __name__=='__main__':
 for mobile in [False,True]:
  for builder in [architecture,features,composition,timeline,adapter,scores,days,tasks]:builder(mobile)
 print('Generated 16 responsive SVG figures and 8 PNG previews from',SOURCE)
