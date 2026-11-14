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

def learning(mobile):
 fig,ax=canvas(3.2 if mobile else 9.2,7.2 if mobile else 4.2)
 if mobile:
  stages=[(50,89,'Профиль хоста','обучаемый вектор'),(50,65,'Прогноз контактов','кто окажется рядом завтра'),(50,41,'Реальные связи','из следующего дня истории'),(50,17,'Поправка профиля','уменьшаем ошибку')]
  for x,y,title,sub in stages:box(ax,x,y,86,17,title,sub,size=11)
  for y in [79,55,31]:arrow(ax,(50,y),(50,y-5))
 else:
  for x,y,title,sub in [(25,79,'Профиль хоста','обучаемый вектор'),(75,79,'Прогноз контактов','с кем будет общаться'),(75,24,'Реальные связи','следующий день истории'),(25,24,'Поправка профиля','уменьшаем ошибку')]:box(ax,x,y,42,26,title,sub,size=12)
  arrow(ax,(47,79),(53,79));arrow(ax,(75,64),(75,40));arrow(ax,(53,24),(47,24));arrow(ax,(25,40),(25,64))
 save(fig,'learning-loop',mobile,'Как профиль учится на истории','Концептуальная схема цели обучения. Будущие связи берутся только из обучающего периода; это не онлайн-обучение по тестовым дням.')

def similarity(mobile):
 fig,ax=canvas(3.2 if mobile else 9.2,7.2 if mobile else 4.3)
 if mobile:
  for y,title,sub in [(90,'Устройство A','сервисы · контакты · активность'),(68,'Устройство B','сервисы · контакты · активность')]:box(ax,50,y,86,15,title,sub,size=11)
  arrow(ax,(50,58),(50,51));box(ax,50,40,86,16,'Сравнение профилей','близость направлений векторов',size=10)
  arrow(ax,(50,30),(50,23));box(ax,50,12,86,14,'Список похожих устройств',accent=True,size=10)
 else:
  box(ax,24,83,42,22,'Устройство A','сервисы · контакты · активность',size=11);box(ax,76,83,42,22,'Устройство B','сервисы · контакты · активность',size=11)
  arrow(ax,(24,70),(43,55));arrow(ax,(76,70),(57,55));box(ax,50,43,66,22,'Сравнение профилей','близость направлений векторов',size=12)
  arrow(ax,(50,30),(50,23));box(ax,50,11,66,17,'Список похожих устройств',accent=True,size=12)
 save(fig,'similarity',mobile,'Что значит похожий профиль','Схема сравнения числовых профилей, без вымышленных устройств и измерений. Близость поведения сама по себе не доказывает одинаковую роль.')


def channels(mobile):
 fig,ax=canvas(3.2 if mobile else 9.2,8 if mobile else 5.4)
 labels=[('Поведение','интенсивность и направления'),('Счётчики','собеседники и сервисы'),('TF-IDF','характерные контакты'),('MLP-голова','обучаемое преобразование'),('MLP-голова','вторая обучаемая часть')]
 for j,(title,sub) in enumerate(labels):
  y=90-j*17;box(ax,38,y,66,12,title,sub,size=10 if mobile else 12);arrow(ax,(73,y),(84,y))
 ax.plot([86,86],[21,90],color=ACCENT,lw=2);txt(ax,93,55,'896',11 if mobile else 16,True,color=ACCENT,rotation=90)
 save(fig,'encoder-channels',mobile,'Состав базового кодировщика','Пять смысловых ветвей по описанию multiday_codebook. Размеры отдельных ветвей не опубликованы в карточке; их ширина на схеме условна.')

def decay(mobile):
 import numpy as np
 fig,ax=plt.subplots(figsize=(3.2 if mobile else 8.8,4.8));fig.subplots_adjust(left=.19 if mobile else .12,right=.94,top=.92,bottom=.19)
 x=np.linspace(0,12,121);ax.plot(x,2**(-x/3),color=ACCENT,lw=2.5)
 for day in [0,3,6,9,12]:ax.scatter([day],[2**(-day/3)],color=INK,s=24,zorder=3)
 ax.set(xlim=(-.3,12.3),ylim=(0,1.07),xticks=[0,3,6,9,12],yticks=[0,.25,.5,.75,1]);chart_font(ax,10);ax.grid(axis='y',color=LINE,alpha=.3,lw=.7)
 ax.set_xlabel('Возраст наблюдения, дней',fontproperties=FONT,fontsize=10,labelpad=12)
 ax.set_ylabel('Относительный вес',fontproperties=FONT,fontsize=10,labelpad=10)
 save(fig,'recency',mobile,'Свежие связи весят больше','Расчётная функция веса 2^(−возраст/3), полураспад 3 дня. Это параметр модели, не измеренная кривая качества.')

def network(mobile):
 import math
 fig,ax=canvas(3.2 if mobile else 9.2,6.8 if mobile else 5.8)
 points=[(22,78),(45,87),(70,77),(16,50),(49,55),(80,48),(28,22),(64,20)]
 edges=[(0,1),(0,3),(1,2),(1,4),(2,5),(3,4),(3,6),(4,5),(4,6),(4,7),(5,7),(6,7)]
 for a,b in edges:
  x,y=points[a];xx,yy=points[b];ax.plot([x,xx],[y,yy],color=ACCENT if 4 in (a,b) else LINE,lw=1.8 if 4 in (a,b) else 1,zorder=1)
 for j,(x,y) in enumerate(points):
  ax.scatter([x],[y],s=450 if mobile else 1000,color=ACCENT if j==4 else SOFT,edgecolor=ACCENT if j==4 else LINE,linewidth=1,zorder=2)
  txt(ax,x,y,chr(65+j),10 if mobile else 13,bold=True,color=SOFT if j==4 else INK,zorder=3)
 txt(ax,50,3,'Точка — устройство. Линия — связь.',9 if mobile else 12,color=MUTED)
 save(fig,'network-context',mobile,'Как устройство выглядит в графе связей','Условная сеть A–H для объяснения графового блока. Это не реальные хосты LANL и не результат кластеризации; выделены связи устройства E.')

def transfer(mobile):
 fig,ax=canvas(3.2 if mobile else 9.2,7.2 if mobile else 4.8)
 if mobile:
  for y,title,sub in [(88,'Исходное устройство','кого хотим понять'),(63,'Похожие устройства','находим по профилям'),(38,'Их собеседники','кандидаты на будущие связи'),(13,'Ранжирование','поднимаем полезных кандидатов')]:box(ax,50,y,86,17,title,sub,size=10)
  for y in [77,52,27]:arrow(ax,(50,y),(50,y-3))
 else:
  for x,y,title,sub in [(24,80,'Исходное устройство','кого хотим понять'),(76,80,'Похожие устройства','находим по профилям'),(76,23,'Их собеседники','кандидаты на будущие связи'),(24,23,'Ранжирование','полезные кандидаты выше')]:box(ax,x,y,42,25,title,sub,size=11)
  arrow(ax,(47,80),(53,80));arrow(ax,(76,66),(76,38));arrow(ax,(53,23),(47,23))
 save(fig,'partner-transfer',mobile,'Как соседи помогают искать новые контакты','Концептуальная схема переноса контактов. Число кандидатов и соединения не являются измерениями. Голова New partners обучается делать подобное соседство полезным.')

def ranking(mobile):
 rows=[]
 for line in text.splitlines():
  if not re.match(r'^\| \d+ \|',line):continue
  cells=[c.strip() for c in line.strip('|').split('|')];key=re.search(r'`([^`]+)`',cells[1]).group(1);rows.append((key,number(cells[3])))
 evidence['ranking']=[{'id':key,'score':score} for key,score in rows]
 fig,ax=plt.subplots(figsize=(3.2 if mobile else 10.4,11 if mobile else 9.2));fig.subplots_adjust(left=.08 if mobile else .37,right=.88,top=.98,bottom=.07)
 ax.set_xlim(0,100);ax.set_ylim(-.7,len(rows)-.15);ax.set_yticks([]);ax.set_xticks([0,25,50,75,100]);chart_font(ax,9 if mobile else 11)
 for i,(key,score) in enumerate(rows):
  y=len(rows)-1-i;color=ACCENT if i==0 else INK if i<3 else LINE;ax.barh(y,score,height=.19 if mobile else .5,color=color)
  label=key if len(key)<26 else key.replace('behavioral_dae_consistency','behavioral_dae + consistency')
  ax.text(0 if mobile else -3,y+.26 if mobile else y,label,fontproperties=FONT,fontsize=7.8 if mobile else 10,ha='left' if mobile else 'right',va='center')
  ax.text(101,y,str(score).replace('.',','),fontproperties=BOLD,fontsize=9 if mobile else 10,va='center',clip_on=False)
 save(fig,'all-methods',mobile,'Все 17 методов из таблицы проекта','Общие баллы переписаны программно из scoreboard/README.md. Сравнение сохраняет ограничения протокола: у многодневных моделей больше обучающей истории.')

def tradeoff(mobile):
 fig,ax=plt.subplots(figsize=(3.2 if mobile else 8.8,5.2));fig.subplots_adjust(left=.28 if mobile else .12,right=.95,top=.9,bottom=.2)
 for i,(r,c) in enumerate(zip(records,COLORS)):
  ax.scatter(r['identity'],r['neighbors'],s=85,color=c,zorder=3)
  label=['Кодировщик','История','Plus'][i]
  ax.annotate(label,(r['identity'],r['neighbors']),xytext=(-5 if i==1 else 6,10 if i!=2 else -18),textcoords='offset points',ha='right' if i==1 else 'left',fontproperties=FONT,fontsize=10,color=c)
 ax.set(xlim=(.3,.7),ylim=(.2,.23),xticks=[.3,.4,.5,.6,.7],yticks=[.2,.21,.22,.23]);chart_font(ax,9 if mobile else 11);ax.grid(axis='y',color=LINE,alpha=.3,lw=.7)
 ax.set_xlabel('Узнавание устройства · Top-1',fontproperties=FONT,fontsize=10,labelpad=13);ax.set_ylabel('Похожесть поведения · Neighbors@10',fontproperties=FONT,fontsize=10,labelpad=12)
 save(fig,'quality-tradeoff',mobile,'Общий рост не означает рост каждой задачи','Оси показывают две разные метрики из таблицы. Масштабы увеличены для сравнения; это не карта embedding-пространства.')

if __name__=='__main__':
 for mobile in [False,True]:
  for builder in [architecture,features,composition,timeline,adapter,scores,days,tasks,learning,similarity,channels,decay,network,transfer,ranking,tradeoff]:builder(mobile)
 (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
 print('Generated 32 responsive SVG figures and 16 PNG previews from',SOURCE)
