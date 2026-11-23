"""Локальная таблица из полного пересчёта; development и benchmark раздельно."""
import argparse
import csv
import json
import math
from pathlib import Path

REQUIRED=['consistency@10_mean','link_learned_roc_auc','partner_new_mrr','pair_roc_auc','identity_top1']
NAMES={
    'similarity_graph_recent_control':'Similarity Graph Recent · контроль',
    'directed_bytes_gnn':'Directed Bytes GNN',
    'temporal_graph_mixer_v2':'Temporal Graph Mixer',
    'temporal_gatv2_dev_step13':'GATv2 · 13 шагов, контроль абляций',
    'temporal_gatv2_dev_step30':'GATv2 · 30 шагов',
    'temporal_gatv2_dev_no_graph':'GATv2 · без графа, 13 шагов',
    'temporal_gatv2_dev_no_bytes':'GATv2 · без байтов, 13 шагов',
    'temporal_gatv2_dev_no_vote':'GATv2 · без donor-loss, 13 шагов',
    'codebook_directed_gnn_zero':'Codebook · нулевая поправка',
    'codebook_directed_gnn_dev_identity':'Codebook Directed GNN · Identity',
    'codebook_directed_gnn_dev_no_graph':'Codebook · Identity без графа',
    'codebook_directed_gnn_dev_undirected_bytes':'Codebook · Identity без TX/RX-различия',
    'codebook_directed_gnn_dev_donors':'Codebook Directed GNN · donor-поправка',
    'directed_bytes_gnn_baseline':'Similarity Graph Recent · контроль',
    'directed_bytes_gnn_gnn_w050':'Directed Bytes GNN · вес 0,5',
    'directed_bytes_gnn_no_graph_w050':'Directed Bytes · без графа',
    'directed_bytes_gnn_no_bytes_w050':'Directed Bytes · без байтов',
}


def render(payload,output):
    rows=payload['records'];seen=set()
    for row in rows:
        key=(row['method'],row['window'])
        if key in seen:raise ValueError('Повтор метода/окна')
        seen.add(key)
        if not math.isfinite(row['score']) or any(
            row['metrics'].get(k) is None or not math.isfinite(row['metrics'][k]) for k in REQUIRED):
            raise ValueError('Отсутствует одна из пяти обязательных метрик')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with (output/'scoreboard.csv').open('w',newline='') as stream:
        fields=['method','protocol','window','dim','n_hosts','score',*REQUIRED]
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            writer.writerow({**{k:row[k]for k in ['method','window','dim','n_hosts','score']},
                'protocol':'development13' if row['window']=='development13' else 'benchmark14_16',
                **{k:row['metrics'][k]for k in REQUIRED}})
    text=['# Локальная таблица методов с 17 ноября','',
        'Актуальный пересчёт из сохранённых CSV с новым кэшем. Все пять метрик обязательны. '
        'Это исследовательская таблица, не публикация в общем рейтинге. Окна ранее изучались; закрытого holdout нет.',
        '', '## Benchmark дней 14–16','',
        '| Метод | D | День 14 | День 15 | День 16 | Средний score |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ['similarity_graph_recent_control','directed_bytes_gnn','temporal_graph_mixer_v2']:
        selected={r['window']:r for r in rows if r['method']==name and r['window']!='development13'}
        if set(selected)!={'test','test_day15','test_day16'}:raise ValueError('Для benchmark нужны три окна')
        values=[selected[k]['score']for k in ['test','test_day15','test_day16']]
        text.append(f"| {NAMES[name]} | {selected['test']['dim']} | "+
                    ' | '.join(f'{v:.4f}' for v in [*values,sum(values)/3])+' |')
    text+=['','Среднее вычислено из трёх оконных score. У Mixer выбран no_graph_global_w2 по development. '
           'История: Similarity Graph и Directed Bytes — по часу 02,10,11,12; Mixer — по часу 02–12.',
           '', '## Development дня 13','',
           'Пилоты GATv2 и Codebook Directed GNN на benchmark не оценивались. '
           'Строки GATv2 с 13 шагами нужны для сравнения абляций при одинаковом бюджете. '
           'У надстроек no_graph/no_bytes изменяют только новую ветку; основа фиксирована.',
           '', '| Метод | D | Score | Neighbors@10 | New MRR | Identity Top-1 |',
           '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['window']!='development13':continue
        m=row['metrics']
        text.append(f"| {NAMES[row['method']]} | {row['dim']} | {row['score']:.4f} | "
                    f"{m['consistency@10_mean']:.5f} | {m['partner_new_mrr']:.5f} | {m['identity_top1']:.5f} |")
    text+=['',
        'Все пять сырых метрик и оконные результаты: [CSV](scoreboard.csv), [JSON с SHA входов](results.json).',
        '', '## Код и описания','',
        '- [Directed Bytes GNN](../../../experiments/directed_bytes_gnn/README.md)',
        '- [Temporal GATv2](../../../experiments/temporal_gatv2/README.md)',
        '- [Codebook Directed GNN](../../../experiments/codebook_directed_gnn/README.md)',
        '- [Temporal Graph Mixer](../../../experiments/temporal_graph_mixer/README.md)',
        '- [Математическое устройство, включая основы](../reports/methods_since_2026_11_17.md)',
        '', '## Проверка кэша','',
        'Учтён комментарий physcorgi к PR175: индексы нельзя переносить между словарями партнёров. '
        'Эпизоды заново строят отображение словаря; его упорядоченный SHA фиксируется при обучении. '
        'Сырые кэши Similarity Graph теперь учитывают SHA содержимого flows, порядок хостов и версию схемы; '
        'behavior-кэш также учитывает fitted-словарь. Кэш метрик включает source-значения, порядок host_id и '
        'хеши эталонов. Тесты проверяют смену словаря, refit, перестановку host_id и изменение файла при '
        'сохранённых размере и mtime. Приведённые результаты пересчитаны в новом пустом каталоге.',
        '',
        'В таблице используется штатное чтение экспортированных CSV. Веса, потоки и embedding CSV в Git не включены.',
        '', 'Повтор генерации без данных: python -m experiments.methods_nov17.scoreboard --output runs/nov17_table.',
        'Полный пересчёт из локальных векторов: команды в '
        '[experiments/methods_nov17/README.md](../../../experiments/methods_nov17/README.md).','']
    (output/'README.md').write_text('\n'.join(text))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[2]/'research/DrMogger/nov17/results.json')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    render(json.loads(a.results.read_text()),a.output)


if __name__=='__main__':main()
