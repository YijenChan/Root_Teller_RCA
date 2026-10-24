"""Bounded relation context. Source relation states are not causal verdicts."""
import copy
from collections import Counter


def relation_context(state,maximum=12):
    records=state.raw.get('cross_window_relations',[])
    windows=state.raw['windows'];known=state.raw['evidence'];groups={}
    for r in records:
        a,b=r['source_window'],r['destination_window']
        if a not in windows or b not in windows:raise ValueError('Unknown relation window')
        refs=r.get('supporting_evidence_ids',[])
        if any(e not in known for e in refs):raise ValueError('Unknown relation evidence reference')
        gap=abs(windows[a].get('activation_order',0)-windows[b].get('activation_order',0))
        row={'relation_id':r['relation_id'],'type':r['type'],'source_window':a,'destination_window':b,
             'window_gap':gap,'evidence_refs':list(refs),'source_state':r['state'],'causal_support_eligible':False}
        if 'shared_directed_edges' in r:row['shared_directed_edges']=copy.deepcopy(r['shared_directed_edges'])
        if 'entity_id' in r:row['entity_id']=r['entity_id']
        priority=(0 if gap>1 else 1,-len(r.get('retrieval_witness',{}).get('recurrent_anomaly_entities',[])),
                  -gap,{'propagation-continuation':0,'entity-continuity':1,'semantic-association':2}.get(r['type'],3),r['relation_id'])
        groups.setdefault((a,b),[]).append((priority,row))
    ordered=sorted([sorted(v,key=lambda x:x[0]) for v in groups.values()],key=lambda v:v[0][0])
    selected=[]
    while ordered and len(selected)<maximum:
        remainder=[]
        for values in ordered:
            if len(selected)>=maximum:break
            selected.append(values[0][1])
            if len(values)>1:remainder.append(values[1:])
        ordered=remainder
    return {'relations':selected,'coverage':{'full_relation_count':len(records),'shown_relation_count':len(selected),
            'full_endpoint_pair_count':len(groups),'non_adjacent_record_count':sum(1 for r in records if abs(windows[r['source_window']].get('activation_order',0)-windows[r['destination_window']].get('activation_order',0))>1),
            'types':dict(Counter(r['type'] for r in records))},
            'warning':'Structural/model/semantic context only; no relation alone verifies a fault root or causal propagation.'}
