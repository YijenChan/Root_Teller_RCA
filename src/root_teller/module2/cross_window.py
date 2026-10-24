"""Sparse, provenance-linked associations across activated non-adjacent windows.

Rules use public model observations only. Links are context, not fault-causality
proof, and neither a GT label nor a rejected candidate is a retrieval criterion.
"""
from dataclasses import dataclass, asdict
from .contracts import stable_id


@dataclass(frozen=True)
class CrossWindowPolicy:
    enabled: bool = True
    max_neighbors: int = 3
    max_chain_pairs: int = 8
    anomaly_threshold: float = .5

    def __post_init__(self):
        if self.max_neighbors < 0 or self.max_chain_pairs < 0 or not 0 <= self.anomaly_threshold <= 1:
            raise ValueError("Invalid cross-window relation budget")

    def to_dict(self):
        return asdict(self)


def entities(pack):
    return {r['entity_id']:r for r in pack['entity_evidence']}


def chains(pack):
    unique={c['evidence_id']:c for r in pack['ranked_candidates'] for c in r['candidate_chains']}
    return list(unique.values())


def directed_edges(chain):
    return set(zip(chain['entities'],chain['entities'][1:]))


def pair_features(older,newer,policy):
    a,b=entities(older),entities(newer)
    shared=set(a)&set(b)
    def abnormal(r):
        return r.get('diagnostic_role')!='normal' and float(r.get('anomaly_score',0))>=policy.anomaly_threshold
    recurrent=sorted(e for e in shared if abnormal(a[e]) and abnormal(b[e]))
    left,right=chains(older),chains(newer)
    pairs=[]
    for l in left:
        for r in right:
            overlap=directed_edges(l)&directed_edges(r)
            if overlap:pairs.append((l,r,sorted(overlap)))
    pairs.sort(key=lambda p:(-len(p[2]),p[0]['evidence_id'],p[1]['evidence_id']))
    top_a={r['entity_id']:r for r in older['ranked_candidates'][:3]}
    top_b={r['entity_id']:r for r in newer['ranked_candidates'][:3]}
    common=sorted(set(top_a)&set(top_b))
    return {'shared':sorted(shared),'recurrent':recurrent,'chain_pairs':pairs,'top_a':top_a,'top_b':top_b,'common_top':common}


def select_neighbors(window_id,windows,policy):
    if not policy.enabled or policy.max_neighbors==0:return []
    older=windows[window_id];order=older['window']['activation_order'];matches=[]
    for wid,pack in windows.items():
        gap=order-pack['window']['activation_order']
        if gap<=1:continue
        features=pair_features(older,pack,policy)
        if not (features['shared'] or features['chain_pairs'] or len(features['common_top'])>=2):continue
        key=(-len(features['recurrent']),-len(features['chain_pairs']),-len(features['common_top']),gap,wid)
        matches.append((key,wid,features))
    matches.sort(key=lambda row:row[0])
    return [(wid,features) for _,wid,features in matches[:policy.max_neighbors]]


def build_relations(incident_id,window_id,windows,hashes,policy):
    result=[];older=windows[window_id];a=entities(older)
    for wid,features in select_neighbors(window_id,windows,policy):
        newer=windows[wid];b=entities(newer)
        gap=older['window']['activation_order']-newer['window']['activation_order']
        common={'source_window':window_id,'destination_window':wid,'window_gap':gap,
                'link_scope':'non-adjacent','policy_version':'sparse-cross-window-1',
                'source_pack_hashes':{window_id:hashes[window_id],wid:hashes[wid]},
                'causal_support_eligible':False,'final_support_eligible':False,
                'retrieval_witness':{'recurrent_anomaly_entities':features['recurrent'],
                                    'shared_directed_chain_pairs':len(features['chain_pairs']),
                                    'shared_top_candidate_entities':features['common_top']}}
        for entity in features['shared']:
            result.append({**common,'relation_id':stable_id('REL',incident_id,'entity-continuity',window_id,wid,entity),
                           'type':'entity-continuity','entity_id':entity,'state':'validated',
                           'validation_scope':'canonical identity only, not same fault cause',
                           'supporting_evidence_ids':[a[entity]['evidence_id'],b[entity]['evidence_id']]})
        for l,r,overlap in features['chain_pairs'][:policy.max_chain_pairs]:
            result.append({**common,'relation_id':stable_id('REL',incident_id,'propagation-continuation',l['evidence_id'],r['evidence_id']),
                           'type':'propagation-continuation','state':'hypothesized',
                           'shared_entities':sorted({e for pair in overlap for e in pair}),
                           'shared_directed_edges':[list(e) for e in overlap],
                           'supporting_evidence_ids':[l['evidence_id'],r['evidence_id']],
                           'validation_scope':'ordered model-chain overlap; physical propagation unverified'})
        if len(features['common_top'])>=2:
            result.append({**common,'relation_id':stable_id('REL',incident_id,'semantic-association',window_id,wid,*features['common_top']),
                           'type':'semantic-association','state':'hypothesized','strength':'weak',
                           'shared_candidate_entities':features['common_top'],
                           'supporting_evidence_ids':[features[key][e]['evidence_id'] for e in features['common_top'] for key in ('top_a','top_b')],
                           'validation_scope':'candidate-overlap proxy, not semantic entailment or causal proof'})
    return result


def select_prompt_relations(relations,maximum=20):
    """Cover endpoint pairs, prioritizing recurrence and non-adjacent context."""
    groups={}
    for r in relations:
        gap=r.get('window_gap',abs(int(r['source_window'][1:])-int(r['destination_window'][1:])))
        key=(0 if gap>1 else 1,-len(r.get('retrieval_witness',{}).get('recurrent_anomaly_entities',[])),
             -gap,{'propagation-continuation':0,'entity-continuity':1,'semantic-association':2}.get(r['type'],3),r['relation_id'])
        groups.setdefault((r['source_window'],r['destination_window']),[]).append((key,r))
    queue=sorted([sorted(v,key=lambda x:x[0]) for v in groups.values()],key=lambda v:v[0][0])
    selected=[]
    while queue and len(selected)<maximum:
        next_queue=[]
        for values in queue:
            if len(selected)>=maximum:break
            r=values[0][1]
            selected.append({k:v for k,v in r.items() if k not in {'source_pack_hashes','retrieval_witness'}})
            if len(values)>1:next_queue.append(values[1:])
        queue=next_queue
    return selected
