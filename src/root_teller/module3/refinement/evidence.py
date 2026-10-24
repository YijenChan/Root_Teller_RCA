from __future__ import annotations
import copy
from collections import Counter, defaultdict
from root_teller.module3.structured.agents import TypedEvidenceGate


FACT_KINDS = {'model_observation','observation','measured_temporal_order','observed_dependency'}
HYP_KINDS = {'root_role_global','root_role_local','propagation_explanation','proposed_hypothesis'}


def public_claim(c):
    return {k:copy.deepcopy(c[k]) for k in ['claim_id','kind','text','state','entity_ids','window_ids','evidence_refs']}


def facts_for(state, claim):
    return [state.raw['evidence'][eid]['facts'] for eid in claim['evidence_refs'] if eid in state.raw['evidence']]


def factual_claims(state):
    """Recheck stored claims rather than trusting their state string alone."""
    gate=TypedEvidenceGate()
    return {cid:c for cid,c in state.claims.items() if c['kind'] in FACT_KINDS and c['state'] in {'supported','contradicted'}
            and gate.evaluate(c,state.raw['evidence'],cid)['status'].lower()==c['state']}


def anomaly_indicator(state, claim):
    if claim['state']!='supported':return False
    for f in facts_for(state,claim):
        if f.get('diagnostic_role') in {'root','symptom','anomalous','anomaly'}:return True
        if f.get('log_contains_timeout') is True or f.get('queue_deviation') is True:return True
    return False


def candidate_link(entity, c, shown):
    if entity in c['entity_ids']:return 'direct_scope'
    # A neighbour's observation is context only, and only through a shown,
    # supported dependency. This does NOT make the neighbour the causal source.
    for d in shown.values():
        if d['kind']=='observed_dependency' and d['state']=='supported' and entity in d['entity_ids'] and set(d['entity_ids']) & set(c['entity_ids']) and set(d['window_ids']) & set(c['window_ids']):
            return 'observed_dependency_context'
    return None


def select_context(state, config, include_feedback=True, representation='rmg'):
    if representation not in {'rmg','flat'}:raise ValueError('Unsupported representation')
    state.assert_immutable(); all_facts=factual_claims(state)
    settings=config['premise_selection']; max_total=settings['maximum_factual_premises']; per=settings['maximum_per_entity']
    if max_total<1 or per<1 or settings['maximum_hypotheses_per_entity']<1:raise ValueError('Positive coverage budgets required')
    feedback = state.feedback_items[-settings['maximum_feedback_items']:] if include_feedback else []
    validations = state.validations[-settings['maximum_validations']:] if include_feedback else []
    selected={};reasons=defaultdict(list);proposed=[]
    def propose(cid,why):
        if cid in all_facts:
            proposed.append((cid,why))
    for row in feedback:
        propose(row['target_claim_id'],'feedback_target_fact')
    for row in validations:
        propose(row.get('claim_id'),'independently_checked_return')
    time=lambda c:min(float(state.raw['windows'][w]['start_time']) for w in c['window_ids'])
    summaries=[]
    # Round-robin per-entity coverage prevents the first candidate consuming cap.
    queues=[]
    for entity in state.base_scores:
        observations=sorted([c for c in all_facts.values() if c['kind'] in {'model_observation','observation'} and c['entity_ids']==[entity]],key=lambda c:(time(c),c['claim_id']))
        choices=[]
        if observations:
            anomaly=[c for c in observations if anomaly_indicator(state,c)]
            if anomaly:
                peak=max(anomaly,key=lambda c:(max((f.get('anomaly_score',0) for f in facts_for(state,c)),default=0),-time(c)))
                choices.append((peak['claim_id'],'peak_observed_model_anomaly_not_causal_truth'))
            choices.extend([(observations[0]['claim_id'],'earliest_observed_reference'),(observations[-1]['claim_id'],'latest_observed_reference')])
            for left,right in zip(observations,observations[1:]):
                lf=facts_for(state,left);rf=facts_for(state,right)
                if lf and rf and lf[0].get('diagnostic_role')!=rf[0].get('diagnostic_role'):
                    choices.extend([(left['claim_id'],'model_role_transition_before'),(right['claim_id'],'model_role_transition_after')]);break
        preferred={w for cid,_ in choices[:1] for w in all_facts[cid]['window_ids']}
        preferred.update(w for row in feedback if row.get('target_claim_id') in state.claims and entity in state.claims[row['target_claim_id']]['entity_ids']
                         and len(state.claims[row['target_claim_id']]['window_ids'])<len(state.raw['windows']) for w in state.claims[row['target_claim_id']]['window_ids'])
        relations=sorted([c for c in all_facts.values() if entity in c['entity_ids'] and c['kind'] in {'measured_temporal_order','observed_dependency'}],key=lambda c:(c['kind']!='measured_temporal_order',not bool(set(c['window_ids'])&preferred),time(c),c['claim_id']))
        choices.extend((c['claim_id'],'observed_relation_not_causal_edge') for c in relations[:2])
        seen=set(); unique=[]
        for cid,why in choices:
            if cid not in seen:unique.append((cid,why));seen.add(cid)
        queues.append(unique[:per])
        records=[f for c in observations for f in facts_for(state,c)]
        summaries.append({'entity_id':entity,'T0_score':state.base_scores[entity],
                          'model_role_counts':dict(Counter(f.get('diagnostic_role','non_model_observation') for f in records)),
                          'source_observation_count':len(observations)})
    for offset in range(max((len(q) for q in queues),default=0)):
        for q in queues:
            if offset<len(q):proposed.append(q[offset])
    for cid,why in proposed:
        if cid not in selected and len(selected)>=max_total:continue
        selected[cid]=all_facts[cid]
        if why not in reasons[cid]:reasons[cid].append(why)
    premises=[]
    for cid,c in selected.items():
        premises.append({**public_claim(c),'selection_reasons':reasons[cid],
                         'source_facts':copy.deepcopy(facts_for(state,c))})
    hypotheses={}
    targeted_hypotheses={r.get('target_claim_id') for r in feedback}
    targeted_hypotheses.update(r.get('claim_id') for r in validations)
    for entity in state.base_scores:
        candidates=sorted([c for c in state.claims.values() if c['kind'] in HYP_KINDS and entity in c['entity_ids'] and c['state'] in {'unresolved','pending'}],
                          key=lambda c:(c['claim_id'] not in targeted_hypotheses,c['kind']!='root_role_global',c['claim_id']))
        for c in candidates[:settings['maximum_hypotheses_per_entity']]:hypotheses[c['claim_id']]=public_claim(c)
    payload={'incident_id':state.raw['incident_id'],'snapshot_version':state.version,
             'candidate_roster':list(state.base_scores),'entity_summaries':summaries,'factual_premises':premises,
             'hypotheses':list(hypotheses.values()),'window_scope':copy.deepcopy(state.raw['windows']),
             'feedback_items':copy.deepcopy(feedback),'validation_results':copy.deepcopy(validations),
             'claim_state_counts':dict(Counter(c['kind']+':'+c['state'] for c in state.claims.values())),
             'coverage':{'full_claim_count':len(state.claims),'verified_factual_count':len(all_facts),'shown_factual_count':len(selected),
                         'omitted_factual_ids':sorted(set(all_facts)-set(selected)),
                         'candidates_without_selected_factual_scope':[e for e in state.base_scores if not any(e in c['entity_ids'] for c in selected.values())],
                         'omitted_hypothesis_count':sum(c['kind'] in HYP_KINDS for c in state.claims.values())-len(hypotheses),
                         'all_retained_windows':sorted(state.raw['windows']), 'selection_uses_root_labels':False,
                         'new_independent_evidence':False},
             'representation':representation,'warning':'Model outputs and dependencies are not proof of unique physical root cause.'}
    if representation=='rmg':
        payload['graph_view']={'adjacency':{e:sorted({v for d in selected.values() if d['kind']=='observed_dependency' and e in d['entity_ids'] for v in d['entity_ids'] if v!=e}) for e in state.base_scores},
                               'window_claim_ids':{w:[cid for cid,c in selected.items() if w in c['window_ids']] for w in state.raw['windows']}}
    # flat contains exactly the same rows/endpoints/window information; only the
    # redundant relationship indices differ. No intentional information removal.
    return payload,selected
