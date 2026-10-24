from __future__ import annotations
import copy
from .evidence import select_context, candidate_link, anomaly_indicator
from .scoring import fuse
from root_teller.module3.structured.agents import OfflineEvidenceSteward
from root_teller.module3.structured.relations import relation_context


def proposal_schema(roster, facts, hypotheses):
    def ids(values, maximum):
        # Empty arrays remain legal; no reference is better than a fabricated one.
        return {'type':'array','maxItems':maximum,'items':{'type':'string', **({'enum':list(values)} if values else {})}}
    fields={'entity_id':{'type':'string','enum':roster},'supporting_claim_ids':ids(facts,2),
            'counterevidence_claim_ids':ids(facts,2),'context_claim_ids':ids(facts,2),
            'unresolved_claim_ids':ids(hypotheses,4),'missing_information':{'type':'array','minItems':1,'maxItems':2,'items':{'type':'string'}},
            'ranking_basis':{'type':'string','enum':['observational_support','insufficient_discriminating_evidence']},
            'explanation':{'type':'string'}}
    return {'type':'object','additionalProperties':False,'required':['ranked_entities','rationales'],
            'properties':{'ranked_entities':{'type':'array','minItems':len(roster),'maxItems':len(roster),'items':{'type':'string','enum':roster}},
                          'rationales':{'type':'array','minItems':len(roster),'maxItems':len(roster),'items':{'type':'object','additionalProperties':False,'required':list(fields),'properties':fields}}}}


def validate_proposal(value, payload, shown, state):
    if not isinstance(value,dict) or set(value)!={'ranked_entities','rationales'}:raise ValueError('Unexpected proposal fields')
    roster=payload['candidate_roster']; order=value['ranked_entities'];rows=value['rationales']
    if not isinstance(order,list) or any(not isinstance(e,str) for e in order) or len(order)!=len(set(order)) or set(order)!=set(roster):raise ValueError('Candidate roster mismatch')
    if not isinstance(rows,list) or any(not isinstance(r,dict) for r in rows) or len(rows)!=len(roster):raise ValueError('One rationale per candidate required')
    if any(not isinstance(r.get('entity_id'),str) for r in rows) or len({r['entity_id'] for r in rows})!=len(roster) or {r['entity_id'] for r in rows}!=set(roster):raise ValueError('Rationale binding mismatch')
    required=set(proposal_schema(roster,shown,[])['properties']['rationales']['items']['required'])
    hyp={r['claim_id']:r for r in payload['hypotheses']}
    for row in rows:
        if set(row)!=required:raise ValueError('Unexpected rationale fields; no factual/causal state rewrites allowed')
        entity=row['entity_id']
        used=set()
        for field in ['supporting_claim_ids','counterevidence_claim_ids','context_claim_ids','unresolved_claim_ids']:
            ids=row[field]; limit=4 if field=='unresolved_claim_ids' else 2
            if not isinstance(ids,list) or any(not isinstance(i,str) for i in ids) or len(ids)>limit or len(ids)!=len(set(ids)):raise ValueError('Invalid reference array')
            if used & set(ids):raise ValueError('Same premise cannot serve conflicting evidence roles')
            used.update(ids)
            for cid in ids:
                if field=='unresolved_claim_ids':
                    if cid not in hyp or entity not in hyp[cid]['entity_ids']:raise ValueError('Unresolved hypothesis outside candidate scope')
                    continue
                if cid not in shown:raise ValueError('Premise not shown or not independently checked')
                c=shown[cid];link=candidate_link(entity,c,shown)
                if link is None:raise ValueError('Premise unrelated to candidate')
                if field=='supporting_claim_ids':
                    if link!='direct_scope' or not anomaly_indicator(state,c):
                        raise ValueError('Normal record/dependency/neighbour context cannot be positive root support')
                elif field=='counterevidence_claim_ids':
                    # Counterindicator is narrower than causal disproof: normal
                    # model output or a contradicted typed claim in direct scope.
                    normal=any(f.get('diagnostic_role')=='normal' for eid in c['evidence_refs'] for f in [state.raw['evidence'][eid]['facts']])
                    if link!='direct_scope' or not (c['state']=='contradicted' or normal):raise ValueError('Unsupported counterindicator')
        gaps=row['missing_information']
        if not isinstance(gaps,list) or not 1<=len(gaps)<=2 or any(not isinstance(g,str) or not g.strip() or len(g)>240 for g in gaps):raise ValueError('Missing information must remain explicit')
        if row['ranking_basis'] not in {'observational_support','insufficient_discriminating_evidence'}:raise ValueError('Invalid ranking basis')
        if row['ranking_basis']=='observational_support' and not row['supporting_claim_ids']:raise ValueError('No anomaly premise for claimed observational support')
        if not isinstance(row['explanation'],str) or not row['explanation'].strip() or len(row['explanation'])>400:raise ValueError('Invalid bounded explanation')
    return copy.deepcopy(value)


class OfflineContractClient:
    """Deterministic fixture producer, never an LLM or a cache-based replicate.

    Keeps T0 model order deliberately. Used to exercise contracts/transactions,
    not to fabricate a successful model investigation or simulate model quality.
    """
    requires_external_execution=False
    calls=0
    def complete(self, *, payload, validator, **kwargs):
        self.calls+=1
        rows=[]
        for entity in payload['candidate_roster']:
            facts=[c for c in payload['factual_premises'] if entity in c['entity_ids']]
            supporting=[c['claim_id'] for c in facts if c['state']=='supported' and any(f.get('diagnostic_role') in {'root','symptom','anomalous','anomaly'} or f.get('log_contains_timeout') is True or f.get('queue_deviation') is True for f in c['source_facts'])][:1]
            counters=[c['claim_id'] for c in facts if c['state']=='contradicted' or any(f.get('diagnostic_role')=='normal' for f in c['source_facts'])][:1]
            context=[c['claim_id'] for c in facts if c['claim_id'] not in supporting+counters][:1]
            unresolved=[c['claim_id'] for c in payload['hypotheses'] if entity in c['entity_ids']][:2]
            rows.append({'entity_id':entity,'supporting_claim_ids':supporting,'counterevidence_claim_ids':counters,
                         'context_claim_ids':context,'unresolved_claim_ids':unresolved,
                         'missing_information':['Independent causal validation is unavailable.'],
                         'ranking_basis':'insufficient_discriminating_evidence',
                         'explanation':'Engineering fixture retains T0 order; no LLM inference performed.'})
        by_entity={r['entity_id']:r['T0_score'] for r in payload['entity_summaries']}
        order=sorted(payload['candidate_roster'],key=lambda e:-by_entity[e])
        return validator({'ranked_entities':order,'rationales':rows}),{'network_request':False,'cached':False,'execution_mode':'offline_contract_fixture'}


class AuditedSteward:
    def __init__(self, client, revision_config, representation='rmg', include_feedback=True):
        self.client=client;self.revision_config=copy.deepcopy(revision_config)
        self.representation=representation;self.include_feedback=include_feedback
        self.requires_external_execution=getattr(client,'requires_external_execution',True)

    def reason(self,state,core_config):
        before=state.diagnostic_digest();state.assert_immutable()
        payload,shown=select_context(state,self.revision_config,self.include_feedback,self.representation)
        # Same relation information is supplied to both representations. Only
        # organization differs; flat must not be denied cross-window context.
        payload['cross_window_context']=relation_context(state,maximum=12)
        schema=proposal_schema(payload['candidate_roster'],shown,[c['claim_id'] for c in payload['hypotheses']])
        response,meta=self.client.complete(role='cp4_evidence_steward',prompt_version='cp4a-audited-contrast-v1',
            system_prompt=('Compare candidates across windows. Return ranked_entities and exactly one rationale per entity, with supporting_claim_ids, '
                           'counterevidence_claim_ids, context_claim_ids, unresolved_claim_ids, missing_information, ranking_basis, explanation. '
                           'Normal model records and dependencies cannot alone be positive root support. Distinguish anomaly observations, '
                           'counterindicators, neighbour context and unresolved causal hypotheses. Preserve gaps and every candidate. '
                           'Do not claim unique causality, create evidence or overwrite any claim state. Use JSON.'),
            payload=copy.deepcopy(payload),response_schema=schema,
            validator=lambda v:validate_proposal(v,payload,shown,state))
        if before!=state.diagnostic_digest():raise ValueError('Proposal client mutated derived state')
        state.assert_immutable()
        synthesis=OfflineEvidenceSteward().reason(state,core_config)
        synthesis['ranked_candidates']=fuse(synthesis['ranked_candidates'],response['ranked_entities'],self.revision_config['scoring'])
        synthesis.update(selected_entity=synthesis['ranked_candidates'][0]['entity_id'],model_proposal=response,
                         client_metadata=meta,model_view=payload['coverage'],selected_context=payload,
                         execution_mode='model_feedback_synthesis' if self.requires_external_execution else 'offline_contract_fixture_no_LLM',
                         limitations=['Model-generated rankings are diagnostic hypotheses, not independent causal confirmation.' if self.requires_external_execution else 'Offline contract execution only; no model inference or comparative efficacy result.',
                                      'Typed reference relevance checks are not complete natural-language entailment verification or causal proof.'])
        return synthesis
