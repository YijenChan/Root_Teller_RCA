from __future__ import annotations
import copy
from root_teller.module3.structured.agents import JSONClientInvestigator


class AuditedWindowInvestigator(JSONClientInvestigator):
    """Same task contract, bounded shown evidence, same CP4 ledger as Steward."""
    def __init__(self,client,max_records=16,max_context_claims=6):
        super().__init__(client)
        self.requires_external_execution=getattr(client,'requires_external_execution',True)
        self.max_records=max_records;self.max_context_claims=max_context_claims
    def inspect(self,task):
        if self.max_records<1 or self.max_context_claims<1:raise ValueError('Positive context budgets required')
        claim=task['claim'];priority=set(claim.get('predicate',{}).get('evidence_refs',[])) if claim else set()
        anchors=set(claim['entity_ids']) if claim else set()
        order=sorted(task['evidence'],key=lambda eid:(eid not in priority,not bool(set(task['evidence'][eid]['entity_ids'])&anchors),eid))
        compact=copy.deepcopy(task);compact['evidence']={eid:copy.deepcopy(task['evidence'][eid]) for eid in order[:self.max_records]}
        compact['context_claims']=[c for c in compact['context_claims'] if c['state'] in {'supported','contradicted'}][:self.max_context_claims]
        result=super().inspect(compact)
        if not set(result['evidence_refs'])<=set(compact['evidence']):raise ValueError('Investigator cited unshown evidence')
        result['model_view']={'shown_evidence_ids':list(compact['evidence']),'omitted_evidence_count':len(task['evidence'])-len(compact['evidence']),
                              'shown_window_ids':list(compact['window_ids']),'not_full_history':True}
        return result
