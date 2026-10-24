from __future__ import annotations
import copy
import math
from root_teller.module3.structured.contracts import digest
from root_teller.module3.structured.coordinator import FeedbackCoordinator


class RankOnlyPolicy:
    """Track-A service-verdict policies, independent of oracle/ground truth."""
    def __init__(self, scores, policy, gamma=.9, budget=20, k=None):
        if policy not in {'STATIC_TOPK','REVISIT_INSPECTION','SOFT_DECAY'}:raise ValueError('Unknown policy')
        if not scores or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in scores.values()):raise ValueError('Invalid T0')
        if not 0<gamma<=1 or not isinstance(budget,int) or budget<1:raise ValueError('Invalid budget/decay')
        if k is not None and (not isinstance(k,int) or not 1<=k<=len(scores)):raise ValueError('Invalid k')
        if k is not None and policy!='STATIC_TOPK':raise ValueError('k only applies to static Top-k inspection')
        self.policy=policy;self.gamma=gamma;self.budget=budget;self.k=k or len(scores)
        self.base=copy.deepcopy(scores);self.roster=sorted(scores,key=lambda e:-scores[e]);self.counts={e:0 for e in scores}
        self.pointer=0;self.interactions=0;self.closed=False;self.exhausted=False;self.accepted=None
        self.receipts={};self.trace=[]
    def ranking(self):
        if self.policy=='SOFT_DECAY':
            order=sorted(self.roster,key=lambda e: -self.base[e]*self.gamma**self.counts[e])
        elif self.policy=='REVISIT_INSPECTION':order=self.roster[self.pointer:]+self.roster[:self.pointer]
        else:order=self.roster[self.pointer:self.k]+self.roster[:self.pointer]+self.roster[self.k:]
        return order
    def leader(self):return None if self.closed or self.exhausted or self.interactions>=self.budget else self.ranking()[0]
    def submit(self, feedback_id, inspected_entity, verdict):
        body={'entity':inspected_entity,'verdict':verdict};fingerprint=digest(body)
        if feedback_id in self.receipts:
            receipt=self.receipts[feedback_id]
            return {**copy.deepcopy(receipt['result']),'status':'IDEMPOTENT_REPLAY','new_interactions':0} if receipt['hash']==fingerprint else {'status':'ID_CONFLICT','new_interactions':0}
        if verdict not in {'challenge','final_accept'}:raise ValueError('Track A accepts challenge or separate final_accept only')
        if not isinstance(feedback_id,str) or not feedback_id or self.leader() is None or inspected_entity!=self.leader():raise ValueError('Feedback must bind the current inspected candidate')
        before=self.ranking();self.interactions+=1
        if verdict=='final_accept':self.closed=True;self.accepted=inspected_entity
        elif self.policy=='SOFT_DECAY':self.counts[inspected_entity]+=1
        elif self.policy=='REVISIT_INSPECTION':self.pointer=(self.pointer+1)%len(self.roster)
        else:
            self.pointer+=1;self.exhausted=self.pointer>=self.k
        row={'feedback_id':feedback_id,'inspected_entity':inspected_entity,'verdict':verdict,'before_ranking':before,
             'after_ranking':self.ranking(),'interactions':self.interactions,'new_interactions':1,
             'status':'OPERATOR_ACCEPTED' if self.closed else 'ROSTER_EXHAUSTED' if self.exhausted else 'BUDGET_EXHAUSTED' if self.interactions>=self.budget else 'CONTINUE',
             'accepted_entity':self.accepted,'acceptance_is_not_ground_truth':True,
             'scores':{e:self.base[e]*self.gamma**self.counts[e] for e in self.base},'network_requests':0}
        self.trace.append(copy.deepcopy(row));self.receipts[feedback_id]={'hash':fingerprint,'result':copy.deepcopy(row)}
        return row


class StewardOnlyCoordinator(FeedbackCoordinator):
    """Same binding/gate/feedback/state, no Window Investigator tasks."""
    def _plan(self, bindings, state, feedback_id):return []
