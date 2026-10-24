from __future__ import annotations
import copy
from root_teller.module3.structured.agents import OfflineReportInterpreter,DeterministicReportVerifier


class AuditedReportInterpreter(OfflineReportInterpreter):
    def generate(self,snapshot,state):
        report=super().generate(snapshot,state)
        syn=snapshot['synthesis'];rows=syn.get('model_proposal',{}).get('rationales',[])
        report['candidate_reasoning']=copy.deepcopy(rows)
        report['synthesis_execution_mode']=syn['execution_mode']
        report['selected_context_coverage']=copy.deepcopy(syn.get('model_view'))
        report['score_components']={r['entity_id']:copy.deepcopy(r.get('score_components')) for r in syn['ranked_candidates']}
        selected=next((r for r in rows if r['entity_id']==syn['selected_entity']),None)
        if selected:report['recommended_next_checks']=copy.deepcopy(selected['missing_information'])
        return report


class AuditedReportVerifier(DeterministicReportVerifier):
    def verify(self,report,snapshot,state):
        result=super().verify(report,snapshot,state)
        expected=AuditedReportInterpreter().generate(snapshot,state)
        for field in ['candidate_reasoning','synthesis_execution_mode','selected_context_coverage','score_components','recommended_next_checks']:
            if report.get(field)!=expected[field]:result['discrepancies'].append('CP4 report field mismatch: '+field)
        if result['discrepancies']:result['verdict']='REVISE'
        return result


def no_feedback_reference(initial_state,config,steward):
    if initial_state.version!=1 or initial_state.challenge_counts or initial_state.feedback_items or initial_state.validations:
        raise ValueError('No-feedback reference must use unmodified T0, not a post-feedback state')
    return steward.reason(copy.deepcopy(initial_state),copy.deepcopy(config))
