from __future__ import annotations
import copy
import math


def fuse(ranked_candidates, model_order, config):
    alpha = config['model_rank_weight']
    if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('model_rank_weight must be finite in [0,1]')
    entities = [r['entity_id'] for r in ranked_candidates]
    if len(model_order) != len(set(model_order)) or set(model_order) != set(entities):
        raise ValueError('Model order must retain candidate roster')
    positions = {e:i for i,e in enumerate(model_order)}
    rows = copy.deepcopy(ranked_candidates)
    for row in rows:
        signal = 1 - positions[row['entity_id']] / max(1,len(positions)-1)
        base = row['base_score']
        if not math.isfinite(base) or not 0 <= base <= 1:
            raise ValueError('Fusion requires T0 scores in [0,1]; do not silently normalize')
        q = (1-alpha)*base + alpha*signal
        prior = row['global_prior_factor']*row['local_prior_factor']
        row.update(score=q*prior, score_policy=config['policy_id'], evidence_score_proposal=q,
                   score_components={'T0':base,'model_rank_signal':signal,'model_rank_weight':alpha,
                                     'global_prior':row['global_prior_factor'],'local_prior':row['local_prior_factor'],
                                     'pre_feedback_fused_score':q,'feedback_prior_delta':q*prior-q,
                                     'not_probability':True})
    # Preserve input row order for exact ties; never use private labels.
    initial = {e:i for i,e in enumerate(entities)}
    rows.sort(key=lambda r:(-r['score'],initial[r['entity_id']]))
    return rows
