"""Private split/provenance audit; never part of an agent prompt."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path


def audit_fold_manifest(path):
    path=Path(path);rows=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows,list):raise ValueError('Fold manifest must be an array')
    required={'outer_fold','incident_id','role'}
    if any(not isinstance(r,dict) or not required<=set(r) or r['role'] not in {'train','validation','test'} for r in rows):raise ValueError('Invalid fold row')
    test={r['incident_id'] for r in rows if r['role']=='test'}
    val={r['incident_id'] for r in rows if r['role']=='validation'}
    result={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'entries':len(rows),
            'unique_incidents':len({r['incident_id'] for r in rows}),'validation_union':len(val),'test_union':len(test),
            'global_validation_test_union_overlap':len(val&test),'global_validation_never_test':len(val-test),'folds':[]}
    for fold in sorted({r['outer_fold'] for r in rows}):
        local=[r for r in rows if r['outer_fold']==fold]
        roles={role:{r['incident_id'] for r in local if r['role']==role} for role in ['train','validation','test']}
        overlap=sum(len(roles[a]&roles[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
        grouping='capture_id' if all('capture_id' in r for r in local) else 'family' if all('family' in r for r in local) else None
        group_overlap=None
        if grouping:
            groups={role:{json.dumps(r[grouping],sort_keys=True) for r in local if r['role']==role} for role in roles}
            group_overlap=sum(len(groups[a]&groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')])
        result['folds'].append({'outer_fold':fold,'role_counts':{k:len(v) for k,v in roles.items()},'incident_overlap':overlap,
                                'grouping_metadata':grouping,'group_overlap':group_overlap,
                                'capture_separation_verified':grouping=='capture_id' and group_overlap==0})
    return result


def validate_development_proof(proof):
    """Checks declared split/training provenance, not retraining its truth.

    Provider must supply hash-linked training logs/T0 packs. Passing logical
    declarations alone never auto-authorizes external compute.
    """
    required={'system','outer_fold','validation_case_ids','train_case_ids','test_case_ids',
              'perception_training_case_ids','perception_calibration_case_ids','t0_case_ids',
              'training_manifest_sha256','checkpoint_sha256','t0_bundle_sha256'}
    if not isinstance(proof,dict) or not required<=set(proof):raise ValueError('Incomplete provenance proof')
    id_fields=[f for f in required if f.endswith('_case_ids')]
    for f in id_fields:
        if not isinstance(proof[f],list) or any(not isinstance(i,str) or not i for i in proof[f]) or len(proof[f])!=len(set(proof[f])):raise ValueError('Invalid/duplicate case IDs')
    train=set(proof['train_case_ids']);val=set(proof['validation_case_ids']);test=set(proof['test_case_ids'])
    if not val or not proof['t0_case_ids'] or not train:raise ValueError('Nonempty development/train/T0 sets required')
    if train&val or train&test or val&test:raise ValueError('Within-fold split overlap')
    if not set(proof['t0_case_ids'])<=val:raise ValueError('T0 cases are not this fold validation material')
    if not set(proof['perception_training_case_ids'])<=train or not set(proof['perception_calibration_case_ids'])<=train:raise ValueError('Validation/test exposed to Perception training/calibration; refit checkpoint cannot create held-out development T0')
    for f in ['training_manifest_sha256','checkpoint_sha256','t0_bundle_sha256']:
        h=proof[f]
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Missing hash-linked provenance')
    return {'logical_split_checks':'passed','actual_training_reproduced':False,'external_compute_authorized':False,
            'requires_provider_artifact_hash_verification':True,'scope':'this dataset/fold only; not global shared-parameter tuning approval'}
