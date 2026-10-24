"""Readable, reversible wire projection; no fact pruning or number rounding.

Local audit retains canonical IDs and omitted-ID lists. The model receives
compact row tables, exact fact values and shared narrative templates. Alias
maps never contain private evaluator labels and are restored before grounding.
"""
from __future__ import annotations
import copy
import hashlib
import json
import re


def compact_json(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False)


def semantic_projection(context):
    p=copy.deepcopy(context)
    coverage=p['coverage']
    omitted=coverage.pop('omitted_factual_ids',[])
    coverage['omitted_factual_count']=len(omitted)
    coverage['omitted_id_audit_sha256']=hashlib.sha256(compact_json(omitted).encode()).hexdigest()
    # Selection reasons stay in the local audit; factual content is unchanged.
    for row in p['factual_premises']:row.pop('selection_reasons',None)
    for row in p['validation_results']:
        metadata=row.pop('client_metadata',None)
        view=row.pop('model_view',None)
        if view:
            row['context_coverage_summary']={k:copy.deepcopy(view[k]) for k in ['shown_window_ids','omitted_evidence_count'] if k in view}
            row['context_coverage_summary']['shown_evidence_count']=len(view.get('shown_evidence_ids',[]))
        # Tokens/cache IDs/duplicated prompt views remain in the local audit,
        # not inputs that could influence a causal ranking or leak arm identity.
    # The incident ID is a routing identifier, never a cue about fault labels.
    p['incident_id']='INC-'+hashlib.sha256(str(p['incident_id']).encode()).hexdigest()[:16]
    return p


class WireCodec:
    def __init__(self,context):
        self.source=semantic_projection(context)
        entities=self.source['candidate_roster']
        claims=[r['claim_id'] for r in self.source['factual_premises']+self.source['hypotheses']]
        for row in self.source['feedback_items']:
            if row.get('target_claim_id'):claims.append(row['target_claim_id'])
        refs=[eid for r in self.source['factual_premises']+self.source['hypotheses'] for eid in r['evidence_refs']]
        for row in self.source['validation_results']:refs.extend(row.get('evidence_refs',[]))
        self.entities={e:f'E{i}' for i,e in enumerate(entities)}
        self.claims={c:f'C{i}' for i,c in enumerate(dict.fromkeys(claims))}
        self.refs={r:f'R{i}' for i,r in enumerate(dict.fromkeys(refs))}
        routing=[]
        def identifiers(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if k in {'feedback_id','task_id','item_id'} and isinstance(x,str):routing.append(x)
                    identifiers(x)
            elif isinstance(v,list):
                for x in v:identifiers(x)
        identifiers(self.source['feedback_items']);identifiers(self.source['validation_results'])
        self.routing={v:f'D{i}' for i,v in enumerate(dict.fromkeys(routing)) if v not in self.entities and v not in self.claims and v not in self.refs}
        self.alias={**self.entities,**self.claims,**self.refs,**self.routing}
        self.reverse={v:k for k,v in self.alias.items()}
        if len(self.reverse)!=len(self.alias):raise ValueError('Alias collision')
        self.templates=[];self.template_lookup={}
        keys={k for row in self.source['factual_premises'] for f in row['source_facts'] for k in self._keys(f) if k not in self.entities}
        self.fields={k:f'F{i}' for i,k in enumerate(sorted(keys))}
        self.field_reverse={v:k for k,v in self.fields.items()}
        self.kinds={k:f'K{i}' for i,k in enumerate(sorted({r['kind'] for r in self.source['factual_premises']+self.source['hypotheses']}))}
        self.states={s:f'V{i}' for i,s in enumerate(sorted({r['state'] for r in self.source['factual_premises']+self.source['hypotheses']}))}

    @staticmethod
    def _keys(value):
        if isinstance(value,dict):return set(value)|{k for v in value.values() for k in WireCodec._keys(v)}
        if isinstance(value,list):return {k for v in value for k in WireCodec._keys(v)}
        return set()

    def encode(self,value,field_keys=False):
        if isinstance(value,str):
            if value in self.alias:return self.alias[value]
            return '~literal:'+value if value in self.reverse or value.startswith('~literal:') else value
        if isinstance(value,list):return [self.encode(v,field_keys) for v in value]
        if isinstance(value,dict):return {(self.alias.get(k,self.fields.get(k,k)) if field_keys else self.alias.get(k,k)):self.encode(v,field_keys) for k,v in value.items()}
        return value

    def decode(self,value,field_keys=False):
        if isinstance(value,str):return value[len('~literal:'):] if value.startswith('~literal:') else self.reverse.get(value,value)
        if isinstance(value,list):return [self.decode(v,field_keys) for v in value]
        if isinstance(value,dict):return {(self.reverse.get(k,self.field_reverse.get(k,k)) if field_keys else self.reverse.get(k,k)):self.decode(v,field_keys) for k,v in value.items()}
        return value

    def encode_text(self,text):
        # Literal braces are escaped for exact reconstruction, not interpreted.
        escaped=text.replace('{','{{').replace('}','}}');args=[]
        needles='|'.join(re.escape(e) for e in sorted(self.entities,key=len,reverse=True))
        pattern=(needles+'|' if needles else '')+r'\bW\d+\b|(?<=diagnostic_role=)[A-Za-z_]+'
        def replace(m):
            args.append(self.encode(m.group()));return '{'+str(len(args)-1)+'}'
        template=re.sub(pattern,replace,escaped)
        if template not in self.template_lookup:
            self.template_lookup[template]=len(self.templates);self.templates.append(template)
        return [self.template_lookup[template],args]

    def decode_text(self,row,templates=None):
        return (self.templates if templates is None else templates)[row[0]].format(*[self.decode(v) for v in row[1]])

    def pack(self):
        p=self.source;wire={k:self.encode(v) for k,v in p.items() if k not in {'factual_premises','hypotheses','window_scope','entity_summaries'}}
        fact_columns=['claim_id','kind','entity_ids','window_ids','state','text','evidence_refs','source_facts']
        hyp_columns=['claim_id','kind','entity_ids','window_ids','state','text','evidence_refs']
        def value(r,k):
            if k=='text':return self.encode_text(r[k])
            if k=='source_facts':return self.encode(r[k],True)
            if k=='kind':return self.kinds[r[k]]
            if k=='state':return self.states[r[k]]
            return self.encode(r[k])
        wire['facts']=[[value(r,k) for k in fact_columns] for r in p['factual_premises']]
        wire['hypotheses']=[[value(r,k) for k in hyp_columns] for r in p['hypotheses']]
        window_columns=[];window_rows=[]
        for w,r in p['window_scope'].items():
            cols=list(r)
            if cols not in window_columns:window_columns.append(cols)
            window_rows.append([w,window_columns.index(cols),list(r.values())])
        wire['windows']=window_rows
        wire['entity_summaries']=[[self.encode(r['entity_id']),r['T0_score'],r['model_role_counts'],r['source_observation_count']] for r in p['entity_summaries']]
        # Preserve every submitted field and finding, but avoid repeating the
        # field names and typed-check explanations for each investigation row.
        self.record_shapes=[]
        wire['feedback_items']=self.encode_records(p['feedback_items'])
        wire['validation_results']=self.encode_records(p['validation_results'])
        wire['legend']={'fact_columns':fact_columns,'hypothesis_columns':hyp_columns,
                        'entity_summary_columns':['entity_id','T0_score','model_role_counts','source_observation_count'],
                        'entities':{v:k for k,v in self.entities.items()},'fact_field_keys':self.field_reverse,
                        'text_templates':self.templates,'window_columns':window_columns,
                        'claim_kinds':{v:k for k,v in self.kinds.items()},'claim_states':{v:s for s,v in self.states.items()},
                        'record_shapes':self.record_shapes,
                        'instructions':'text=[template_index,args], with {0},{1} slots; E/C/R identify entities/claims/source references. Preserve model-vs-observation distinction. Output C/E aliases only in ID fields. No causal truth follows from these rows.'}
        return wire

    def encode_records(self,rows):
        text_fields={'reason','requested_check','operator_reason','explanation'}
        def value(k,v):
            if k in text_fields and isinstance(v,str):return {'text_template':self.encode_text(v)}
            if k=='limitations' and isinstance(v,list):return [{'text_template':self.encode_text(t)} for t in v]
            if isinstance(v,dict):return {'record':self.encode_records([v])[0]}
            return self.encode(v)
        result=[]
        for r in rows:
            keys=list(r)
            if keys not in self.record_shapes:self.record_shapes.append(keys)
            index=self.record_shapes.index(keys)
            result.append([index,[value(k,v) for k,v in r.items()]])
        return result

    def decode_records(self,rows,legend):
        def value(v):
            if isinstance(v,dict) and set(v)=={'text_template'}:return self.decode_text(v['text_template'],legend['text_templates'])
            if isinstance(v,dict) and set(v)=={'record'}:return self.decode_records([v['record']],legend)[0]
            if isinstance(v,list):return [value(x) for x in v]
            return self.decode(v)
        return [{k:value(v) for k,v in zip(legend['record_shapes'][index],values,strict=True)} for index,values in rows]

    def unpack(self,wire):
        p={k:self.decode(v) for k,v in wire.items() if k not in {'legend','facts','hypotheses','windows','entity_summaries'}}
        cols=wire['legend']['fact_columns']
        def value(k,v):
            if k=='text':return self.decode_text(v,wire['legend']['text_templates'])
            if k=='source_facts':return self.decode(v,True)
            if k=='kind':return wire['legend']['claim_kinds'][v]
            if k=='state':return wire['legend']['claim_states'][v]
            return self.decode(v)
        p['factual_premises']=[{k:value(k,v) for k,v in zip(cols,row,strict=True)} for row in wire['facts']]
        cols=wire['legend']['hypothesis_columns']
        p['hypotheses']=[{k:value(k,v) for k,v in zip(cols,row,strict=True)} for row in wire['hypotheses']]
        p['window_scope']={w:dict(zip(wire['legend']['window_columns'][index],values,strict=True)) for w,index,values in wire['windows']}
        cols=wire['legend']['entity_summary_columns']
        p['entity_summaries']=[{k:self.decode(v) for k,v in zip(cols,row,strict=True)} for row in wire['entity_summaries']]
        p['feedback_items']=self.decode_records(wire['feedback_items'],wire['legend'])
        p['validation_results']=self.decode_records(wire['validation_results'],wire['legend'])
        return p

    def response_schema(self,schema):
        result=copy.deepcopy(schema)
        def visit(node):
            if isinstance(node,dict):
                for k,v in node.items():
                    if k=='enum':node[k]=[self.alias.get(s,s) for s in v]
                    else:visit(v)
            elif isinstance(node,list):
                for v in node:visit(v)
        visit(result);return result

    def decode_response(self,response):
        # Do not replace free prose ("E1" may be a deliberate quote). Restore
        # IDs only, then let the canonical grounding checker validate all refs.
        out=copy.deepcopy(response)
        if not isinstance(out,dict) or not isinstance(out.get('rationales'),list) or not isinstance(out.get('ranked_entities'),list):raise ValueError('Invalid wire response')
        def entity(alias):
            if alias not in self.reverse or self.reverse[alias] not in self.entities:raise ValueError('Unknown entity alias')
            return self.reverse[alias]
        def claim(alias):
            if alias not in self.reverse or self.reverse[alias] not in self.claims:raise ValueError('Unknown claim alias')
            return self.reverse[alias]
        out['ranked_entities']=[entity(e) for e in out['ranked_entities']]
        for row in out['rationales']:
            if not isinstance(row,dict):raise ValueError('Invalid rationale row')
            row['entity_id']=entity(row.get('entity_id'))
            for field in ['supporting_claim_ids','counterevidence_claim_ids','context_claim_ids','unresolved_claim_ids']:
                if not isinstance(row.get(field),list):raise ValueError('Invalid reference list')
                row[field]=[claim(c) for c in row[field]]
        return out

    def encode_response(self,response):
        out=copy.deepcopy(response);out['ranked_entities']=[self.entities[e] for e in out['ranked_entities']]
        for row in out['rationales']:
            row['entity_id']=self.entities[row['entity_id']]
            for field in ['supporting_claim_ids','counterevidence_claim_ids','context_claim_ids','unresolved_claim_ids']:row[field]=[self.claims[c] for c in row[field]]
        return out
