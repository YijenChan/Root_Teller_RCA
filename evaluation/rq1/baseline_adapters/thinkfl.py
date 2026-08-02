"""ThinkFL tool-interface compatibility evaluation on RE2-OB.

The upstream ThinkFL artifact provides GRPO/SFT training code only; it does
not release inference weights or RE2 preprocessing. This runner uses the
user-authorized GPT-4o-mini endpoint as a frozen lightweight actor. It gives
the actor the two published diagnostic tool views (trace and metric evidence),
then asks it to self-check and return an ordered service list in one response.
"""
from __future__ import annotations

import os

import argparse, json, re
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import requests

PROJECT=Path(os.environ.get("ROOTTELLER_WORKSPACE", Path.cwd())).expanduser().resolve()
RAW=PROJECT/"dataset"/"RCAEval RE"/"RE2"/"RE2-OB"/"RE2-OB"
MANIFEST=Path(os.environ.get("ROOTTELLER_ACTIVE_SPLIT_MANIFEST", PROJECT/"evaluation"/"rq1"/"manifests"/"active_split_manifest.csv"))
SERVICES=("adservice","cartservice","checkoutservice","currencyservice","emailservice","frontend","paymentservice","productcatalogservice","recommendationservice","redis","shippingservice")

def service(v):
    v=str(v).lower().replace("_","-")
    return {"frontendservice":"frontend","frontend-external":"frontend","redis-cart":"redis"}.get(v,v)

@dataclass(frozen=True)
class Row:
    incident_id:str; root:str
    @property
    def d(self): return RAW/Path(self.incident_id)

def rows(split):
    x=pd.read_csv(MANIFEST);x=x[(x.dataset_system=="RCAEval RE2-OB")&(x.split==split)]
    if split=="test": x=x[x.eligible]
    return [Row(str(r.incident_id),service(r.root_cause_service)) for r in x.itertuples(index=False)]

def actor_config() -> tuple[str, str]:
    key = os.environ["ROOTTELLER_API_KEY"]
    base = os.environ.get("ROOTTELLER_API_BASE", "https://api.openai.com/v1")
    return key, base.rstrip("/")
def evidence(row):
    m=pd.read_csv(row.d/"simple_metrics.csv",low_memory=False); m["time"]=pd.to_numeric(m.time)
    start=float(m.time.min()); normal=m[m.time<start+60]; prod=m[m.time>=start+60]
    metric=[]
    for s in SERVICES:
        cols=[c for c in m.columns if c.startswith(s+"_")]
        best=[]
        for c in cols:
            a=pd.to_numeric(normal[c],errors="coerce").dropna();b=pd.to_numeric(prod[c],errors="coerce").dropna()
            if len(a)>3 and len(b)>3:
                z=abs(float(b.mean()-a.mean()))/max(float(a.std(ddof=0)),1e-6);best.append((z,c))
        if best:
            z,c=max(best);metric.append((s,round(z,2),c.rsplit("_",1)[-1]))
    metric=sorted(metric,key=lambda x:-x[1])[:8]
    tr=pd.read_csv(row.d/"tracets_err.csv",low_memory=False);tr["time"]=pd.to_numeric(tr.time);n=tr[tr.time<start+60];p=tr[tr.time>=start+60]
    trace=[]
    for c in tr.columns:
        if c=="time":continue
        s=service(c.split("_")[0]);
        if s not in SERVICES:continue
        a=pd.to_numeric(n[c],errors="coerce").fillna(0);b=pd.to_numeric(p[c],errors="coerce").fillna(0)
        trace.append((s,round(float(b.mean()-a.mean()),2)))
    trace=sorted(trace,key=lambda x:-x[1])[:8]
    return metric,trace

def ask(metric,trace):
    key,base=actor_config()
    prompt=("You are ThinkFL's lightweight failure-localization actor. A fixed 60-second healthy reference and later incident data were observed. "
      "Tool search_fluctuating_metrics returned: "+json.dumps(metric)+". Tool search_traces (service error changes) returned: "+json.dumps(trace)+". "
      "Reason briefly, self-check that downstream symptoms may not be roots, then return ONLY JSON: {\"ranking\":[five service names]}. "
      "Candidates: "+json.dumps(SERVICES))
    r=requests.post(base+"/chat/completions",headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},json={"model":"gpt-4o-mini","messages":[{"role":"system","content":"Use only supplied tool evidence."},{"role":"user","content":prompt}],"temperature":0,"max_tokens":180},timeout=90)
    r.raise_for_status(); content=r.json()["choices"][0]["message"]["content"]
    match=re.search(r"\{.*\}",content,re.S); data=json.loads(match.group(0)); return [service(x) for x in data.get("ranking",[])],content

def rank(row):
    metric,trace=evidence(row);fallback=[x[0] for x in metric]
    try: proposed,raw=ask(metric,trace)
    except Exception as e: proposed,raw=[],"API_FALLBACK:"+type(e).__name__
    ordered=[]
    for s in proposed+fallback+list(SERVICES):
        if s in SERVICES and s not in ordered:ordered.append(s)
    return ordered,{"metric_tool":metric,"trace_tool":trace,"actor_output":raw,"api_used":not raw.startswith("API_FALLBACK")}

def main():
 p=argparse.ArgumentParser();p.add_argument("--split",choices=("validation","test"),required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();rs=rows(a.split);pred=[];h=np.zeros(5)
 for i,row in enumerate(rs,1):
  r,d=rank(row);pos=r.index(row.root)+1;h += [pos<=k for k in range(1,6)];pred.append({"incident_id":row.incident_id,"ground_truth_service":row.root,"ranking":r[:5],"rank":pos,"diagnostics":d});print(f"[{i}/{len(rs)}] {row.incident_id}: rank={pos} top5={r[:5]}",flush=True)
 metrics={"A@1":float(h[0]/len(rs)),"A@5":float(h[4]/len(rs)),"Avg@5":float(h.mean()/len(rs)),"cases":len(rs)};config={"variant":"ThinkFL-GPT-4o-mini tool-interface compatibility variant","reference_policy":"first fixed 60 seconds","uses_injection_time_at_inference":False,"uses_labels_at_inference":False,"actor":"gpt-4o-mini temperature=0","tools":["search_traces","search_fluctuating_metrics"],"self_refinement":"single actor self-check before JSON ranking"};a.output.mkdir(parents=True,exist_ok=True);(a.output/"summary.json").write_text(json.dumps({"metrics":metrics,"config":config},indent=2),encoding="utf8");(a.output/"predictions_private.json").write_text(json.dumps(pred,indent=2),encoding="utf8");print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
