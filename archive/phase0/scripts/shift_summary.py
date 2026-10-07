import json,glob,numpy as np,collections as C
R=C.defaultdict(list)
for f in glob.glob("outputs/shift_*/shift-v1/e*/seed*.json"):
    enc=f.split("/")[1][6:]; r=json.load(open(f)); m=r["metrics"]; cell=r["cell"]; e,rung,unit=cell.split("-",2)
    if e=="e2":
        R[(e,enc,rung)].append((m["test_src"]["icbhi_score"],m["test_tgt"]["icbhi_score"],m["test_src"]["split"]["coverage"],m["test_tgt"]["split"]["coverage"],m["test_src"]["mondrian"]["coverage"],m["test_tgt"]["mondrian"]["coverage"],m["test_tgt"].get("decodability_auc",np.nan),unit))
    else:
        if "test_unseen" not in m: continue
        t=m["test_unseen"]; R[(e,enc,rung)].append((t["icbhi_score"],t["split"]["coverage"],t["mondrian"]["coverage"],t.get("decodability_auc",np.nan),unit))
print("E2 KAUH: Score src->tgt | V1 cov src->tgt | V2 cov src->tgt | AUC | units")
for k,v in sorted(R.items()):
    if k[0]!="e2":continue
    a=np.array([x[:7] for x in v],float);print(k[1:],"%.3f->%.3f | %.3f->%.3f | %.3f->%.3f | %.2f | n=%d"%(*a[:,:6].mean(0),np.nanmean(a[:,6]),len(a)))
print("\nE1 held-out device (test_unseen), mean over devices of seed-mean; deficit=0.9-cov")
for k,v in sorted(R.items()):
    if k[0]!="e1":continue
    byd=C.defaultdict(list)
    for x in v: byd[x[4]].append(x[:4])
    d={u:np.array(z,float).mean(0) for u,z in byd.items()}
    print(k[1:],"Score %.3f  V1 cov %.3f  V2 cov %.3f  AUC %.2f"%tuple(np.mean(list(d.values()),0)), " per-device V1:",{u:round(float(z[1]),2) for u,z in d.items()})
