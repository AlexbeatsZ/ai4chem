"""Evaluate three fixed coverage2048 seeds on existing quarantined DFT labels."""
import json,os
from pathlib import Path
import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ensemble-curvature-20261127'
def save(name,obj):
    p=OUT/name; t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');t.replace(p)
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),status='running',holdout_labels_evaluated=False)
    save('state.json',state)
    try:
        torch.set_num_threads(4)
        frames=read(ROOT/'results/coordination/compute-20261126/offpath-return/alignment/quarantined.extxyz',index=':')
        assert len(frames)==288
        calculators=[MACECalculator(model_paths=str(ROOT/f'models/v0.10.0/mh1_coverage2048_s{i}.model'),device='cpu',default_dtype='float64',head='Default') for i in range(3)]
        records=[]; groups={}
        for i,a in enumerate(frames):
            forces=[]
            for c in calculators:
                a.calc=c;forces.append(a.get_forces().copy())
            p=np.asarray(forces); ref=a.arrays['REF_forces']
            disagreement=float(np.sqrt(np.mean((p-p.mean(axis=0))**2)))
            error=float(np.sqrt(np.mean((p[0]-ref)**2)))
            records.append(dict(reaction=a.info['reaction'],mode=a.info['mode'],disagreement=disagreement,force_error=error))
            groups.setdefault(a.info['reaction'],{})[a.info['mode']]=(a.positions.copy(),ref.copy(),p)
            if (i+1)%16==0:save('state.json',dict(state,frames=i+1))
        curvature=[]
        for reaction,g in groups.items():
            for mode in sorted(g):
                if '-plus-' not in mode:continue
                r1,f1,p1=g[mode];r2,f2,p2=g[mode.replace('-plus-','-minus-')]
                rr=g['reference'][0]
                assert np.max(np.abs((r1+r2)/2-rr))<2e-6
                delta=np.linalg.norm(r1-r2)
                df=-(f1-f2)/delta;dp=-(p1-p2)/delta
                curvature.append(dict(reaction=reaction,mode=mode,hvp_rmse=[float(np.sqrt(np.mean((v-df)**2))) for v in dp]))
        d=np.array([r['disagreement'] for r in records]);e=np.array([r['force_error'] for r in records])
        top=np.argsort(d)[-72:]
        rng=np.random.default_rng(20261127)
        random=[float(e[rng.choice(len(e),72,replace=False)].mean()) for _ in range(1000)]
        save('summary.json',dict(frames=288,pairs=len(curvature),spearman=float(spearmanr(d,e).statistic),top_quartile_error=float(e[top].mean()),random_mean=float(np.mean(random)),random_95=np.quantile(random,[.025,.975]).tolist(),hvp_rmse_mean=np.mean([c['hvp_rmse'] for c in curvature],axis=0).tolist(),records=records,curvature=curvature,scope='development training-side quarantined labels; no merge; fixed perturbations, not active learning'))
        save('state.json',dict(state,status='completed',frames=288))
    except BaseException as e:
        save('state.json',dict(state,status='failed',error=repr(e)));raise
if __name__=='__main__':main()
