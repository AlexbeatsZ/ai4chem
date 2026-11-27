"""Finite local supervisor: submit a prepared pilot only after previous native owner drains."""
import json,os,subprocess,time,zipfile
from datetime import datetime,timedelta,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/active-learning-dispatch-20261127'
PILOT=ROOT/'results/active-learning-pilot-20261127'
RELAY='meta@100.113.70.121'
RELROOT='C:/Users/Meta/source/repos/ai4chem'
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    deadline=datetime.now(timezone.utc)+timedelta(hours=12)
    state=dict(pid=os.getpid(),deadline_utc=deadline.isoformat(),status='waiting_for_pilot',holdout_labels_evaluated=False)
    def save(**kw):
        state.update(kw,updated_at_utc=datetime.now(timezone.utc).isoformat());p=OUT/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(OUT/'state.json')
    def relay(action):
        return json.loads(subprocess.check_output(['ssh','-n','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',RELAY,'powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',RELROOT+'/scripts/relay_active_pilot.ps1','-Action',action],text=True,timeout=100).lstrip('\ufeff'))
    save()
    try:
        while datetime.now(timezone.utc)<deadline:
            p=PILOT/'state.json'
            d=json.loads(p.read_text()) if p.exists() else {}
            if d.get('status')=='failed':save(status='pilot_failed');return
            if d.get('status')!='prepared_not_dispatched':time.sleep(60);continue
            try:status=relay('Status')
            except Exception as e:save(status='waiting_for_relay',error=repr(e));time.sleep(120);continue
            save(status='waiting_for_prior_dft',observed=status,error=None)
            if status['exists']:save(status='existing_remote_pilot_no_retry');return
            if status['previous'] in ('blocked','admission_deadline_reached'):save(status='predecessor_blocked');return
            if status['previous']!='chain_terminal' or status['owner_alive'] or status['gaussian_count']:time.sleep(120);continue
            q=json.loads((PILOT/'queue.json').read_text());q['admission_deadline_utc']=deadline.isoformat()
            archive=OUT/'active-learning-pilot.zip'
            with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
                z.writestr('queue.json',json.dumps(q,indent=2))
                z.write(PILOT/'protocol.json','protocol.json')
                for item in q['force_items']:z.write(PILOT/item['input'],item['input'])
            with zipfile.ZipFile(archive) as z:assert z.testzip() is None
            subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(archive),RELAY+':'+RELROOT+'/results/coordination/equal-budget-20261127/active-learning-pilot.zip'],check=True,timeout=120)
            save(status='dispatching',unique_labels=len(q['force_items']))
            receipt=relay('Dispatch');save(status='submitted',receipt=receipt)
            # Existing collector uses the verified relay transport and raw ZIP CRC.
            collector=[os.sys.executable,str(ROOT/'scripts/collect_local_queue.py'),'--relay-ssh',RELAY,'--relay-root',RELROOT,'--worker-root','C:/Users/Boywe/Project/Workspaces/ai4chem-worker','--queue','C:/Users/Boywe/Project/Workspaces/ai4chem-worker/results/active-learning-pilot-20261127','--output',str(OUT/'return'),'--worker-host-name','GUGUGAGA.local','--worker-host-key-alias','192.168.31.41','--max-hours','12']
            with (OUT/'collector.log').open('w') as log:subprocess.Popen(collector,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            return
        save(status='admission_deadline_reached')
    except Exception as e:save(status='blocked_no_retry',error=repr(e));raise
if __name__=='__main__':main()
