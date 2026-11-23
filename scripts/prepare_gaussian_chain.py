"""Prepare exact native Windows inputs for one frozen pilot reaction.

Uses the standard library only. Copy the result folder, run_gaussian_job.ps1,
scripts/run_gaussian_chain.ps1 and scripts/status_gaussian_chain.ps1 to a worker.
"""
import argparse
import json
from pathlib import Path


def prepare(protocol, reaction, output):
    job = next(j for j in protocol["jobs"] if j["reaction"] == reaction and j["engine"] == "Gaussian16W-A.03")
    configuration = {key: protocol[key] for key in ("version", "dft_method", "dft_nproc", "dft_mem", "charge",
                                                   "multiplicity", "dft_policy", "irc_endpoint_rmsd_A")}
    task = {**job, "configuration": configuration, "protocol_frozen_at_utc": protocol["frozen_at_utc"]}
    output.mkdir(parents=True, exist_ok=False)
    (output/"task.json").write_text(json.dumps(task, indent=2)+"\n")
    common = "scf=(xqc,maxcycle=128) int=ultrafine nosymm"

    def write(name, route, old=None, atoms=False):
        lines = ([f"%oldchk={old}.chk"] if old else [])
        lines += [f"%chk={name}.chk", f"%mem={configuration['dft_mem']}",
                  f"%nprocshared={configuration['dft_nproc']}", f"#p {configuration['dft_method']} {route} {common}", ""]
        if atoms:
            lines += [f"ai4chem {configuration['version']} {reaction} post-hoc DFT diagnostic", "",
                      f"{configuration['charge']} {configuration['multiplicity']}"]
            lines += [f"{s} {p[0]:.12f} {p[1]:.12f} {p[2]:.12f}" for s, p in zip(job["symbols"], job["positions_A"], strict=True)]
        (output/f"{name}.gjf").write_text("\n".join(lines)+"\n\n", encoding="ascii")

    write("ts", "opt=(ts,calcfc,noeigentest,maxcycles=80) freq", atoms=True)
    for direction in ("forward", "reverse"):
        name = f"irc_{direction}"
        write(name, f"irc=(rcfc,{direction},lqa,maxpoints=160,stepsize=5,maxcycle=80) geom=allcheck guess=read", old="ts")
        write(name+"_extended", "irc=(restart,maxpoints=320) geom=allcheck guess=read", old=name)
        for suffix in ("", "_extended"):
            write(f"minimum_{direction}{suffix}", "opt=(calcfc,maxcycles=100) freq geom=allcheck guess=read", old=name+suffix)
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--reaction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = prepare(json.loads(args.protocol.read_text()), args.reaction, args.output)
    print(json.dumps({"job_id": task["job_id"], "output": str(args.output.resolve()), "input_files": 9}))


if __name__ == "__main__":
    main()
