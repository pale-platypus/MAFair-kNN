"""
runjobs.py — run shell commands from a jobs file with N parallel workers.

    python scripts/runjobs.py JOBS.txt --workers 4

Each non-empty, non-# line of JOBS.txt is one command, run from the repo root;
a leading "python" is replaced by the interpreter running this script. A job
whose command hash appears in logs/done.txt is skipped, so re-running the same
jobs file resumes where it stopped. Output of every job goes to logs/<hash>.log.
"""
import argparse, hashlib, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGD = os.path.join(ROOT, "logs")
DONE = os.path.join(LOGD, "done.txt")


def h(cmd):
    return hashlib.sha1(cmd.encode()).hexdigest()[:12]


def run(cmd):
    t0 = time.time()
    log = os.path.join(LOGD, h(cmd) + ".log")
    exe = f'"{sys.executable}" ' + cmd[len("python "):] if cmd.startswith("python ") else cmd
    with open(log, "w", encoding="utf-8") as f:
        f.write(cmd + "\n\n")
        f.flush()
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        rc = subprocess.call(exe, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env)
    return cmd, rc, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(LOGD, exist_ok=True)
    done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
    cmds = [l.strip() for l in open(a.jobs, encoding="utf-8") if l.strip() and not l.strip().startswith("#")]
    todo = [c for c in cmds if h(c) not in done]
    print(f"{len(cmds)} jobs, {len(cmds) - len(todo)} already done, running {len(todo)} with {a.workers} workers", flush=True)
    n_ok = n_bad = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, c) for c in todo]
        for fu in as_completed(futs):
            cmd, rc, dt = fu.result()
            if rc == 0:
                n_ok += 1
                with open(DONE, "a") as f:
                    f.write(h(cmd) + "\n")
            else:
                n_bad += 1
            print(f"[{'OK' if rc == 0 else 'FAIL rc=%d' % rc}] {dt:7.1f}s {h(cmd)} {cmd[:150]}", flush=True)
    print(f"finished: {n_ok} ok, {n_bad} failed", flush=True)
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
