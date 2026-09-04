"""Reference logarithmic derivative D = d ln(R_L/a) / d ln q_E.

Central differences of the extended-precision solver at delta = 5e-3.
q_E = 10^k, k = -4, -4+1/6, ..., 4 (49 points), plus table extras 0.3 and 3.
Checkpoints after each mass ratio so the run can resume.
"""
import json
import os
import sys
import time

import numpy as np

from reference import Req, lobe
from paths import PAPER

DELTA = 5e-3
NPTS = 200
HERE = str(PAPER)
OUT = os.path.join(HERE, "derivative_reference.txt")
PARTIAL = os.path.join(HERE, "derivative_reference_partial.json")
VERIFY = os.path.join(HERE, "derivative_verify.json")


def R_of_qE(qE, npts=NPTS):
    V, *_ = lobe(1.0 / qE, npts)
    return float(Req(V))


def D_at(qE, delta, npts=NPTS):
    Rp = R_of_qE(qE * np.exp(+delta), npts)
    Rm = R_of_qE(qE * np.exp(-delta), npts)
    D = (np.log(Rp) - np.log(Rm)) / (2.0 * delta)
    return float(D), float(Rp), float(Rm)


def grid_qE():
    ks = np.arange(-4.0, 4.0 + 1e-12, 1.0 / 6.0)
    qE = 10.0 ** ks
    extra = np.array([0.3, 3.0])
    return np.unique(np.concatenate([qE, extra]))


def load_partial():
    if not os.path.exists(PARTIAL):
        return {}
    with open(PARTIAL, encoding="utf-8") as f:
        return json.load(f)


def save_partial(data):
    tmp = PARTIAL + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, PARTIAL)


def write_table(data):
    ks = np.arange(-4.0, 4.0 + 1e-12, 1.0 / 6.0)
    qE49 = 10.0 ** ks
    rows = []
    for q in qE49:
        key = f"{q:.12e}"
        # match by closeness
        hit = None
        for k, v in data.items():
            if abs(float(v["qE"]) / q - 1.0) < 1e-8:
                hit = v
                break
        if hit is None:
            raise SystemExit(f"missing grid point qE={q}")
        rows.append((q, hit["D"]))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(
            "# q_E=M1/M2   D_ref = d ln(R_L/a) / d ln q_E\n"
            "# central difference, delta=5e-3, npts=200\n"
        )
        for q, D in rows:
            f.write(f"{q:.12e}  {D:.12e}\n")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    t0 = time.time()
    if mode == "grid":
        qEs = grid_qE()
        data = load_partial()
        done = 0
        for i, qE in enumerate(qEs):
            key = f"{qE:.12e}"
            if key in data or any(abs(float(v["qE"]) / qE - 1.0) < 1e-8 for v in data.values()):
                done += 1
                print(f"[{i+1}/{len(qEs)}] qE={qE:.6e} cached", flush=True)
                continue
            t1 = time.time()
            D, Rp, Rm = D_at(qE, DELTA)
            data[key] = dict(qE=float(qE), D=D, Rp=Rp, Rm=Rm, delta=DELTA, elapsed=time.time() - t1)
            save_partial(data)
            print(
                f"[{i+1}/{len(qEs)}] qE={qE:.6e}  D={D:.10f}  "
                f"dt={time.time()-t1:.1f}s  total={time.time()-t0:.0f}s",
                flush=True,
            )
        write_table(data)
        print("wrote", OUT, "total", time.time() - t0, flush=True)
    elif mode == "verify":
        out = {}
        for qE in (0.1, 1.0, 10.0):
            D1, *_ = D_at(qE, DELTA)
            D2, *_ = D_at(qE, DELTA / 2)
            out[str(qE)] = dict(D_delta=D1, D_half=D2, absdiff=abs(D1 - D2))
            print(f"qE={qE:g}  D(d)={D1:.12f}  D(d/2)={D2:.12f}  |d|={abs(D1-D2):.3e}", flush=True)
        with open(VERIFY, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("wrote", VERIFY, flush=True)
    else:
        raise SystemExit("usage: make_derivative_reference.py [grid|verify]")


if __name__ == "__main__":
    main()
