'''
    CG preconditioner omega sweep: runs the CG solver on the benchmark
    problem with different SGS over-relaxation factors and reports
    runtime, iteration count and maximum concentration.

    Usage:
        SIM_TIME=600 uv run python tests/cg_omega_sweep.py \
            [nx ny nz] [threads] [omega values ...]

    Defaults: 200 200 25, 8 threads, omega 1.0 1.2 1.4 1.6
'''
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sor_cg_equivalence import CONFIG_TEMPLATE  # noqa: E402

SIM_TIME = float(os.environ.get('SIM_TIME', 600.0))
DT_MAX = float(os.environ.get('DT_MAX', 200.0))


def run_case(omega, threads, nx, ny, nz):
    rundir = os.path.join(tempfile.gettempdir(), 'tse_omega_%s' % omega)
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    cfg = CONFIG_TEMPLATE.format(psolver='CG', nx=nx, ny=ny, nz=nz)
    cfg = cfg.replace('parallel_processes = 1',
                      'parallel_processes = %d' % threads)
    cfg = cfg.replace('t = 3000.0', 't = %g' % SIM_TIME)
    cfg = cfg.replace('dt_max = 1000.0', 'dt_max = %g' % DT_MAX)
    cfg = cfg.replace('PRINTOUT = 1000', 'PRINTOUT = 20')
    cfg = cfg.replace('PLOTSAVE = True', 'PLOTSAVE = False')
    cfg += '\nCG_OMEGA = %g\n' % omega
    with open(os.path.join(rundir, 'config.py'), 'w') as f:
        f.write(cfg)
    start = time.time()
    log = subprocess.run(
        [sys.executable, '-u', os.path.join(REPO, 'tse.py'), 'config.py'],
        cwd=rundir, check=True, timeout=7200,
        capture_output=True, text=True).stdout
    runtime = time.time() - start
    summary = log[log.rfind('SIMULATION SUMMARY'):]
    match = re.search(r'\[P\]\s*(\d+)', summary)
    iterations = int(match.group(1)) if match else -1
    cmax = re.findall(r'\[C\+\]\s*([0-9.]+)', log)
    maxc = float(cmax[-1]) if cmax else float('nan')
    shutil.rmtree(rundir, ignore_errors=True)
    return runtime, iterations, maxc


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    nz = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    omegas = ([float(o) for o in sys.argv[5:]] if len(sys.argv) > 5
              else [1.0, 1.2, 1.4, 1.6])

    print('%dx%dx%d grid, %d cells, %d threads, t = %g s'
          % (nx, ny, nz, nx*ny*nz, threads, SIM_TIME))
    print('%6s %10s %12s %8s %9s' % ('omega', 'runtime/s', 'P iters',
                                     'it/s step', 'max C'))
    for omega in omegas:
        runtime, iterations, maxc = run_case(omega, threads, nx, ny, nz)
        print('%6.2f %10.1f %12d %8.2f %9.4f'
              % (omega, runtime, iterations, runtime/max(iterations, 1),
                 maxc), flush=True)


if __name__ == '__main__':
    main()
