'''
    Pressure solver benchmark: SOR vs CG at different thread counts.

    Runs the same 3-D density-driven flow problem with both solvers for
    each requested thread count and prints runtimes and P iteration
    counts.

    Usage:
        uv run python tests/benchmark_solvers.py [nx ny nz] [threads ...]

    Defaults: 101 51 10, threads 1 2 4 8 16 32
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


def run_case(name, psolver, threads, nx, ny, nz):
    rundir = os.path.join(tempfile.gettempdir(), 'tse_bench_' + name)
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    cfg = CONFIG_TEMPLATE.format(psolver=psolver, nx=nx, ny=ny, nz=nz)
    cfg = cfg.replace('parallel_processes = 1',
                      'parallel_processes = %d' % threads)
    with open(os.path.join(rundir, 'config.py'), 'w') as f:
        f.write(cfg)
    start = time.time()
    log = subprocess.run(
        [sys.executable, '-u', os.path.join(REPO, 'tse.py'), 'config.py'],
        cwd=rundir, check=True, timeout=7200,
        capture_output=True, text=True).stdout
    runtime = time.time() - start
    match = re.search(r'\[P\]\s*(\d+)', log[log.rfind('SIMULATION SUMMARY'):])
    iterations = int(match.group(1)) if match else -1
    shutil.rmtree(rundir, ignore_errors=True)
    return runtime, iterations


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 51
    nz = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    threads = ([int(t) for t in sys.argv[4:]] if len(sys.argv) > 4
               else [1, 2, 4, 8, 16, 32])

    print('%dx%dx%d grid, %d cells' % (nx, ny, nz, nx*ny*nz))
    print('%8s %8s %10s %12s %8s' % ('solver', 'threads', 'runtime/s',
                                     'P iters', 'speedup'))
    base = {}
    for psolver in ('SOR', 'CG'):
        for nthreads in threads:
            name = '%s_t%d' % (psolver.lower(), nthreads)
            runtime, iterations = run_case(name, psolver, nthreads, nx, ny, nz)
            if psolver not in base:
                base[psolver] = runtime
            print('%8s %8d %10.1f %12d %7.2fx' % (psolver, nthreads, runtime,
                                                  iterations,
                                                  base[psolver]/runtime))


if __name__ == '__main__':
    main()
