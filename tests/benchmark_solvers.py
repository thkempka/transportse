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
LOGDIR = os.path.join(REPO, 'benchmark_logs')

# simulation time (s) and maximum time step size (s) for the benchmark
# (overridable via environment variables)
SIM_TIME = float(os.environ.get('SIM_TIME', 1000.0))
DT_MAX = float(os.environ.get('DT_MAX', 200.0))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_sor_cg_equivalence import CONFIG_TEMPLATE  # noqa: E402


def run_case(name, psolver, threads, nx, ny, nz):
    rundir = os.path.join(tempfile.gettempdir(), 'tse_bench_' + name)
    shutil.rmtree(rundir, ignore_errors=True)
    os.makedirs(rundir)
    os.makedirs(LOGDIR, exist_ok=True)
    cfg = CONFIG_TEMPLATE.format(psolver=psolver, nx=nx, ny=ny, nz=nz)
    cfg = cfg.replace('parallel_processes = 1',
                      'parallel_processes = %d' % threads)
    # simulation time scaled to the grid: keep the number of time steps
    # moderate (Courant number near the well controls dt anyway)
    cfg = cfg.replace('t = 3000.0', 't = %g' % SIM_TIME)
    cfg = cfg.replace('dt_max = 1000.0', 'dt_max = %g' % DT_MAX)
    cfg = cfg.replace('PRINTOUT = 1000', 'PRINTOUT = 5')
    cfg = cfg.replace('PLOTSAVE = True', 'PLOTSAVE = False')
    with open(os.path.join(rundir, 'config.py'), 'w') as f:
        f.write(cfg)
    with open(os.path.join(LOGDIR, name + '_config.py'), 'w') as f:
        f.write(cfg)
    logpath = os.path.join(LOGDIR, name + '.log')
    print('starting %s (writing %s) ...' % (name, logpath), flush=True)
    start = time.time()
    with open(logpath, 'w') as f:
        subprocess.run(
            [sys.executable, '-u', os.path.join(REPO, 'tse.py'), 'config.py'],
            cwd=rundir, check=True, timeout=7200,
            stdout=f, stderr=subprocess.STDOUT, text=True)
    runtime = time.time() - start
    with open(logpath) as f:
        log = f.read()
    summary = log[log.rfind('SIMULATION SUMMARY'):]
    match = re.search(r'\[P\]\s*(\d+)', summary)
    iterations = int(match.group(1)) if match else -1
    cmax = re.findall(r'\[C\+\]\s*([0-9.]+)', log)
    maxc = float(cmax[-1]) if cmax else float('nan')
    reductions = log.count('Reducing dt')
    with open(os.path.join(LOGDIR, name + '.log'), 'w') as f:
        f.write(log)
    shutil.rmtree(rundir, ignore_errors=True)
    return runtime, iterations, maxc, reductions


def main():
    nx = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    ny = int(sys.argv[2]) if len(sys.argv) > 2 else 51
    nz = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    threads = ([int(t) for t in sys.argv[4:]] if len(sys.argv) > 4
               else [1, 2, 4, 8, 16, 32])

    print('%dx%dx%d grid, %d cells' % (nx, ny, nz, nx*ny*nz))
    print('%8s %8s %10s %12s %8s  %9s  %6s'
          % ('solver', 'threads', 'runtime/s', 'P iters', 'speedup',
             'max C', 'dtred'))
    resultsfile = os.path.join(LOGDIR, 'results.txt')
    os.makedirs(LOGDIR, exist_ok=True)
    with open(resultsfile, 'a') as f:
        f.write('# %dx%dx%d grid, %d cells, t = %g s, dt_max = %g s\n'
                % (nx, ny, nz, nx*ny*nz, SIM_TIME, DT_MAX))
    base = {}
    for psolver in ('SOR', 'CG'):
        for nthreads in threads:
            name = '%s_t%d' % (psolver.lower(), nthreads)
            runtime, iterations, maxc, reductions = run_case(
                name, psolver, nthreads, nx, ny, nz)
            if psolver not in base:
                base[psolver] = runtime
            row = ('%8s %8d %10.1f %12d %7.2fx  %9.4f  %6d'
                   % (psolver, nthreads, runtime, iterations,
                      base[psolver]/runtime, maxc, reductions))
            print(row, flush=True)
            with open(resultsfile, 'a') as f:
                f.write(row + '\n')


if __name__ == '__main__':
    main()
