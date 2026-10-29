'''
    tse2csv.py - convert TSE pickle output (*.p) into a flat CSV table.

    Updated for the current tse.py pickle format, which carries two additional
    fields at the end of the tuple (WC, e). The first 31 entries are identical
    to the old format, so the unpacking below is format agnostic.

    usage:  pixi run python tse2csv.py [<glob>]  (default: *[0-9].p)
'''

import sys
import glob
import numpy
import joblib
import pandas as pd

# normalisation reference concentrations (influx composition, mol/kgw)
C_NA, C_CL, C_O2 = 0.005, 0.005, 0.0005

pattern = sys.argv[1] if len(sys.argv) > 1 else '*[0-9].p'

all_data = []
num = 0
x = y = z = None

for filename in sorted(glob.glob(pattern)):
    print(f"Processing file: {filename}")

    (lx, ly, lz, nx, ny, nz, dx, dy, dz, t_sim, dt, P, C,
     T, qx, qy, qz, phi, Kx, Ky, Kz, rho_f, nue, cp_f, l_f,
     BP, BC, BT, L, mass, WP, WT) = joblib.load(filename)[:32]

    # generate grid coordinates (node centres) once
    if num == 0:
        dx = numpy.atleast_1d(dx) * numpy.ones(nx) if numpy.shape(dx) == () \
            else numpy.atleast_1d(dx)
        dy = numpy.atleast_1d(dy) * numpy.ones(ny) if numpy.shape(dy) == () \
            else numpy.atleast_1d(dy)
        dz = numpy.atleast_1d(dz) * numpy.ones(nz) if numpy.shape(dz) == () \
            else numpy.atleast_1d(dz)

        X = numpy.cumsum(numpy.append([0], dx))
        Y = numpy.cumsum(numpy.append([0], dy))
        Z = numpy.cumsum(numpy.append([0], dz))

        x, y, z = numpy.meshgrid(X[:-1], Y[:-1], Z[:-1], indexing='ij')

    data_length = nx * ny * nz
    flat_data = {
        "X": x.ravel(),
        "Y": y.ravel(),
        "Z": z.ravel(),
        "Temperature": T.ravel(),
        "Pressure": P.ravel(),
        "Porosity": phi.ravel(),
        "Kx": Kx.ravel(),
        "Ky": Ky.ravel(),
        "Kz": Kz.ravel(),
        "Density": rho_f.ravel(),
        "Viscosity": nue.ravel(),
        "Tracer": C[:, :, :, 0].ravel(),
        "pH": C[:, :, :, 1].ravel(),
        "pe": C[:, :, :, 2].ravel(),
        "Na (mol/kgw)": C[:, :, :, 3].ravel(),
        "Na - normalised": abs(C[:, :, :, 3] / C_NA).ravel(),
        "Cl (mol/kgw)": C[:, :, :, 4].ravel(),
        "Cl - normalised": abs(C[:, :, :, 4] / C_CL).ravel(),
        "S (mol/kgw)": C[:, :, :, 5].ravel(),
        "O(0) (mol/kgw)": C[:, :, :, 6].ravel(),
        "O(0) - normalised": abs(C[:, :, :, 6] / C_O2).ravel(),
        "Fe (mol/kgw)": C[:, :, :, 7].ravel(),
        "SO4-2 (mol/kgw)": C[:, :, :, 8].ravel(),
        "HCO3- (mol/kgw)": C[:, :, :, 9].ravel(),
        "pCO2(g)": C[:, :, :, 10].ravel(),
        "pO2(g)": C[:, :, :, 11].ravel(),
        "Pyrite (kg/m^3)": C[:, :, :, 12].ravel(),
        "Goethite (kg/m^3)": C[:, :, :, 13].ravel(),
        "qx": qx.ravel(),
        "qy": qy.ravel(),
        "qz": qz.ravel(),
        "BP": BP.ravel(),
        "BC": BC.ravel(),
        "BT": BT.ravel(),
        "L": L.ravel(),
        "WP": WP.ravel(),
        "WT": WT.ravel(),
        "Simulation Time": t_sim,   # same for all rows
    }

    frame = pd.DataFrame(flat_data)
    frame["Simulation Time"] = t_sim
    frame["dt"] = dt
    all_data.append(frame)
    num += 1

if not all_data:
    sys.exit(f"No pickle files matching '{pattern}' found.")

df = pd.concat(all_data, ignore_index=True)
df.to_csv("simulation_output.csv", index=False)
print(f"{len(df)} rows saved to simulation_output.csv")
