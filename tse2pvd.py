# Licence: GNU General Public License, Version 3, 29 June 2007
# Copyright © 2022 Helmholtz Centre Potsdam GFZ German Research Centre
# for Geosciences, Potsdam, Germany
# Citation: Kempka, T. (2020): Verification of a Python-based TRANSPORT
# Simulation Environment for density-driven fluid flow and coupled transport of
# heat and chemical species. Advances in Geosciences, 54, 67-77.
# https://doi.org/10.5194/adgeo-54-67-2020

'''
    Program to convert TRANSPORTSE simulation output files to Paraview data
'''

import glob
from pyevtk.hl import gridToVTK
from pyevtk.vtk import VtkGroup
import numpy
import joblib

vtkg = VtkGroup("sim")

for num, filename in enumerate(sorted(glob.glob('*[0-9].p'))):
    with open(filename, 'rb') as pfile:
        lx, ly, lz, nx, ny, nz, dx, dy, dz, t_sim, dt, P, C, \
                T, qx, qy, qz, phi, Kx, Ky, Kz, rho_f, nue, cp_f, l_f, \
                BP, BC, BT, L, mass, WP, WT = joblib.load(pfile)

        print(filename)
        # geometric assignments only once
        if num == 0:
            # Dimensions
            if numpy.shape(dx) == ():
                dx = numpy.ones(nx)*lx/nx
                dy = numpy.ones(ny)*ly/ny
                dz = numpy.ones(nz)*lz/nz

            # Coordinates
            X = numpy.cumsum(numpy.append([0], dx))
            Y = numpy.cumsum(numpy.append([0], dy))
            Z = numpy.cumsum(numpy.append([0], dz))

            x = numpy.zeros((nx + 1, ny + 1, nz + 1))
            y = numpy.zeros((nx + 1, ny + 1, nz + 1))
            z = numpy.zeros((nx + 1, ny + 1, nz + 1))

            for i in range(nx + 1):
                for j in range(ny + 1):
                    for k in range(nz + 1):
                        x[i,j,k] = X[i]
                        y[i,j,k] = Y[j]
                        z[i,j,k] = Z[k]

        # Copy of concentration arrays
        C0 = C[:,:,:,0].copy()
        #C1 = C[:,:,:,1].copy()
        #C2 = C[:,:,:,2].copy()
        #C3 = C[:,:,:,3].copy()

        #GPP = numpy.zeros((nx,ny,nz))
        #GPP[GP == True] = 1

        vtsfile = "sim"+str(num).zfill(5)

        gridToVTK(vtsfile, x, y, z,
          cellData = {"Temperature" : T,
                      "Pressure" : P,
                      "Porosity" : phi,
                      "Kx" : Kx,
                      "Ky" : Ky,
                      "Kz" : Kz,
                      "Density" : rho_f,
                      "Viscosity " : nue,
                      "Tracer" : C0,
                      #"NaCl" : C1,
                      #"Hydrate (m^3/m^3)" : C1,
                      #"CH4 (kg/m^3)" : C2,
                      #"Inhibitor (kg/m^3)" : C3,
                      "qx" : qx,
                      "qy" : qy,
                      "qz" : qz,
                      "BP" : BP,
                      "BC" : BC,
                      "BT" : BT,
                      #"GP (-)" : GPP,
                      "L" : L,
                      "WP" : WP,
                      "WT" : WT,
                      })

        # add vts to pvd file
        vtkg.addFile(filepath = vtsfile+".vts", sim_time = t_sim)

# save pvd file
vtkg.save()
