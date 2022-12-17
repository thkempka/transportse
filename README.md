    +------------------------------------------+
    ░▀█▀░█▀▄░█▀█░█▀█░█▀▀░█▀█░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
    ░░█░░█▀▄░█▀█░█░█░▀▀█░█▀▀░█░█░█▀▄░░█░░▀▀█░█▀▀
    ░░▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀▀▀
    +---| TRANSPORT Simulation Environment |---+

Python script to calculate fluid flow in porous media, and advective, diffusive and dispersive transport of chemical species as well as conductive and convective transport of heat.

Installation of miniforge or mambaforge environment (for Anaconda please check documentation):

    mamba|conda env create --file=environment.yml

tse.py: program code (requires custom config file to run)

	Please use this command to run tse.py:
	[i]python [-i] [-u] [path]tse.py [path]<config file name> [2>&1 | tee <log file name>]

	Examples:
	python tse.py config.py
	python ../../tse.py config.py
	python tse.py simulation/config.py
	python tse.py ~/tse/simulation/config.py

tse2pvd.py: program to convert TRANSPORTSE output files to Paraview format (pvd/vts)

