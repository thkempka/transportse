    +------------------------------------------+
    ░▀█▀░█▀▄░█▀█░█▀█░█▀▀░█▀█░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
    ░░█░░█▀▄░█▀█░█░█░▀▀█░█▀▀░█░█░█▀▄░░█░░▀▀█░█▀▀
    ░░▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀▀▀
    +---| TRANSPORT Simulation Environment |---+

Python script to calculate fluid flow in porous media, and advective, diffusive and dispersive transport of chemical species as well as conductive and convective transport of heat.

Installation (requires [uv](https://docs.astral.sh/uv/), creates a local .venv):

    uv sync

Run with the project environment:

    uv run python tse.py config.py

tse.py: program code (requires custom config file to run)

	Please use this command to run tse.py:
	[i]python [-i] [-u] [path]tse.py [path]<config file name> [2>&1 | tee <log file name>]

	Examples:
	python tse.py config.py
	python ../../tse.py config.py
	python tse.py simulation/config.py
	python tse.py ~/tse/simulation/config.py

tse2pvd.py: program to convert TRANSPORTSE output files to Paraview format (pvd/vts)

