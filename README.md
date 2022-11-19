    +------------------------------------------+
    ░▀█▀░█▀▄░█▀█░█▀█░█▀▀░█▀█░█▀█░█▀▄░▀█▀░█▀▀░█▀▀
    ░░█░░█▀▄░█▀█░█░█░▀▀█░█▀▀░█░█░█▀▄░░█░░▀▀█░█▀▀
    ░░▀░░▀░▀░▀░▀░▀░▀░▀▀▀░▀░░░▀▀▀░▀░▀░░▀░░▀▀▀░▀▀▀
    +---| TRANSPORT Simulation Environment |---+

Python script to calculate fluid flow in porous media, and advective, diffusive and dispersive transport of chemical species as well as conductive and convective transport of heat.

tse.py: program code (requires custom config file to run)

	Please use this command to run tse.py:
	ipython -i [path]tse.py [path]<config file name>

	Examples:
	ipython -i [path]tse.py config.py
	ipython -i [path]tse.py simulation/config.py
	ipython -i [path]tse.py ~/tse/simulation/config.py

tse2pvd.py: program to convert TRANSPORTSE output files to Paraview format (pvd)



