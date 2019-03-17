# Water-Rocket-Flight-Simulator
Simulation of Water Rocket Ascent with Fill Ratio Optimization

This code was developed for the Airbus Rocket Sloshing Workshop 2018/2019.

It is currently set-up to optimize a given design introduced by the user in terms of the fill ratio capable of optimizing the apogee height. The code allows the simulation of launch tubes, and also the use of a number of propellant water bottles greater than 1. Also, the sloshing mass assumed by the code is hard-coded to the total mass of propellant water used, which is the bare minimum required by the competition. The code also shows plots of various flight variables during flihgt, such as altitude, rocket velocity, acceleration and thrust, exhaust velocity, pressure, density, temperature and mass flows, although they are currently commented and disabled.

The code is currently set-up to be run quickly from the console, by using, for example:
> python3 WaterRocketFlight.py

The user is then asked to insert several parameters:
1. Structural mass of the rocket (without sloshing);
2. Volume of each propellant water bottle;
3. Number of propellant bottles;
4. Launch tube length;
5. Lower limit of the fill ratio to look for maximum apogee height;
6. Upper limit of the fill ratio to look for maximum apogee height;

Any questions regarding the code's functionality may be sent to: luismsfern@gmail.com
