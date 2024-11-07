# Aircraft Takeoff Simulation

This program simulates aircraft takeoff performance and optimizes lift coefficient and thrust for a given target takeoff distance.

## Installation

1. Ensure you have Python 3.7 or later installed on your system.

2. Clone this repository or download the script and requirements.txt file.

3. Open a terminal/command prompt and navigate to the directory containing the files.

4. Install the required packages using pip:
pip install -r requirements.txt


   This will install all necessary dependencies for the program.

## Running the Program

1. In the terminal/command prompt, run the script using Python:

python aircraft_takeoff_simulation.py

2. Follow the prompts to enter the required input parameters:
   - Electric motor watt limit
   - Wingspan (ft)
   - Chord (ft)
   - Target takeoff distance (ft)
   - Aircraft weight (lbs)
   - Atmospheric density ratio (sigma)

3. The program will display a progress bar during optimization, then show the results and display plots of the takeoff performance.

## Output

The program will output:
- Optimal lift coefficient (CL_max)
- Optimal thrust
- Takeoff distance
- Time to reach takeoff velocity
- Takeoff velocity
- Plots of velocity, distance, acceleration, and thrust during takeoff