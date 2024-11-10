# Aircraft Takeoff Analysis Tool

## Overview
A Python tool for analyzing aircraft takeoff performance by optimizing lift coefficient (CL) and thrust across different wing configurations.

## Quick Start
Open a new command line in the Aircraft Takeoff Analysis Tool directory

Install required packages
pip install numpy matplotlib pandas

Run the analysis
python main.py


## Configuration
Edit `config.py` to set analysis parameters:
python
WINGSPAN = 30.0        # feet
WEIGHT = 1000.0       # pounds
TARGET_TAKEOFF_DISTANCE = 500.0  # feet
CHORD_SWEEP = True    # Enable chord sweep analysis

## Key Features
- Wing chord length optimization
- CL and thrust calculations
- Takeoff trajectory analysis
- Performance visualization
- Automated data logging
- Result plotting

## File Structure
├── main.py           # Entry point
├── config.py         # Configuration settings
├── analysis.py       # Core analysis logic
├── optimization.py   # Optimization algorithms
├── physics.py        # Physics calculations
├── plotting.py       # Visualization functions
└── results/          # Output directory
├── data/         # CSV results
└── plots/        # Generated plots

## Usage

### Running Analysis
1. Configure parameters in `config.py`
2. Run `python main.py`
3. Results saved to `results/` directory

### Analysis Modes
- **Chord Sweep**: Set `CHORD_SWEEP = True`
  - Analyzes multiple chord lengths
  - Generates comparison plots
  
- **Single Analysis**: Set `CHORD_SWEEP = False`
  - Analyzes one configuration
  - Detailed single-point results

## Output Files
- `results/data/takeoff_results.csv`: Analysis data
- `results/plots/`: Performance visualizations
  - Takeoff trajectories
  - CL vs chord plots
  - Thrust optimization plots

  ## Troubleshooting

### Common Issues
1. **Missing Results Directory**
   - Automatically created on first run
   - Check write permissions

2. **Plot Generation Errors**
   - Verify matplotlib installation
   - Check available memory

3. **Slow Performance**
   - Reduce chord sweep range
   - Increase step size
   - Disable debug mode

## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- Pandas


## Contact
Paul-Michael Letulle