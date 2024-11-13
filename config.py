#config.py Version 1.1
import os
from typing import Dict, List, Union, Any
import numpy as np

class ConfigurationError(Exception):
  """Custom exception for configuration errors."""
  pass

# Aircraft physical parameters
WINGSPAN = 15.0  # feet
WEIGHT = 55.0    # pounds
WATT_LIMIT = 750  # maximum power limit in watts


# Environmental parameters
SIGMA = 1.0      # density ratio
MU = 0.02        # friction coefficient
GRAVITY = 32.2   # ft/s^2
RHO = 0.002378   # slug/ft^3 (sea level air density)
EPSILON = 0.85   # Oswald efficiency factor

# Simulation parameters
DT = 0.1         # seconds (time step)
T_MAX = 30.0     # seconds (maximum simulation time)
WATT_LIMIT = 750  # maximum power limit in watts

# Performance targets
TARGET_TAKEOFF_DISTANCE = 90.0  # feet

# Optimization parameters
CL_MIN = 0.1     # minimum lift coefficient
CL_MAX = 3.0     # maximum lift coefficient
THRUST_MIN = 1.0  # minimum thrust (lbf)
THRUST_MAX = 20.0 # maximum thrust (lbf)

# Optimization algorithm parameters
OPTIMIZATION_PARAMS = {
  'ftol': 1e-6,        # function tolerance
  'maxiter': 100,      # maximum iterations
  'initial_step': 0.1, # initial step size
  'constraint_tol': 1e-6 # constraint tolerance
}

# Chord sweep parameters
CHORD_SWEEP = True     # whether to perform chord sweep analysis
MIN_CHORD = 1.0       # minimum chord length (feet)
MAX_CHORD = 4.0       # maximum chord length (feet)
CHORD_STEP = 0.01     # chord length step size (feet)

# Output parameters
SAVE_PLOTS = True       # whether to save plots
SAVE_DATA = True        # whether to save data to CSV
OUTPUT_DIR = "results"  # output directory for results
PLOT_DIR = f"{OUTPUT_DIR}/plots"  # directory for plots
DATA_DIR = f"{OUTPUT_DIR}/data"   # directory for data files

# Plot customization
PLOT_DPI = 300         # DPI for saved plots
PLOT_STYLE = 'default' # matplotlib style
PLOT_FIGSIZE = (10, 6) # default figure size
PLOT_COLORS = {
  'primary': '#1f77b4',    # blue
  'secondary': '#ff7f0e',  # orange
  'tertiary': '#2ca02c',   # green
  'quaternary': '#d62728', # red
  'grid': '#cccccc'        # light gray
}

# Debug parameters
DEBUG_MODE = True      # enable/disable debug mode
VERBOSE = True         # enable/disable verbose output

def create_directories() -> None:
  """
  Create necessary directories for output files.
  
  Raises:
      ConfigurationError: If directory creation fails
  """
  try:
      # Create main output directory if it doesn't exist
      if not os.path.exists(OUTPUT_DIR):
          os.makedirs(OUTPUT_DIR)
      
      # Create subdirectories for plots and data
      if not os.path.exists(PLOT_DIR):
          os.makedirs(PLOT_DIR)
      if not os.path.exists(DATA_DIR):
          os.makedirs(DATA_DIR)
  except Exception as e:
      raise ConfigurationError(f"Failed to create directories: {str(e)}")

def get_optimization_bounds() -> Dict[str, tuple]:
  """
  Return bounds for optimization parameters.
  
  Returns:
      Dict[str, tuple]: Dictionary containing bounds for CL and thrust
  """
  return {
      'CL': (CL_MIN, CL_MAX),
      'thrust': (THRUST_MIN, THRUST_MAX)
  }

def get_chord_range() -> np.ndarray:
  """
  Return range of chord values for sweep analysis.
  
  Returns:
      np.ndarray: Array of chord values to analyze
  """
  if CHORD_SWEEP:
      return np.arange(MIN_CHORD, MAX_CHORD + CHORD_STEP, CHORD_STEP)
  return np.array([MIN_CHORD])  # Return single value if sweep disabled

def get_plot_settings() -> Dict[str, Any]:
  """
  Return dictionary of plot settings.
  
  Returns:
      Dict[str, Any]: Dictionary containing plot configuration
  """
  return {
      'dpi': PLOT_DPI,
      'style': PLOT_STYLE,
      'figsize': PLOT_FIGSIZE,
      'colors': PLOT_COLORS
  }

def validate_config() -> bool:
  """
  Validate configuration parameters.
  
  Returns:
      bool: True if configuration is valid
  
  Raises:
      ConfigurationError: If any configuration parameter is invalid
  """
  try:
      if WINGSPAN <= 0:
          raise ConfigurationError("Wingspan must be positive")
      if WEIGHT <= 0:
          raise ConfigurationError("Weight must be positive")
      if EPSILON <= 0 or EPSILON > 1:
          raise ConfigurationError("Oswald efficiency must be between 0 and 1")
      if TARGET_TAKEOFF_DISTANCE <= 0:
          raise ConfigurationError("Target takeoff distance must be positive")
      if MIN_CHORD >= MAX_CHORD:
          raise ConfigurationError("MIN_CHORD must be less than MAX_CHORD")
      if CHORD_STEP <= 0:
          raise ConfigurationError("CHORD_STEP must be positive")
      if WATT_LIMIT <= 0:
          raise ConfigurationError("Power limit must be positive")
      return True
  except Exception as e:
      raise ConfigurationError(f"Configuration validation failed: {str(e)}")

def print_config() -> None:
  """Print current configuration settings."""
  if not VERBOSE:
      return
      
  print("\nCurrent Configuration:")
  print("-" * 50)
  print(f"Aircraft Parameters:")
  print(f"  Wingspan: {WINGSPAN} ft")
  print(f"  Weight: {WEIGHT} lbs")
  print(f"  Oswald Efficiency: {EPSILON}")
  
  print("\nSimulation Parameters:")
  print(f"  Time Step: {DT} s")
  print(f"  Max Time: {T_MAX} s")
  print(f"  Power Limit: {WATT_LIMIT} W")
  
  print("\nOptimization Parameters:")
  print(f"  CL Range: [{CL_MIN}, {CL_MAX}]")
  print(f"  Thrust Range: [{THRUST_MIN}, {THRUST_MAX}] lbf")
  
  if CHORD_SWEEP:
      print("\nChord Sweep Parameters:")
      print(f"  Range: {MIN_CHORD} to {MAX_CHORD} ft")
      print(f"  Step Size: {CHORD_STEP} ft")
  
  print("\nOutput Settings:")
  print(f"  Save Plots: {SAVE_PLOTS}")
  print(f"  Save Data: {SAVE_DATA}")
  print(f"  Output Directory: {OUTPUT_DIR}")
  print("-" * 50)

def initialize() -> None:
  """
  Initialize configuration and create necessary directories.
  
  Raises:
      ConfigurationError: If initialization fails
  """
  try:
      validate_config()
      create_directories()
      if DEBUG_MODE:
          print_config()
  except Exception as e:
      raise ConfigurationError(f"Initialization failed: {str(e)}")