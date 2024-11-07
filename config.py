# Physical constants
GRAVITY = 32.174  # ft/s²
RHO_SL = 0.00237  # Sea level air density (slugs/ft³)

# Aerodynamic parameters
DEFAULT_CD0 = 0.02  # Zero-lift drag coefficient
OSWALD_EFFICIENCY = 0.7  # Oswald efficiency factor
GROUND_FRICTION = 0.02  # Ground friction coefficient

# Optimization bounds and parameters
CL_BOUNDS = (0.8, 2.5)
THRUST_WEIGHT_RATIO_BOUNDS = (0.2, 1.2)
MAX_ITERATIONS = 100
CONVERGENCE_THRESHOLD = 0.01

# Simulation parameters
TIME_STEP = 0.1  # seconds
MAX_SIM_TIME = 120  # seconds

def get_default_params():
  """Return dictionary of default parameters."""
  return {
      'watt_limit': None,
      'wingspan': None,
      'chord': None,
      'weight': None,
      'target_takeoff_distance': None,
      'sigma': 1.0,
      'prop_diameter': None,
      'prop_pitch': None,
      'max_rpm': None,
      'dt': TIME_STEP,
      't_max': MAX_SIM_TIME,
      'mu': GROUND_FRICTION,
      'epsilon': OSWALD_EFFICIENCY,
      'CD0': DEFAULT_CD0
  }