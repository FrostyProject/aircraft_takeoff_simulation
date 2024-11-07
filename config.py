"""Configuration and constants for aircraft takeoff analysis."""

# Physical constants
GRAVITY = 32.174  # Gravitational acceleration (ft/s²)
RHO_SL = 0.00237  # Sea level air density (slugs/ft³)

# Simulation defaults
DEFAULT_MU = 0.02  # Ground friction coefficient
DEFAULT_EPSILON = 0.7  # Oswald efficiency factor
DEFAULT_CD0 = 0.02  # Zero-lift drag coefficient
DEFAULT_DT = 0.1  # Time step (seconds)
DEFAULT_T_MAX = 120  # Maximum simulation time (seconds)

# Optimization parameters
CL_BOUNDS = (0.1, 5.0)  # CL optimization bounds
THRUST_WEIGHT_RATIO_BOUNDS = (0.1, 2)  # Thrust/Weight ratio bounds
MAX_ITERATIONS = 500  # Maximum optimization iterations
CONVERGENCE_THRESHOLD = 0.0001  # Error threshold for convergence