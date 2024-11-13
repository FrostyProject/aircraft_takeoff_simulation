#physics.py Version 1.1
import numpy as np
from typing import Dict, Optional, Tuple, Union
import config

def calculate_drag_coefficient(CL: float, AR: float, epsilon: float, CD0: float = 0.02) -> float:
  """
  Calculate drag coefficient using drag polar equation.
  
  Args:
      CL (float): Lift coefficient
      AR (float): Wing aspect ratio
      epsilon (float): Oswald efficiency factor
      CD0 (float, optional): Zero-lift drag coefficient. Defaults to 0.02.
  
  Returns:
      float: Total drag coefficient
  """
  return CD0 + (CL**2)/(np.pi * AR * epsilon)

def check_lift_balance(CL: float, velocity: float, S: float, rho: float, 
                    weight: float) -> Tuple[float, float, bool]:
  """
  Check if lift force is sufficient for takeoff.
  
  Args:
      CL (float): Lift coefficient
      velocity (float): Current velocity in ft/s
      S (float): Wing area in sq ft
      rho (float): Air density in slug/ft^3
      weight (float): Aircraft weight in lbf
  
  Returns:
      tuple: (lift, lift_ratio, is_airborne)
          - lift (float): Calculated lift force
          - lift_ratio (float): Ratio of lift to weight
          - is_airborne (bool): True if lift exceeds weight
  """
  lift = 0.5 * rho * velocity**2 * S * CL
  lift_ratio = lift/weight
  return lift, lift_ratio, lift_ratio >= 1.0

def calculate_takeoff_distance(
  CL: float,
  thrust: float,
  m: float,
  S: float,
  rho: float,
  sigma: float,
  mu: float,
  g: float,
  dt: float,
  t_max: float,
  power_limit: float,
  target_distance: float,
  AR: float,
  epsilon: float,
  weight: float
) -> Optional[Dict[str, np.ndarray]]:
  """
  Calculate takeoff distance and trajectory with given parameters.
  
  Args:
      CL (float): Lift coefficient
      thrust (float): Thrust force in lbf
      m (float): Aircraft mass in slugs
      S (float): Wing area in sq ft
      rho (float): Sea level air density in slug/ft^3
      sigma (float): Density ratio
      mu (float): Ground friction coefficient
      g (float): Gravitational acceleration ft/s^2
      dt (float): Time step for simulation
      t_max (float): Maximum simulation time
      power_limit (float): Maximum power in watts
      target_distance (float): Target takeoff distance in ft
      AR (float): Wing aspect ratio
      epsilon (float): Oswald efficiency factor
      weight (float): Aircraft weight in lbf
  
  Returns:
      Optional[Dict[str, np.ndarray]]: Dictionary containing time histories
          - 'time': Array of time points
          - 'distance': Array of distances
          - 'velocity': Array of velocities
          Returns None if takeoff is not achieved or constraints are violated
  """
  # Initialize arrays for time history
  time_points = np.arange(0, t_max + dt, dt)
  distances = np.zeros_like(time_points)
  velocities = np.zeros_like(time_points)
  
  # Initial conditions
  v = 0.0  # initial velocity
  x = 0.0  # initial position
  
  # Calculate CD
  CD = calculate_drag_coefficient(CL, AR, epsilon)
  
  for i, t in enumerate(time_points):
      # Record current state
      distances[i] = x
      velocities[i] = v
      
      # Calculate forces
      q = 0.5 * rho * sigma * v**2  # dynamic pressure
      lift = q * S * CL
      drag = q * S * CD
      friction = mu * (weight - lift)  # friction force
      
      # Check if power limit is exceeded
      power = thrust * v
      if power > power_limit:
          return None
      
      # Net force and acceleration
      F_net = thrust - drag - friction
      a = F_net / m
      
      # Update velocity and position using semi-implicit Euler method
      v_next = v + a * dt
      x_next = x + 0.5 * (v + v_next) * dt
      
      v = v_next
      x = x_next
      
      # Check for takeoff
      lift, lift_ratio, is_airborne = check_lift_balance(CL, v, S, rho * sigma, weight)
      if is_airborne:
          return {
              'time': time_points[:i+1],
              'distance': distances[:i+1],
              'velocity': velocities[:i+1]
          }
      
      # Check if we've exceeded the target distance without takeoff
      if x > target_distance * 1.5:
          return None
  
  # If we get here, we didn't achieve takeoff
  return None

def validate_inputs(
  CL: float,
  thrust: float,
  m: float,
  S: float,
  rho: float,
  sigma: float,
  mu: float,
  g: float,
  dt: float,
  t_max: float
) -> bool:
  """
  Validate input parameters for physical correctness.
  
  Args:
      Various physical parameters (see above function for descriptions)
  
  Returns:
      bool: True if all inputs are valid, False otherwise
  
  Raises:
      ValueError: If any input parameter is invalid
  """
  if CL <= 0:
      raise ValueError("Lift coefficient must be positive")
  if thrust < 0:
      raise ValueError("Thrust cannot be negative")
  if m <= 0:
      raise ValueError("Mass must be positive")
  if S <= 0:
      raise ValueError("Wing area must be positive")
  if rho <= 0:
      raise ValueError("Air density must be positive")
  if sigma <= 0:
      raise ValueError("Density ratio must be positive")
  if mu < 0:
      raise ValueError("Friction coefficient cannot be negative")
  if g <= 0:
      raise ValueError("Gravitational acceleration must be positive")
  if dt <= 0:
      raise ValueError("Time step must be positive")
  if t_max <= 0:
      raise ValueError("Maximum time must be positive")
  
  return True