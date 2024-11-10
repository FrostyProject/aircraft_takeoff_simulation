#physics.py 
import numpy as np
import config

def calculate_drag_coefficient(CL, AR, epsilon, CD0=0.02):
  """Calculate drag coefficient using drag polar."""
  return CD0 + (CL**2)/(np.pi * AR * epsilon)

def check_lift_balance(CL, velocity, S, rho, weight):
  """Check if lift force is sufficient for takeoff."""
  lift = 0.5 * rho * velocity**2 * S * CL
  lift_ratio = lift/weight
  return lift, lift_ratio, lift_ratio >= 1.0

def calculate_takeoff_distance(CL, thrust, m, S, rho, sigma, mu, g, dt, t_max, 
                           power_limit, target_distance, AR, epsilon, weight):
  """
  Calculate takeoff distance with given parameters.
  Returns time history of distance and velocity if successful, None if failed.
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
      
      # Update velocity and position
      v += a * dt
      x += v * dt
      
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