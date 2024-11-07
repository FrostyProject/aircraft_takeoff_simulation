import numpy as np
from config import GRAVITY, RHO_SL, DEFAULT_CD0

def calculate_CD_i(CL, AR, epsilon):
  """Calculate induced drag coefficient."""
  return CL**2 / (np.pi * AR * epsilon)

def calculate_static_thrust(power_limit, sigma):
  """Calculate static thrust from power limit."""
  return power_limit * sigma

def check_lift_balance(CL, velocity, S, rho, weight):
  """
  Check if lift force equals or exceeds aircraft weight.
  
  Args:
      CL (float): Lift coefficient
      velocity (float): Aircraft velocity (ft/s)
      S (float): Wing area (ft²)
      rho (float): Air density (slugs/ft³)
      weight (float): Aircraft weight (lbf)
  
  Returns:
      tuple: (is_balanced (bool), lift_ratio (float), lift_force (float))
  """
  lift_force = 0.5 * rho * velocity**2 * S * CL
  lift_ratio = lift_force / weight
  is_balanced = abs(lift_ratio - 1.0) < 0.01  # Within 1% tolerance
  
  return is_balanced, lift_ratio, lift_force

def calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon):
  """Calculate takeoff distance and related parameters."""
  CL_max, T = params
  V = np.zeros(int(t_max/dt))
  s = np.zeros(int(t_max/dt))
  weight = m * g
  
  # Calculate rotation speed (when lift equals weight)
  V_R = np.sqrt((2 * weight) / (rho * S * CL_max))
  # Takeoff speed with 1.1 safety factor
  V_TO = V_R * 1.1
  
  lift_achieved = False
  rotation_point = None
  
  for i in range(1, len(V)):
      # Calculate current CL (limited by CL_max)
      CL = min(CL_max, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = DEFAULT_CD0 + CD_i
      
      # Calculate forces
      D = 0.5 * rho * V[i-1]**2 * S * CD
      L = 0.5 * rho * V[i-1]**2 * S * CL
      Fr = mu * (weight - L)
      
      # Check lift balance at each step
      is_balanced, lift_ratio, _ = check_lift_balance(CL, V[i-1], S, rho, weight)
      
      if not lift_achieved and lift_ratio >= 1.0:
          lift_achieved = True
          rotation_point = s[i-1]
      
      T_current = min(T, calculate_static_thrust(power_limit, sigma))
      F_net = T_current - D - Fr
      a = F_net / m
      V[i] = V[i-1] + a * dt
      s[i] = s[i-1] + V[i] * dt
      
      if V[i] >= V_TO:
          return s[i], V[i], i * dt, rotation_point
  
  return s[-1], V[-1], t_max, rotation_point