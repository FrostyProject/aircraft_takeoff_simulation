"""Physics calculations and core functions."""

import numpy as np
from config import GRAVITY, RHO_SL, DEFAULT_CD0  # Add DEFAULT_CD0 to imports

def calculate_CD_i(CL, AR, epsilon):
  """Calculate induced drag coefficient using lifting-line theory."""
  return (CL**2) / (np.pi * AR * epsilon)

def calculate_rpm(thrust, velocity, prop_diameter, prop_pitch):
  """Calculate required RPM for given thrust using dynamic thrust equation."""
  thrust_N = thrust * 4.44822
  A = 4.392399e-8 * ((prop_diameter**3.5)/np.sqrt(prop_pitch)) * 4.23333e-4 * prop_pitch
  B = -4.392399e-8 * ((prop_diameter**3.5)/np.sqrt(prop_pitch)) * velocity
  C = -thrust_N
  discriminant = B**2 - 4*A*C
  if discriminant < 0:
      return float('nan')
  rpm = (-B + np.sqrt(discriminant))/(2*A)
  return rpm

def calculate_static_thrust(power, sigma):
  """Calculate static thrust based on power and atmospheric density ratio."""
  return power * (1.132 * sigma - 0.132)

def calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon):
  """Calculate takeoff distance and related parameters."""
  CL_max, T = params
  V = np.zeros(int(t_max/dt))
  s = np.zeros(int(t_max/dt))
  V_TO = np.sqrt((2 * m * g) / (rho * S * CL_max)) * 1.1
  
  for i in range(1, len(V)):
      CL = min(CL_max, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = DEFAULT_CD0 + CD_i  # Now using imported DEFAULT_CD0
      
      D = 0.5 * rho * V[i-1]**2 * S * CD
      L = 0.5 * rho * V[i-1]**2 * S * CL
      Fr = mu * (m * g - L)
      
      T_current = min(T, calculate_static_thrust(power_limit, sigma))
      F_net = T_current - D - Fr
      a = F_net / m
      V[i] = V[i-1] + a * dt
      s[i] = s[i-1] + V[i] * dt
      
      if V[i] >= V_TO:
          return s[i], V[i], i * dt
  
  return s[-1], V[-1], t_max