"""
Aircraft Takeoff Simulation - Version 8.2 (V8.2)
This program simulates aircraft takeoff performance considering:
- Power and thrust limitations
- Aerodynamic characteristics
- Ground effects
- Motor RPM limitations
- Propeller characteristics

The simulation optimizes for minimum takeoff distance while respecting all constraints.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# ============================================================================
# Utility Functions
# ============================================================================

def watts_to_hp(watts):
  """
  Convert watts to horsepower.
  Args:
      watts (float): Power in watts
  Returns:
      float: Power in horsepower
  """
  return watts / 745.7

def calculate_CD_i(CL, AR, epsilon):
  """
  Calculate induced drag coefficient using lifting-line theory.
  Args:
      CL (float): Lift coefficient
      AR (float): Wing aspect ratio
      epsilon (float): Oswald efficiency factor
  Returns:
      float: Induced drag coefficient
  """
  return (CL**2) / (np.pi * AR * epsilon)

def calculate_rpm(thrust, velocity, prop_diameter, prop_pitch):
  """
  Calculate required RPM for given thrust using dynamic thrust equation:
  T = 4.392399e-8 * RPM * ((d^3.5)/sqrt(pitch)) * (4.23333e-4 * RPM * pitch - V)
  
  Args:
      thrust (float): Required thrust in lbf
      velocity (float): Current velocity in m/s
      prop_diameter (float): Propeller diameter in inches
      prop_pitch (float): Propeller pitch in inches
  Returns:
      float: Required RPM
  """
  # Convert thrust from lbf to N
  thrust_N = thrust * 4.44822
  
  # Calculate coefficients for quadratic equation
  A = 4.392399e-8 * ((prop_diameter**3.5)/np.sqrt(prop_pitch)) * 4.23333e-4 * prop_pitch
  B = -4.392399e-8 * ((prop_diameter**3.5)/np.sqrt(prop_pitch)) * velocity
  C = -thrust_N
  
  # Solve quadratic equation: A*RPM^2 + B*RPM + C = 0
  discriminant = B**2 - 4*A*C
  if discriminant < 0:
      return float('nan')
  
  # Return positive solution only
  rpm = (-B + np.sqrt(discriminant))/(2*A)
  return rpm

def calculate_static_thrust(power, sigma):
  """
  Calculate static thrust based on power and atmospheric density ratio.
  Args:
      power (float): Available power in watts
      sigma (float): Atmospheric density ratio
  Returns:
      float: Static thrust in lbf
  """
  static_T = power*(1.132*sigma-0.132)
  return static_T

# ============================================================================
# Core Simulation Functions
# ============================================================================

def calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon):
  """
  Calculate takeoff distance and related parameters using numerical integration.
  
  Args:
      params (list): [CL_max, T] - Maximum lift coefficient and thrust
      m (float): Aircraft mass in slugs
      S (float): Wing area in ft²
      rho (float): Air density in slugs/ft³
      sigma (float): Atmospheric density ratio
      mu (float): Ground friction coefficient
      g (float): Gravitational acceleration in ft/s²
      dt (float): Time step for integration
      t_max (float): Maximum simulation time
      power_limit (float): Maximum available power in watts
      AR (float): Wing aspect ratio
      epsilon (float): Oswald efficiency factor
  
  Returns:
      tuple: (takeoff_distance, takeoff_velocity, takeoff_time)
  """
  CL_max, T = params
  
  # Initialize arrays for velocity and distance
  V = np.zeros(int(t_max/dt))
  s = np.zeros(int(t_max/dt))
  
  # Calculate takeoff velocity (with 10% safety margin)
  V_TO = np.sqrt((2 * m * g) / (rho * S * CL_max)) * 1.1
  
  # Numerical integration loop
  for i in range(1, len(V)):
      # Calculate lift coefficient (limited by CL_max)
      CL = min(CL_max, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      
      # Calculate drag coefficients
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = CD0 + CD_i
      
      # Calculate aerodynamic forces
      D = 0.5 * rho * V[i-1]**2 * S * CD  # Drag force
      L = 0.5 * rho * V[i-1]**2 * S * CL  # Lift force
      Fr = mu * (m * g - L)  # Rolling friction
      
      # Calculate thrust and net force
      T_current = min(T, calculate_static_thrust(power_limit, sigma))
      F_net = T_current - D - Fr
      
      # Update velocity and position
      a = F_net / m
      V[i] = V[i-1] + a * dt
      s[i] = s[i-1] + V[i] * dt
      
      # Check if takeoff velocity is reached
      if V[i] >= V_TO:
          return s[i], V[i], i * dt
  
  # Return final values if takeoff speed not reached
  return s[-1], V[-1], t_max

def objective(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_distance, AR, epsilon):
  """
  Objective function for optimization. Calculates normalized error between
  achieved and target takeoff distance.
  """
  distance, _, _ = calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon)
  return abs(distance - target_distance) / target_distance

def optimize_CL_and_T():
  """
  Optimize CL_max and thrust to achieve target takeoff distance.
  Uses alternating optimization strategy for CL and T.
  """
  # Initialize optimization parameters
  best_CL = 1.0  # Initial guess for CL_max
  best_T = 0.8 * weight  # Initial thrust guess (80% of weight)
  best_error = float('inf')
  
  # Setup progress bar
  pbar = tqdm(total=500, desc="Optimizing", ncols=70)
  
  # Main optimization loop
  for i in range(500):
      # Optimize thrust for current CL
      res_T = minimize_scalar(
          lambda T: objective([best_CL, T], m, S, rho, sigma, mu, g, dt, t_max, 
                            power_limit, target_takeoff_distance, AR, epsilon),
          bounds=(0.1 * weight, weight),
          method='bounded'
      )
      
      # Optimize CL for new thrust
      res_CL = minimize_scalar(
          lambda CL: objective([CL, res_T.x], m, S, rho, sigma, mu, g, dt, t_max,
                             power_limit, target_takeoff_distance, AR, epsilon),
          bounds=(1.0, 3.0),
          method='bounded'
      )
      
      # Calculate error for new parameters
      error = objective([res_CL.x, res_T.x], m, S, rho, sigma, mu, g, dt, t_max,
                       power_limit, target_takeoff_distance, AR, epsilon)
      
      # Update best values if better solution found
      if error < best_error:
          best_CL = res_CL.x
          best_T = res_T.x
          best_error = error
      
      # Update progress bar
      pbar.update(1)
      pbar.set_postfix({'error': f'{best_error:.4f}'})
      
      # Check for convergence
      if best_error <= 0.001:
          break
  
  pbar.close()
  return best_CL, best_T
# ============================================================================
# Main Execution Code
# ============================================================================

if __name__ == "__main__":
  # Input Collection
  print("\n=== Aircraft Configuration Input ===")
  watt_limit = float(input("Enter electric motor watt limit: "))
  wingspan = float(input("Enter wingspan (ft): "))
  chord = float(input("Enter chord (ft): "))
  target_takeoff_distance = float(input("Enter target takeoff distance (ft): "))
  weight = float(input("Enter aircraft weight (lbs): "))
  sigma = float(input("Enter atmospheric density ratio (sigma): "))
  prop_diameter = float(input("Enter propeller diameter (inches): "))
  prop_pitch = float(input("Enter propeller pitch (inches): "))
  max_rpm = float(input("Enter maximum motor RPM: "))

  # Set power limit
  power_limit = watt_limit

  # Calculate aircraft geometry parameters
  S = wingspan * chord  # Wing area (ft²)
  AR = wingspan**2 / S  # Aspect ratio

  # Define physical constants
  g = 32.174  # Gravitational acceleration (ft/s²)
  m = weight / g  # Mass (slugs)
  mu = 0.02  # Ground friction coefficient
  rho = 0.00237  # Sea level air density (slugs/ft³)
  epsilon = 0.7  # Oswald efficiency factor
  CD0 = 0.02  # Zero-lift drag coefficient

  # Simulation parameters
  dt = 0.1  # Time step (seconds)
  t_max = 120  # Maximum simulation time (seconds)

  # ============================================================================
  # Optimization and Initial Calculations
  # ============================================================================

  # Run optimization to find best CL and thrust
  print("\nOptimizing takeoff parameters...")
  best_CL, best_T = optimize_CL_and_T()

  # Calculate initial takeoff parameters
  final_distance, V_TO, takeoff_time = calculate_takeoff_distance(
      [best_CL, best_T], m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon
  )

  # ============================================================================
  # Data Generation for Analysis
  # ============================================================================

  # Initialize arrays for time-history data
  t = np.arange(0, takeoff_time, dt)
  V = np.zeros_like(t)  # Velocity
  s = np.zeros_like(t)  # Distance
  a = np.zeros_like(t)  # Acceleration
  T = np.zeros_like(t)  # Thrust
  rpm = np.zeros_like(t)  # RPM

  # Initialize RPM analysis variables
  rpm_warning = False
  rpm_limited_thrust = False
  thrust_reduction = []

  # Main simulation loop
  for i in range(1, len(t)):
      # Calculate aerodynamic coefficients
      CL = min(best_CL, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = CD0 + CD_i
      
      # Calculate forces
      D = 0.5 * rho * V[i-1]**2 * S * CD  # Drag
      L = 0.5 * rho * V[i-1]**2 * S * CL  # Lift
      Fr = mu * (m * g - L)  # Rolling friction
      
      # Calculate thrust and RPM
      T[i] = min(best_T, calculate_static_thrust(power_limit, sigma))
      required_rpm = calculate_rpm(T[i], V[i-1] * 0.3048, prop_diameter, prop_pitch)
      
      # Check RPM limits and adjust thrust if necessary
      if required_rpm > max_rpm:
          rpm_warning = True
          rpm[i] = max_rpm
          # Recalculate thrust with RPM limit
          velocity_ms = V[i-1] * 0.3048
          actual_thrust_N = 4.392399e-8 * max_rpm * ((prop_diameter**3.5)/np.sqrt(prop_pitch)) * \
                           (4.23333e-4 * max_rpm * prop_pitch - velocity_ms)
          T[i] = actual_thrust_N / 4.44822
          thrust_reduction.append((i * dt, (1 - T[i]/best_T) * 100))
          rpm_limited_thrust = True
      else:
          rpm[i] = required_rpm
      
      # Calculate motion
      F_net = T[i] - D - Fr
      a[i] = F_net / m
      V[i] = V[i-1] + a[i] * dt
      s[i] = s[i-1] + V[i] * dt

  # ============================================================================
  # Results Display
  # ============================================================================

  print("\n=== OPTIMAL PERFORMANCE (No RPM Limit) ===")
  print(f"Optimal CL_max: {best_CL:.4f}")
  print(f"Optimal Thrust: {best_T:.2f} lbf")
  print(f"Fixed CD0: {CD0:.4f}")
  print(f"Optimal Takeoff Distance: {final_distance:.2f} ft")
  print(f"Target Takeoff Distance: {target_takeoff_distance:.2f} ft")
  print(f"Optimal Difference: {abs(final_distance - target_takeoff_distance):.2f} ft "
        f"({abs(final_distance - target_takeoff_distance) / target_takeoff_distance * 100:.2f}%)")
  print(f"Optimal Thrust as percentage of aircraft weight: {(best_T / weight) * 100:.2f}%")
  print(f"Optimal Time to reach takeoff velocity: {takeoff_time:.2f} seconds")
  print(f"Takeoff Velocity: {V_TO:.2f} ft/s")

  print(f"\n=== RPM ANALYSIS ===")
  print(f"Maximum Motor RPM: {max_rpm:.0f}")
  print(f"Peak Required RPM: {np.nanmax(rpm):.0f}")

  # Display RPM-limited performance if applicable
  if rpm_warning:
      print("\n=== ACTUAL PERFORMANCE (RPM Limited) ===")
      # Calculate actual performance metrics
      actual_takeoff_distance = s[-1]
      actual_takeoff_time = t[-1]
      actual_takeoff_velocity = V[-1]
      average_thrust = np.mean(T[1:])
      min_thrust = np.min(T[1:])
      
      print("\nWARNING: Required RPM exceeds maximum motor RPM!")
      print("Performance is limited due to RPM constraints.")
      print(f"\nActual Takeoff Distance: {actual_takeoff_distance:.2f} ft")
      print(f"Distance Increase: {(actual_takeoff_distance - final_distance):.2f} ft "
            f"({((actual_takeoff_distance/final_distance) - 1) * 100:.1f}%)")
      print(f"Actual Takeoff Time: {actual_takeoff_time:.2f} seconds")
      print(f"Time Increase: {(actual_takeoff_time - takeoff_time):.2f} seconds "
            f"({((actual_takeoff_time/takeoff_time) - 1) * 100:.1f}%)")
      
      if thrust_reduction:
          max_thrust_reduction = max(reduction[1] for reduction in thrust_reduction)
          first_occurrence = thrust_reduction[0][0]
          print(f"\nThrust Analysis:")
          print(f"RPM limiting first occurred at: {first_occurrence:.1f} seconds")
          print(f"Maximum thrust reduction: {max_thrust_reduction:.1f}%")
          print(f"Average thrust during takeoff: {average_thrust:.2f} lbf")
          print(f"Minimum thrust during takeoff: {min_thrust:.2f} lbf")
          
      print("\nPerformance Impact Summary:")
      print(f"- Takeoff distance increased by {((actual_takeoff_distance/final_distance) - 1) * 100:.1f}%")
      print(f"- Takeoff time increased by {((actual_takeoff_time/takeoff_time) - 1) * 100:.1f}%")
      print(f"- Maximum thrust reduced by {max_thrust_reduction:.1f}%")
  else:
      print("\nMotor RPM is sufficient for optimal performance.")
      print("No performance reduction due to RPM limitations.")

  # ============================================================================
  # Visualization
  # ============================================================================

  plt.figure(figsize=(12, 12))

  # Velocity plot
  plt.subplot(5, 1, 1)
  plt.plot(t, V)
  plt.ylabel('Velocity (ft/s)')
  plt.title('Aircraft Takeoff Simulation (Power Limited)')
  plt.axhline(y=V_TO, color='r', linestyle='--', label='Takeoff Velocity')
  plt.legend()

  # Distance plot
  plt.subplot(5, 1, 2)
  plt.plot(t, s)
  plt.ylabel('Distance (ft)')
  plt.axhline(y=final_distance, color='r', linestyle='--', label='Takeoff Distance')
  plt.legend()

  # Acceleration plot
  plt.subplot(5, 1, 3)
  plt.plot(t, a)
  plt.ylabel('Acceleration (ft/s²)')

  # Thrust plot
  plt.subplot(5, 1, 4)
  plt.plot(t, T)
  plt.ylabel('Thrust (lbf)')

  # RPM plot
  plt.subplot(5, 1, 5)
  plt.plot(t, rpm)
  plt.axhline(y=max_rpm, color='r', linestyle='--', label='Max RPM')
  plt.ylabel('RPM')
  plt.xlabel('Time (s)')
  plt.legend()

  plt.tight_layout()
  plt.show()

  # File creation notification
  print("\nNo files were created or modified during the execution of this script.")