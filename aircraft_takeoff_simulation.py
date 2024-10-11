# Aircraft Takeoff Simulation - Version 7 (V7)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# Function to convert watts to horsepower
def watts_to_hp(watts):
  return watts / 745.7  # Equation: HP = Watts / 745.7

# Function to calculate induced drag coefficient (CD_i)
def calculate_CD_i(CL, AR, epsilon):
  # Equation: CD_i = CL^2 / (π * AR * e)
  # Where:
  # CL = Lift coefficient
  # AR = Aspect ratio
  # e = Oswald efficiency factor
  return (CL**2) / (np.pi * AR * epsilon)

# Function to calculate thrust based on power and velocity
def calculate_thrust(power, velocity):
  if velocity < 1e-6:
      return power * 2  # Static thrust assumption: T = 2 * Power
  # Equation: T = min(Power / Velocity, 2 * Power)
  return min(power / velocity, power * 2)

# Function to calculate takeoff distance
def calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon, P):
  CL_max, T = params
  
  V = np.zeros(int(t_max/dt))
  s = np.zeros(int(t_max/dt))
  
  # Equation: V_TO = sqrt((2 * m * g) / (ρ * S * CL_max)) * 1.1
  # 1.1 factor for safety margin
  V_TO = np.sqrt((2 * m * g) / (rho * S * CL_max)) * 1.1
  
  for i in range(1, len(V)):
      # Equation: CL = min(CL_max, (2 * m * g) / (ρ * V^2 * S))
      CL = min(CL_max, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = CD0 + CD_i  # Total drag coefficient
      
      # Equation: D = 0.5 * ρ * V^2 * S * CD
      D = 0.5 * rho * V[i-1]**2 * S * CD  # Drag force
      # Equation: L = 0.5 * ρ * V^2 * S * CL
      L = 0.5 * rho * V[i-1]**2 * S * CL  # Lift force
      # Equation: Fr = μ * (m * g - L)
      Fr = mu * (m * g - L)  # Rolling friction force
      
      T_current = min(T, calculate_thrust(power_limit, V[i-1]))
      # Equation: F_net = T - D - Fr
      F_net = T_current - D - Fr  # Net force
      
      # Equation: a = F_net / m
      a = F_net / m  # Acceleration
      # Equation: V = V_previous + a * dt
      V[i] = V[i-1] + a * dt  # Update velocity
      # Equation: s = s_previous + V * dt
      s[i] = s[i-1] + V[i] * dt  # Update distance
      
      if V[i] >= V_TO:
          return s[i], V[i], i * dt
  
  return s[-1], V[-1], t_max

# Objective function for optimization
def objective(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_distance, AR, epsilon, P):
  distance, _, _ = calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon, P)
  # Equation: error = |calculated_distance - target_distance| / target_distance
  return abs(distance - target_distance) / target_distance

# Optimization function
def optimize_CL_and_T():
  # Initialize best values
  best_CL = 1.0  # Start with a reasonable guess for max lift coefficient
  best_T = 0.8 * weight  # Initial thrust guess: 80% of aircraft weight
  best_error = float('inf')  # Initialize best error as infinity
  
  # Set up progress bar for 500 iterations
  pbar = tqdm(total=500, desc="Optimizing", ncols=70)
  
  # Start optimization loop
  for i in range(500):  # Run for a maximum of 500 iterations
      # Optimize thrust (T) for the current best lift coefficient (CL)
      res_T = minimize_scalar(
          # Anonymous function that calculates the objective for a given T
          lambda T: objective([best_CL, T], m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_takeoff_distance, AR, epsilon, P),
          bounds=(0.1 * weight, weight),  # Thrust bounds: 10% to 100% of weight
          method='bounded'  # Use bounded optimization method
      )
      
      # Optimize lift coefficient (CL) for the new best thrust
      res_CL = minimize_scalar(
          # Anonymous function that calculates the objective for a given CL
          lambda CL: objective([CL, res_T.x], m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_takeoff_distance, AR, epsilon, P),
          bounds=(1.0, 3.0),  # CL bounds: typical range for aircraft
          method='bounded'  # Use bounded optimization method
      )
      
      # Calculate the error for the new CL and T combination
      error = objective([res_CL.x, res_T.x], m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_takeoff_distance, AR, epsilon, P)
      
      # If this error is better than the best so far, update best values
      if error < best_error:
          best_CL = res_CL.x  # Update best CL
          best_T = res_T.x  # Update best T
          best_error = error  # Update best error
      
      # Update progress bar
      pbar.update(1)  # Increment progress bar by 1
      pbar.set_postfix({'error': f'{best_error:.4f}'})  # Display current best error
      
      # Check if the error is within acceptable tolerance
      if best_error <= 0.001:  # If error is less than 0.1%
          break  # Exit the loop early
  
  # Close the progress bar
  pbar.close()
  
  # Return the best found values for CL and T
  return best_CL, best_T

# Input parameters
watt_limit = float(input("Enter electric motor watt limit: "))
wingspan = float(input("Enter wingspan (ft): "))
chord = float(input("Enter chord (ft): "))
target_takeoff_distance = float(input("Enter target takeoff distance (ft): "))
weight = float(input("Enter aircraft weight (lbs): "))
sigma = float(input("Enter atmospheric density ratio (sigma): "))

# Convert watt limit to horsepower and set power limit
P = watts_to_hp(watt_limit)
power_limit = watt_limit

# Calculate wing area and aspect ratio
S = wingspan * chord  # Equation: S = wingspan * chord
AR = wingspan**2 / S  # Equation: AR = wingspan^2 / S

# Constants
g = 32.174  # Acceleration due to gravity (ft/s^2)
m = weight / g  # Equation: m = weight / g (mass in slugs)
mu = 0.02  # Coefficient of rolling friction
rho = 0.00237  # Sea level air density (slugs/ft^3)
epsilon = 0.7  # Oswald efficiency factor
CD0 = 0.02  # Fixed parasitic drag coefficient

# Simulation parameters
dt = 0.1  # Time step (s)
t_max = 120  # Maximum simulation time (s)

# Run optimization
best_CL, best_T = optimize_CL_and_T()

# Calculate final takeoff distance and time
final_distance, V_TO, takeoff_time = calculate_takeoff_distance([best_CL, best_T], m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon, P)

# Check if the solution is within 0.1% of the target
if abs(final_distance - target_takeoff_distance) / target_takeoff_distance <= 0.001:
  print("\nOptimal solution found within 0.1% of target takeoff distance.")
else:
  print("\nWarning: Optimal solution not within 0.1% of target takeoff distance.")

# Generate data for plotting
t = np.arange(0, takeoff_time, dt)
V = np.zeros_like(t)
s = np.zeros_like(t)
a = np.zeros_like(t)
T = np.zeros_like(t)

for i in range(1, len(t)):
  CL = min(best_CL, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
  CD_i = calculate_CD_i(CL, AR, epsilon)
  CD = CD0 + CD_i
  D = 0.5 * rho * V[i-1]**2 * S * CD
  L = 0.5 * rho * V[i-1]**2 * S * CL
  Fr = mu * (m * g - L)
  T[i] = min(best_T, calculate_thrust(power_limit, V[i-1]))
  F_net = T[i] - D - Fr
  a[i] = F_net / m
  V[i] = V[i-1] + a[i] * dt
  s[i] = s[i-1] + V[i] * dt

# Display results
print(f"\nOptimal CL_max: {best_CL:.4f}")
print(f"Optimal Thrust: {best_T:.2f} lbf")
print(f"Fixed CD0: {CD0:.4f}")
print(f"Takeoff Distance: {final_distance:.2f} ft")
print(f"Target Takeoff Distance: {target_takeoff_distance:.2f} ft")
print(f"Difference: {abs(final_distance - target_takeoff_distance):.2f} ft ({abs(final_distance - target_takeoff_distance) / target_takeoff_distance * 100:.2f}%)")
print(f"Thrust as percentage of aircraft weight: {(best_T / weight) * 100:.2f}%")
print(f"Time to reach takeoff velocity: {takeoff_time:.2f} seconds")
print(f"Takeoff Velocity: {V_TO:.2f} ft/s")

# Plot results
plt.figure(figsize=(12, 10))
plt.subplot(4, 1, 1)
plt.plot(t, V)
plt.ylabel('Velocity (ft/s)')
plt.title('Aircraft Takeoff Simulation (Power Limited)')
plt.axhline(y=V_TO, color='r', linestyle='--', label='Takeoff Velocity')
plt.legend()

plt.subplot(4, 1, 2)
plt.plot(t, s)
plt.ylabel('Distance (ft)')
plt.axhline(y=final_distance, color='r', linestyle='--', label='Takeoff Distance')
plt.legend()

plt.subplot(4, 1, 3)
plt.plot(t, a)
plt.ylabel('Acceleration (ft/s^2)')

plt.subplot(4, 1, 4)
plt.plot(t, T)
plt.ylabel('Thrust (lbf)')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# Created/Modified files during execution:
print("No files were created or modified during the execution of this script.")