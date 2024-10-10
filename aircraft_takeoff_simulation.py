# Aircraft Takeoff Simulation - Version 6 (V6)

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from tqdm import tqdm

# Function to convert watts to horsepower
def watts_to_hp(watts):
  return watts / 745.7

# Function to calculate CD_i
def calculate_CD_i(CL, AR, epsilon):
  return (CL**2) / (np.pi * AR * epsilon)

# Function to calculate thrust based on power and velocity
def calculate_thrust(power, velocity):
  if velocity < 1e-6:
      return power * 2  # Assume static thrust is twice the power
  return min(power / velocity, power * 2)  # Limit static thrust

# Function to calculate takeoff distance
def calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon, P):
  CL_max, T = params
  
  V = np.zeros(int(t_max/dt))
  s = np.zeros(int(t_max/dt))
  
  V_TO = np.sqrt((2 * m * g) / (rho * S * CL_max)) * 1.1
  
  for i in range(1, len(V)):
      CL = min(CL_max, (2 * m * g) / (rho * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = CD0 + CD_i
      D = 0.5 * rho * V[i-1]**2 * S * CD
      L = 0.5 * rho * V[i-1]**2 * S * CL
      Fr = mu * (m * g - L)
      
      T_current = min(T, calculate_thrust(power_limit, V[i-1]))
      F_net = T_current - D - Fr
      a = F_net / m
      V[i] = V[i-1] + a * dt
      s[i] = s[i-1] + V[i] * dt
      
      if V[i] >= V_TO:
          return s[i], V[i], i * dt
  
  return s[-1], V[-1], t_max

# Objective function for optimization
def objective(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_distance, AR, epsilon, P):
  distance, _, _ = calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon, P)
  return abs(distance - target_distance) / target_distance

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
S = wingspan * chord
AR = wingspan**2 / S

# Constants
g = 32.174  # Acceleration due to gravity (ft/s^2)
m = weight / g  # Mass of the aircraft (slugs)
mu = 0.02  # Coefficient of rolling friction
rho = 0.00237  # Sea level air density (slugs/ft^3)
epsilon = 0.7  # Oswald efficiency factor
CD0 = 0.02  # Fixed parasitic drag coefficient

# Simulation parameters
dt = 0.1  # Time step (s)
t_max = 120  # Maximum simulation time (s)

# Optimization
def optimize_CL_and_T():
  best_CL = 1.0
  best_T = 0.8 * weight
  best_error = float('inf')
  
  pbar = tqdm(total=500, desc="Optimizing", ncols=70)
  
  for i in range(500):  # 500 iterations
      # Optimize T for current CL
      res_T = minimize_scalar(
          lambda T: objective([best_CL, T], m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_takeoff_distance, AR, epsilon, P),
          bounds=(0.1 * weight, weight),
          method='bounded'
      )
      
      # Optimize CL for new T
      res_CL = minimize_scalar(
          lambda CL: objective([CL, res_T.x], m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_takeoff_distance, AR, epsilon, P),
          bounds=(1.0, 3.0),
          method='bounded'
      )
      
      error = objective([res_CL.x, res_T.x], m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_takeoff_distance, AR, epsilon, P)
      
      if error < best_error:
          best_CL = res_CL.x
          best_T = res_T.x
          best_error = error
      
      pbar.update(1)
      pbar.set_postfix({'error': f'{best_error:.4f}'})
      
      if best_error <= 0.001:  # 0.1% error
          break
  
  pbar.close()
  return best_CL, best_T

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