"""Analysis types for aircraft takeoff simulation."""

import numpy as np
from physics import (calculate_takeoff_distance, calculate_rpm, 
                  calculate_static_thrust, calculate_CD_i)
from optimization import optimize_CL_and_T
from data_handling import save_data_to_csv, save_time_series_to_csv
from config import DEFAULT_CD0

def run_single_configuration(params, debug_mode=False):
  """Run single configuration analysis."""
  (watt_limit, wingspan, chord, weight, target_takeoff_distance, 
   sigma, prop_diameter, prop_pitch, max_rpm, mu, epsilon, dt, t_max) = params
  
  # Calculate aircraft parameters
  S = wingspan * chord
  AR = wingspan**2 / S
  m = weight / 32.174  # Convert weight to mass
  
  # Run optimization
  print("\nOptimizing takeoff parameters...")
  best_CL, best_T = optimize_CL_and_T(
      m, S, 0.00237, sigma, mu, 32.174, dt, t_max, 
      watt_limit, target_takeoff_distance, AR, epsilon, weight, debug_mode
  )

  # Calculate takeoff performance
  final_distance, V_TO, takeoff_time = calculate_takeoff_distance(
      [best_CL, best_T], m, S, 0.00237, sigma, mu, 32.174, dt, t_max, 
      watt_limit, AR, epsilon
  )

  # Generate time history data
  t = np.arange(0, takeoff_time, dt)
  V = np.zeros_like(t)
  s = np.zeros_like(t)
  a = np.zeros_like(t)
  T = np.zeros_like(t)
  rpm = np.zeros_like(t)

  # Initialize RPM analysis variables
  rpm_warning = False
  rpm_limited_thrust = False
  thrust_reduction = []

  # Main simulation loop
  for i in range(1, len(t)):
      CL = min(best_CL, (2 * m * 32.174) / (0.00237 * V[i-1]**2 * S)) if V[i-1] > 0 else 0
      CD_i = calculate_CD_i(CL, AR, epsilon)
      CD = DEFAULT_CD0 + CD_i
      
      D = 0.5 * 0.00237 * V[i-1]**2 * S * CD
      L = 0.5 * 0.00237 * V[i-1]**2 * S * CL
      Fr = mu * (m * 32.174 - L)
      
      T[i] = min(best_T, calculate_static_thrust(watt_limit, sigma))
      required_rpm = calculate_rpm(T[i], V[i-1] * 0.3048, prop_diameter, prop_pitch)
      
      if required_rpm > max_rpm:
          rpm_warning = True
          rpm[i] = max_rpm
          velocity_ms = V[i-1] * 0.3048
          actual_thrust_N = 4.392399e-8 * max_rpm * ((prop_diameter**3.5)/np.sqrt(prop_pitch)) * \
                          (4.23333e-4 * max_rpm * prop_pitch - velocity_ms)
          T[i] = actual_thrust_N / 4.44822
          thrust_reduction.append((i * dt, (1 - T[i]/best_T) * 100))
          rpm_limited_thrust = True
      else:
          rpm[i] = required_rpm
      
      F_net = T[i] - D - Fr
      a[i] = F_net / m
      V[i] = V[i-1] + a[i] * dt
      s[i] = s[i-1] + V[i] * dt

  # Save time series data
  time_series_data = {
      'Time(s)': t,
      'Velocity(ft/s)': V,
      'Distance(ft)': s,
      'Acceleration(ft/s²)': a,
      'Thrust(lbf)': T,
      'RPM': rpm
  }
  time_series_file = save_time_series_to_csv(time_series_data, 'single_config_time_series')
  print(f"\nTime series data saved to: {time_series_file}")

  # Save and print results
  print_single_config_results(best_CL, best_T, final_distance, target_takeoff_distance, 
                            weight, takeoff_time, V_TO)

  if rpm_warning:
      handle_rpm_warning(s, t, T, thrust_reduction)

  return best_CL, best_T, final_distance, V_TO, takeoff_time

def run_chord_sweep(params, debug_mode=False):
  """Run chord sweep analysis."""
  (watt_limit, wingspan, weight, target_takeoff_distance, sigma, 
   prop_diameter, prop_pitch, max_rpm, mu, epsilon, dt, t_max,
   min_chord, max_chord, step_size) = params

  # Initialize arrays for sweep results
  chords = np.arange(min_chord, max_chord + step_size, step_size)
  optimal_thrusts = []
  optimal_cls = []
  takeoff_distances = []
  takeoff_times = []

  # Perform sweep analysis
  print("\nPerforming chord sweep analysis...")
  for idx, chord in enumerate(chords):
      print(f"\nAnalyzing Step {idx+1}/{len(chords)} - Chord: {chord:.2f} ft")
      
      # Calculate parameters for this chord
      S = wingspan * chord
      AR = wingspan**2 / S
      m = weight / 32.174

      # Optimize for this configuration
      show_debug = debug_mode and idx == 0
      best_CL, best_T = optimize_CL_and_T(
          m, S, 0.00237, sigma, mu, 32.174, dt, t_max,
          watt_limit, target_takeoff_distance, AR, epsilon, weight, show_debug
      )
      
      # Calculate takeoff performance
      distance, _, time = calculate_takeoff_distance(
          [best_CL, best_T], m, S, 0.00237, sigma, mu, 32.174, dt, t_max,
          watt_limit, AR, epsilon
      )

      # Store and print results
      optimal_thrusts.append(best_T)
      optimal_cls.append(best_CL)
      takeoff_distances.append(distance)
      takeoff_times.append(time)

      print(f"Required thrust: {best_T:.2f} lbf")
      print(f"Optimal CL: {best_CL:.2f}")
      print(f"Takeoff distance: {distance:.2f} ft")
      print(f"Takeoff time: {time:.2f} s")
      print("-" * 50)

  # Save and print sweep results
  save_sweep_results(chords, optimal_thrusts, optimal_cls, takeoff_distances, takeoff_times)

def print_single_config_results(CL, T, distance, target_distance, weight, time, V_TO):
  """Print results for single configuration analysis."""
  print("\n=== ANALYSIS RESULTS ===")
  print(f"Optimal CL_max: {CL:.4f}")
  print(f"Optimal Thrust: {T:.2f} lbf")
  print(f"Fixed CD0: {DEFAULT_CD0:.4f}")
  print(f"Takeoff Distance: {distance:.2f} ft")
  print(f"Target Distance: {target_distance:.2f} ft")
  print(f"Distance Error: {abs(distance - target_distance):.2f} ft "
        f"({abs(distance - target_distance) / target_distance * 100:.2f}%)")
  print(f"Thrust/Weight Ratio: {(T / weight) * 100:.2f}%")
  print(f"Time to Takeoff: {time:.2f} seconds")
  print(f"Takeoff Velocity: {V_TO:.2f} ft/s")

def handle_rpm_warning(s, t, T, thrust_reduction):
  """Handle RPM warning and save related data."""
  print("\n=== RPM LIMITED PERFORMANCE ===")
  actual_takeoff_distance = s[-1]
  actual_takeoff_time = t[-1]
  average_thrust = np.mean(T[1:])
  min_thrust = np.min(T[1:])
  
  rpm_data = {
      'Actual_Takeoff_Distance(ft)': actual_takeoff_distance,
      'Actual_Takeoff_Time(s)': actual_takeoff_time,
      'Average_Thrust(lbf)': average_thrust,
      'Minimum_Thrust(lbf)': min_thrust
  }
  rpm_file = save_data_to_csv(rpm_data, 'rpm_limited_performance')
  print(f"RPM limited performance data saved to: {rpm_file}")
  
  if thrust_reduction:
      max_reduction = max(reduction[1] for reduction in thrust_reduction)
      first_occurrence = thrust_reduction[0][0]
      print(f"\nThrust Analysis:")
      print(f"RPM limiting first occurred at: {first_occurrence:.1f} seconds")
      print(f"Maximum thrust reduction: {max_reduction:.1f}%")
      print(f"Average thrust: {average_thrust:.2f} lbf")
      print(f"Minimum thrust: {min_thrust:.2f} lbf")

def save_sweep_results(chords, thrusts, cls, distances, times):
  """Save and print sweep analysis results."""
  sweep_results = {
      'Chord(ft)': chords,
      'Required_Thrust(lbf)': thrusts,
      'Optimal_CL': cls,
      'Takeoff_Distance(ft)': distances,
      'Takeoff_Time(s)': times
  }
  sweep_file = save_time_series_to_csv(sweep_results, 'chord_sweep_results')
  print(f"\nChord sweep results saved to: {sweep_file}")

  # Find and save optimal configuration
  min_thrust_idx = np.argmin(thrusts)
  optimal_chord = chords[min_thrust_idx]
  
  optimal_config = {
      'Optimal_Chord(ft)': optimal_chord,
      'Required_Thrust(lbf)': thrusts[min_thrust_idx],
      'Optimal_CL': cls[min_thrust_idx],
      'Takeoff_Distance(ft)': distances[min_thrust_idx],
      'Takeoff_Time(s)': times[min_thrust_idx],
      'Min_Thrust(lbf)': min(thrusts),
      'Max_Thrust(lbf)': max(thrusts),
      'Min_CL': min(cls),
      'Max_CL': max(cls),
      'Min_Distance(ft)': min(distances),
      'Max_Distance(ft)': max(distances),
      'Min_Time(s)': min(times),
      'Max_Time(s)': max(times)
  }
  optimal_file = save_data_to_csv(optimal_config, 'chord_sweep_optimal')
  print(f"Optimal configuration data saved to: {optimal_file}")

  print("\n=== CHORD SWEEP SUMMARY ===")
  print(f"Optimal chord length: {optimal_chord:.2f} ft")
  print(f"Required thrust: {thrusts[min_thrust_idx]:.2f} lbf")
  print(f"Optimal CL: {cls[min_thrust_idx]:.2f}")
  print(f"Takeoff distance: {distances[min_thrust_idx]:.2f} ft")
  print(f"Takeoff time: {times[min_thrust_idx]:.2f} s")