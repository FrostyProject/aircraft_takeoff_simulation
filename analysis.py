import numpy as np
from tqdm import tqdm
from optimization import optimize_CL_and_T
from physics import calculate_takeoff_distance, check_lift_balance
from data_handling import save_results_to_csv
from config import GRAVITY, RHO_SL

def run_single_configuration(params, debug_mode=False):
  """Run analysis for a single aircraft configuration."""
  # Calculate derived parameters
  S = params['wingspan'] * params['chord']
  AR = params['wingspan']**2 / S
  m = params['weight'] / GRAVITY
  
  print("\n=== Running Single Configuration Analysis ===")
  print(f"Wing Area: {S:.2f} ft²")
  print(f"Aspect Ratio: {AR:.2f}")
  
  # Run optimization
  best_CL, best_T = optimize_CL_and_T(
      m, S, RHO_SL, params['sigma'], params['mu'], GRAVITY, 
      params['dt'], params['t_max'], params['watt_limit'],
      params['target_takeoff_distance'], AR, params['epsilon'],
      params['weight'], debug_mode
  )
  
  if best_CL is None or best_T is None:
      print("\nERROR: Failed to find valid configuration")
      return None
  
  # Calculate takeoff performance
  final_distance, V_TO, takeoff_time, rotation_point = calculate_takeoff_distance(
      [best_CL, best_T], m, S, RHO_SL, params['sigma'], params['mu'],
      GRAVITY, params['dt'], params['t_max'], params['watt_limit'],
      AR, params['epsilon']
  )
  
  # Print results
  print("\n=== RESULTS ===")
  print(f"Optimal CL: {best_CL:.3f}")
  print(f"Optimal Thrust: {best_T:.1f} lbf")
  print(f"Takeoff Distance: {final_distance:.1f} ft")
  print(f"Takeoff Speed: {V_TO:.1f} ft/s ({V_TO * 0.592:.1f} knots)")
  print(f"Takeoff Time: {takeoff_time:.1f} seconds")
  
  # Print lift analysis
  print("\n=== LIFT ANALYSIS ===")
  if rotation_point is not None:
      print(f"Rotation point: {rotation_point:.2f} ft")
      print(f"Ground roll after rotation: {final_distance - rotation_point:.2f} ft")
      print(f"Percent of takeoff distance in rotation: {(rotation_point/final_distance)*100:.1f}%")
  else:
      print("WARNING: Aircraft did not achieve full lift during takeoff!")
      
  # Calculate lift at takeoff speed
  final_lift_check = check_lift_balance(best_CL, V_TO, S, RHO_SL, params['weight'])
  print(f"Lift ratio at takeoff: {final_lift_check[1]:.3f}")
  if final_lift_check[1] < 1.0:
      print("WARNING: Insufficient lift at takeoff speed!")
  
  # Save results
  results = {
      'chord': params['chord'],
      'CL': best_CL,
      'thrust': best_T,
      'distance': final_distance,
      'velocity': V_TO,
      'time': takeoff_time,
      'rotation_point': rotation_point,
      'lift_ratio': final_lift_check[1]
  }
  
  save_results_to_csv([results], 'single_analysis_results.csv', params)
  
  return results

def check_and_rerun_outliers(results, params, debug_mode=False):
  """
  Check for outliers in CL and thrust values and rerun those configurations.
  
  Args:
      results (list): List of dictionaries containing analysis results
      params (dict): Analysis parameters
      debug_mode (bool): Whether to print debug information
  
  Returns:
      list: Updated results with rerun configurations
  """
  if len(results) < 3:  # Need at least 3 points to check for outliers
      return results
  
  # Sort results by chord
  results = sorted(results, key=lambda x: x['chord'])
  
  # Convert to numpy arrays for analysis
  chords = np.array([r['chord'] for r in results])
  CLs = np.array([r['CL'] for r in results])
  thrusts = np.array([r['thrust'] for r in results])
  
  # Calculate moving averages and standard deviations with window size 3
  window_size = 3
  outliers_to_rerun = set()  # Use set to avoid duplicates
  
  for i in range(1, len(results) - 1):
      # Get window indices
      start_idx = max(0, i - 1)
      end_idx = min(len(results), i + 2)
      
      # Calculate local stats for CL
      local_CL_mean = np.mean(CLs[start_idx:end_idx])
      local_CL_std = np.std(CLs[start_idx:end_idx])
      
      # Calculate local stats for thrust
      local_thrust_mean = np.mean(thrusts[start_idx:end_idx])
      local_thrust_std = np.std(thrusts[start_idx:end_idx])
      
      # Check for outliers (more than 2 standard deviations from local mean)
      if (abs(CLs[i] - local_CL_mean) > 2 * local_CL_std or 
          abs(thrusts[i] - local_thrust_mean) > 2 * local_thrust_std):
          outliers_to_rerun.add(i)
          if debug_mode:
              print(f"\nPotential outlier detected at chord = {chords[i]:.2f}:")
              print(f"CL: {CLs[i]:.3f} (local mean: {local_CL_mean:.3f} ± {local_CL_std:.3f})")
              print(f"Thrust: {thrusts[i]:.1f} (local mean: {local_thrust_mean:.1f} ± {local_thrust_std:.1f})")
  
  # Rerun outlier configurations
  if outliers_to_rerun:
      print(f"\nRerunning {len(outliers_to_rerun)} configurations identified as potential outliers...")
      
      for idx in outliers_to_rerun:
          chord = results[idx]['chord']
          if debug_mode:
              print(f"\nRerunning analysis for chord = {chord:.2f} ft")
          
          # Try up to 3 times to get a better result
          best_error = float('inf')
          best_result = None
          
          for attempt in range(3):
              try:
                  S = params['wingspan'] * chord
                  AR = params['wingspan']**2 / S
                  m = params['weight'] / GRAVITY
                  
                  # Use different initial conditions for each attempt
                  if attempt == 0:
                      # Use average of neighboring values as initial guess
                      init_CL = (CLs[idx-1] + CLs[idx+1]) / 2 if 0 < idx < len(CLs)-1 else CLs[idx]
                      init_T = (thrusts[idx-1] + thrusts[idx+1]) / 2 if 0 < idx < len(thrusts)-1 else thrusts[idx]
                  else:
                      # Use random perturbation for subsequent attempts
                      init_CL = CLs[idx] * (1 + 0.1 * (np.random.random() - 0.5))
                      init_T = thrusts[idx] * (1 + 0.1 * (np.random.random() - 0.5))
                  
                  best_CL, best_T = optimize_CL_and_T(
                      m, S, RHO_SL, params['sigma'], params['mu'], GRAVITY,
                      params['dt'], params['t_max'], params['watt_limit'],
                      params['target_takeoff_distance'], AR, params['epsilon'],
                      params['weight'], debug_mode
                  )
                  
                  distance, V_TO, time, rotation_point = calculate_takeoff_distance(
                      [best_CL, best_T], m, S, RHO_SL, params['sigma'], params['mu'],
                      GRAVITY, params['dt'], params['t_max'], params['watt_limit'],
                      AR, params['epsilon']
                  )
                  
                  lift_check = check_lift_balance(best_CL, V_TO, S, RHO_SL, params['weight'])
                  
                  # Calculate error relative to target distance
                  error = abs(distance - params['target_takeoff_distance'])
                  
                  if error < best_error:
                      best_error = error
                      best_result = {
                          'chord': chord,
                          'wing_area': S,
                          'aspect_ratio': AR,
                          'CL': best_CL,
                          'thrust': best_T,
                          'distance': distance,
                          'velocity': V_TO,
                          'time': time,
                          'rotation_point': rotation_point,
                          'lift_ratio': lift_check[1]
                      }
              
              except Exception as e:
                  if debug_mode:
                      print(f"Rerun attempt {attempt+1} failed: {str(e)}")
                  continue
          
          if best_result is not None:
              results[idx] = best_result
              if debug_mode:
                  print(f"Updated values - CL: {best_result['CL']:.3f}, Thrust: {best_result['thrust']:.1f}")
  
  return results

def run_chord_sweep(params, debug_mode=False):
  """Run analysis across a range of chord values."""
  # Calculate number of steps
  num_steps = int((params['max_chord'] - params['min_chord']) / params['chord_step']) + 1
  chord_values = np.linspace(params['min_chord'], params['max_chord'], num_steps)
  
  results = []
  failed_configs = []
  
  progress_bar = tqdm(chord_values, desc="Analyzing chords", ncols=70)
  
  for chord in progress_bar:
      current_params = params.copy()
      current_params['chord'] = chord
      
      try:
          S = params['wingspan'] * chord
          AR = params['wingspan']**2 / S
          m = params['weight'] / GRAVITY
          
          best_CL, best_T = optimize_CL_and_T(
              m, S, RHO_SL, params['sigma'], params['mu'], GRAVITY,
              params['dt'], params['t_max'], params['watt_limit'],
              params['target_takeoff_distance'], AR, params['epsilon'],
              params['weight'], debug_mode
          )
          
          if best_CL is None or best_T is None:
              raise ValueError("Optimization failed to find valid solution")
          
          distance, V_TO, time, rotation_point = calculate_takeoff_distance(
              [best_CL, best_T], m, S, RHO_SL, params['sigma'], params['mu'],
              GRAVITY, params['dt'], params['t_max'], params['watt_limit'],
              AR, params['epsilon']
          )
          
          lift_check = check_lift_balance(best_CL, V_TO, S, RHO_SL, params['weight'])
          
          results.append({
              'chord': chord,
              'wing_area': S,
              'aspect_ratio': AR,
              'CL': best_CL,
              'thrust': best_T,
              'distance': distance,
              'velocity': V_TO,
              'time': time,
              'rotation_point': rotation_point,
              'lift_ratio': lift_check[1]
          })
          
          progress_bar.set_postfix({
              'CL': f'{best_CL:.2f}',
              'dist': f'{distance:.0f}ft'
          })
          
      except Exception as e:
          failed_configs.append({
              'chord': chord,
              'error': str(e)
          })
          if debug_mode:
              print(f"\nFailed analysis for chord={chord:.2f}: {str(e)}")
  
  progress_bar.close()
  
  # Save and analyze results
  if results:
      # Check for outliers and rerun if necessary
      results = check_and_rerun_outliers(results, params, debug_mode)
      
      # Save final results
      save_results_to_csv(results, 'chord_sweep_results.csv', params)
      
      # Find best configuration
      target_distance = params['target_takeoff_distance']
      distance_errors = [abs(r['distance'] - target_distance) for r in results]
      best_idx = np.argmin(distance_errors)
      best_result = results[best_idx]
      
      # Print best configuration
      print("\n=== Best Configuration ===")
      print(f"Chord: {best_result['chord']:.2f} ft")
      print(f"Wing Area: {best_result['wing_area']:.2f} ft²")
      print(f"Aspect Ratio: {best_result['aspect_ratio']:.2f}")
      print(f"CL: {best_result['CL']:.3f}")
      print(f"Thrust: {best_result['thrust']:.1f} lbf")
      print(f"Takeoff Distance: {best_result['distance']:.1f} ft")
      print(f"Takeoff Speed: {best_result['velocity']:.1f} ft/s ({best_result['velocity'] * 0.592:.1f} knots)")
      print(f"Takeoff Time: {best_result['time']:.1f} seconds")
      if best_result['rotation_point'] is not None:
          print(f"Rotation Point: {best_result['rotation_point']:.1f} ft")
      print(f"Lift Ratio at Takeoff: {best_result['lift_ratio']:.3f}")
      
  else:
      print("\nNo valid configurations found!")
      if failed_configs:
          print("\nAll configurations failed!")
  
  return results, failed_configs

def analyze_results(results):
  """Analyze sweep results for trends and recommendations."""
  if not results:
      return
  
  # Convert results to numpy arrays for analysis
  chords = np.array([r['chord'] for r in results])
  distances = np.array([r['distance'] for r in results])
  CLs = np.array([r['CL'] for r in results])
  thrusts = np.array([r['thrust'] for r in results])
  velocities = np.array([r['velocity'] for r in results])
  
  # Find trends
  print("\n=== Performance Trends ===")
  
  # CL trend
  cl_trend = np.polyfit(chords, CLs, 1)
  print(f"CL trend: {'increases' if cl_trend[0] > 0 else 'decreases'} with chord")
  print(f"CL range: {np.min(CLs):.2f} to {np.max(CLs):.2f}")
  
  # Thrust trend
  thrust_trend = np.polyfit(chords, thrusts, 1)
  print(f"Thrust trend: {'increases' if thrust_trend[0] > 0 else 'decreases'} with chord")
  print(f"Thrust range: {np.min(thrusts):.1f} to {np.max(thrusts):.1f} lbf")
  
  # Velocity trend
  velocity_trend = np.polyfit(chords, velocities, 1)
  print(f"Velocity trend: {'increases' if velocity_trend[0] > 0 else 'decreases'} with chord")
  print(f"Velocity range: {np.min(velocities):.1f} to {np.max(velocities):.1f} ft/s")
  
  # Distance sensitivity
  distance_sensitivity = np.gradient(distances, chords)
  avg_sensitivity = np.mean(np.abs(distance_sensitivity))
  print(f"Average distance sensitivity: {avg_sensitivity:.1f} ft per ft of chord")
  
  # Recommendations
  print("\n=== Design Recommendations ===")
  
  # Find chord range meeting target distance within 10%
  mean_distance = np.mean(distances)
  tolerance = 0.1  # 10%
  valid_mask = np.abs(distances - mean_distance) <= mean_distance * tolerance
  valid_chords = chords[valid_mask]
  
  if len(valid_chords) > 0:
      print(f"Recommended chord range: {np.min(valid_chords):.2f} to {np.max(valid_chords):.2f} ft")
      
      # Find most efficient configuration (lowest thrust)
      valid_results = [r for i, r in enumerate(results) if valid_mask[i]]
      min_thrust_result = min(valid_results, key=lambda x: x['thrust'])
      
      print("\nMost efficient valid configuration:")
      print(f"Chord: {min_thrust_result['chord']:.2f} ft")
      print(f"CL: {min_thrust_result['CL']:.3f}")
      print(f"Thrust: {min_thrust_result['thrust']:.1f} lbf")
      print(f"Distance: {min_thrust_result['distance']:.1f} ft")
      print(f"Velocity: {min_thrust_result['velocity']:.1f} ft/s")
      
      # Find most stable configuration (highest lift ratio)
      max_lift_result = max(valid_results, key=lambda x: x['lift_ratio'])
      print("\nMost stable valid configuration:")
      print(f"Chord: {max_lift_result['chord']:.2f} ft")
      print(f"CL: {max_lift_result['CL']:.3f}")
      print(f"Lift ratio: {max_lift_result['lift_ratio']:.3f}")
      print(f"Distance: {max_lift_result['distance']:.1f} ft")
  else:
      print("No configurations found within 10% of mean distance")