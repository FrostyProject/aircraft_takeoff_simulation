import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
from physics import calculate_takeoff_distance, check_lift_balance
from data_handling import save_convergence_data
from config import (CL_BOUNDS, THRUST_WEIGHT_RATIO_BOUNDS, 
               MAX_ITERATIONS, CONVERGENCE_THRESHOLD)

def optimize_CL_and_T(m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_distance, AR, epsilon, weight, debug_mode=False):
  """
  Optimize to find lowest viable CL and thrust combination that achieves target distance.
  Uses a multi-start grid search followed by local optimization.
  """
  best_overall_CL = None
  best_overall_T = None
  best_overall_error = float('inf')
  
  # Create a finer grid with multiple starting points
  cl_values = np.linspace(CL_BOUNDS[0], CL_BOUNDS[1], 15)  # Increased from 10 to 15 points
  thrust_values = np.linspace(THRUST_WEIGHT_RATIO_BOUNDS[0] * weight, 
                            THRUST_WEIGHT_RATIO_BOUNDS[1] * weight, 15)
  
  convergence_history = []
  
  try:
      pbar = tqdm(total=len(cl_values) * len(thrust_values), desc="Grid Search", ncols=70)
      
      # Grid search phase with multiple starting points
      for cl in cl_values:
          for thrust in thrust_values:
              # Calculate minimum velocity needed for lift
              v_min = np.sqrt((2 * weight) / (rho * S * cl))
              
              try:
                  distance, v_final, time, rotation_point = calculate_takeoff_distance(
                      [cl, thrust], m, S, rho, sigma, mu, g, dt, t_max, 
                      power_limit, AR, epsilon
                  )
                  
                  # Enhanced validation criteria
                  if rotation_point is not None:
                      error = abs(distance - target_distance) / target_distance
                      
                      # Add penalties to avoid getting stuck at CL=1
                      if abs(cl - 1.0) < 0.1:  # If CL is too close to 1
                          error += 0.5  # Add penalty
                      
                      if error < best_overall_error:
                          best_overall_CL = cl
                          best_overall_T = thrust
                          best_overall_error = error
                          
                          if debug_mode:
                              convergence_history.append({
                                  'CL': cl,
                                  'T': thrust,
                                  'distance': distance,
                                  'error': error,
                                  'rotation_point': rotation_point,
                                  'v_final': v_final
                              })
              
              except Exception as e:
                  if debug_mode:
                      print(f"\nWarning: Failed for CL={cl:.2f}, T={thrust:.2f}: {str(e)}")
                  continue
              
              pbar.update(1)
              pbar.set_postfix({'error': f'{best_overall_error:.4f}'})
      
      pbar.close()
      
      # Multiple local optimizations from different starting points
      if best_overall_CL is not None:
          # Try multiple starting points around the best grid point
          start_points = [
              [best_overall_CL, best_overall_T],
              [best_overall_CL * 1.2, best_overall_T],
              [best_overall_CL * 0.8, best_overall_T],
              [best_overall_CL, best_overall_T * 1.2],
              [best_overall_CL, best_overall_T * 0.8]
          ]
          
          for start_point in start_points:
              result = minimize(
                  lambda x: objective_with_constraints(x, m, S, rho, sigma, mu, g, dt, t_max,
                                                  power_limit, target_distance, AR, epsilon, weight),
                  x0=start_point,
                  bounds=[CL_BOUNDS, (THRUST_WEIGHT_RATIO_BOUNDS[0] * weight, 
                                  THRUST_WEIGHT_RATIO_BOUNDS[1] * weight)],
                  method='SLSQP',
                  options={'maxiter': 100}
              )
              
              if result.success and result.fun < best_overall_error:
                  best_overall_CL, best_overall_T = result.x
                  best_overall_error = result.fun
      
      # Validation check
      if best_overall_CL is not None and abs(best_overall_CL - 1.0) < 0.1:
          # If stuck near CL=1, try another optimization with different bounds
          new_bounds = [(1.2, CL_BOUNDS[1]), (THRUST_WEIGHT_RATIO_BOUNDS[0] * weight,
                                             THRUST_WEIGHT_RATIO_BOUNDS[1] * weight)]
          result = minimize(
              lambda x: objective_with_constraints(x, m, S, rho, sigma, mu, g, dt, t_max,
                                              power_limit, target_distance, AR, epsilon, weight),
              x0=[1.5, best_overall_T],
              bounds=new_bounds,
              method='SLSQP'
          )
          
          if result.success and result.fun < best_overall_error * 1.1:  # Allow slightly worse solution
              best_overall_CL, best_overall_T = result.x
      
      if debug_mode:
          print("\n=== Optimization Results ===")
          print(f"Final CL: {best_overall_CL:.3f}")
          print(f"Final Thrust: {best_overall_T:.1f} lbf")
          print(f"Final error: {best_overall_error:.6f}")
          print(f"Thrust-to-Weight ratio: {best_overall_T/weight:.3f}")
          
          if convergence_history:
              save_convergence_data(convergence_history)
      
      return best_overall_CL, best_overall_T
  
  except Exception as e:
      print(f"\nERROR: Optimization failed: {str(e)}")
      return None, None

def objective_with_constraints(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, 
                         target_distance, AR, epsilon, weight):
  """
  Enhanced objective function with additional penalties to avoid CL=1
  """
  CL, T = params
  
  try:
      distance, v_final, time, rotation_point = calculate_takeoff_distance(
          params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon
      )
      
      if rotation_point is None:
          return float('inf')
      
      # Primary objective: match target distance
      error = abs(distance - target_distance) / target_distance
      
      # Enhanced penalties
      cl_penalty = 0.1 * (CL / CL_BOUNDS[1])**2
      thrust_penalty = 0.1 * (T / (weight * THRUST_WEIGHT_RATIO_BOUNDS[1]))**2
      rotation_penalty = 0.1 * (rotation_point / distance)**2
      
      # Additional penalty for CL near 1
      cl_avoidance_penalty = 0.5 / (abs(CL - 1.0) + 0.1)
      
      total_cost = error + cl_penalty + thrust_penalty + rotation_penalty + cl_avoidance_penalty
      
      return total_cost if np.isfinite(total_cost) else float('inf')
  
  except Exception:
      return float('inf')