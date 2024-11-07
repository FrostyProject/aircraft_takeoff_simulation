"""Optimization functions for aircraft takeoff analysis."""

import numpy as np
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from config import (CL_STARTS, CL_BOUNDS, THRUST_WEIGHT_RATIO_BOUNDS, 
                 MAX_ITERATIONS, CONVERGENCE_THRESHOLD)
from physics import calculate_takeoff_distance
from data_handling import save_convergence_data

def objective(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_distance, AR, epsilon, weight):
  """Objective function for optimization with sanity checks."""
  CL, T = params
  
  # Sanity checks
  if CL < 0.8 or CL > 3.2:
      return float('inf')
  if T < 0.15 * weight or T > 1.3 * weight:
      return float('inf')
      
  distance, _, _ = calculate_takeoff_distance(params, m, S, rho, sigma, mu, g, dt, t_max, power_limit, AR, epsilon)
  return abs(distance - target_distance) / target_distance

def optimize_CL_and_T(m, S, rho, sigma, mu, g, dt, t_max, power_limit, target_distance, AR, epsilon, weight, debug_mode=False):
  """Optimize CL_max and thrust using multi-start optimization."""
  best_overall_CL = None
  best_overall_T = None
  best_overall_error = float('inf')
  
  # Start with maximum CL value
  cl_start = CL_BOUNDS[1]  # Use upper bound of CL range
  initial_T = 0.8 * weight
  convergence_history = []
  iteration_counter = 0
  
  pbar = tqdm(total=MAX_ITERATIONS, desc="Optimizing", ncols=70)
  
  current_CL = cl_start
  current_T = initial_T
  
  for i in range(MAX_ITERATIONS):
      # Optimize thrust with current CL
      res_T = minimize_scalar(
          lambda T: objective([current_CL, T], m, S, rho, sigma, mu, g, dt, t_max, 
                            power_limit, target_distance, AR, epsilon, weight),
          bounds=(THRUST_WEIGHT_RATIO_BOUNDS[0] * weight, 
                 THRUST_WEIGHT_RATIO_BOUNDS[1] * weight),
          method='bounded'
      )
      
      # Optimize CL with new thrust
      res_CL = minimize_scalar(
          lambda CL: objective([CL, res_T.x], m, S, rho, sigma, mu, g, dt, t_max,
                             power_limit, target_distance, AR, epsilon, weight),
          bounds=CL_BOUNDS,
          method='bounded'
      )
      
      error = objective([res_CL.x, res_T.x], m, S, rho, sigma, mu, g, dt, t_max,
                      power_limit, target_distance, AR, epsilon, weight)
      
      if debug_mode:
          convergence_history.append({
              'iteration': iteration_counter,
              'start_CL': cl_start,
              'CL': res_CL.x,
              'T': res_T.x,
              'error': error
          })
      
      current_CL = res_CL.x
      current_T = res_T.x
      
      if error < best_overall_error:
          best_overall_CL = current_CL
          best_overall_T = current_T
          best_overall_error = error
      
      iteration_counter += 1
      pbar.update(1)
      pbar.set_postfix({'error': f'{best_overall_error:.4f}'})
      
      if error <= CONVERGENCE_THRESHOLD:
          break
  
  pbar.close()
  
  if debug_mode and convergence_history:
      save_convergence_data(convergence_history)
      print("\n=== Optimization Convergence Details ===")
      print(f"Total iterations: {iteration_counter}")
      print(f"Final error: {best_overall_error:.6f}")
      print(f"Starting CL value: {cl_start}")
      print(f"CL range: {min([x['CL'] for x in convergence_history]):.4f} "
            f"to {max([x['CL'] for x in convergence_history]):.4f}")
      print(f"Thrust range: {min([x['T'] for x in convergence_history]):.2f} "
            f"to {max([x['T'] for x in convergence_history]):.2f}")
      
      if best_overall_error > 0.01:
          print("\nWARNING: Optimization may not have found optimal solution")
          print(f"Best error achieved: {best_overall_error:.6f}")
  
  # Solution verification
  if best_overall_CL < 1.0 or best_overall_CL > 3.0:
      print(f"\nWARNING: Optimal CL ({best_overall_CL:.2f}) outside typical range")
  
  if best_overall_T < 0.2 * weight or best_overall_T > 1.2 * weight:
      print(f"\nWARNING: Optimal thrust ({best_overall_T:.2f} lbf) outside typical range")
  
  return best_overall_CL, best_overall_T