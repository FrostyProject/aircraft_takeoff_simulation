#optimization.py
import numpy as np
import config
from physics import calculate_takeoff_distance, calculate_drag_coefficient

def evaluate_point(CL, T, params):
  """Evaluate a single point in the optimization space."""
  # Calculate CD before calling calculate_takeoff_distance
  CD = calculate_drag_coefficient(CL, params['AR'], params['epsilon'])
  
  time_history = calculate_takeoff_distance(
      CL=CL,
      thrust=T,
      m=params['m'],
      S=params['S'],
      rho=params['rho'],
      sigma=params['sigma'],
      mu=params['mu'],
      g=params['g'],
      dt=params['dt'],
      t_max=params['t_max'],
      power_limit=params['power_limit'],
      target_distance=params['target_distance'],
      AR=params['AR'],
      epsilon=params['epsilon'],
      weight=params['weight']
  )
  
  if time_history is None:
      return float('inf'), None, None
  
  final_distance = time_history['distance'][-1]
  error = abs(final_distance - params['target_distance'])
  return error, time_history, CD

def optimize_CL_and_T(m, S, rho, sigma, mu, g, dt, t_max, power_limit, 
                   target_distance, AR, epsilon, weight):
  bounds = config.get_optimization_bounds()
  
  # Package parameters for evaluate_point
  params = {
      'm': m,
      'S': S,
      'rho': rho,
      'sigma': sigma,
      'mu': mu,
      'g': g,
      'dt': dt,
      't_max': t_max,
      'power_limit': power_limit,
      'target_distance': target_distance,
      'AR': AR,
      'epsilon': epsilon,
      'weight': weight
  }
  
  # Coarse grid search
  CL_range = np.linspace(bounds['CL'][0], bounds['CL'][1], config.COARSE_GRID_POINTS)
  T_range = np.linspace(bounds['thrust'][0], bounds['thrust'][1], config.COARSE_GRID_POINTS)
  
  best_error = float('inf')
  best_CL = None
  best_T = None
  best_history = None
  best_CD = None
  
  for CL in CL_range:
      for T in T_range:
          error, time_history, CD = evaluate_point(CL, T, params)
          if error < best_error:
              best_error = error
              best_CL = CL
              best_T = T
              best_history = time_history
              best_CD = CD
  
  # Fine grid search around best point
  if best_CL is not None and best_T is not None:
      CL_min = max(bounds['CL'][0], best_CL - (bounds['CL'][1] - bounds['CL'][0])/config.COARSE_GRID_POINTS)
      CL_max = min(bounds['CL'][1], best_CL + (bounds['CL'][1] - bounds['CL'][0])/config.COARSE_GRID_POINTS)
      T_min = max(bounds['thrust'][0], best_T - (bounds['thrust'][1] - bounds['thrust'][0])/config.COARSE_GRID_POINTS)
      T_max = min(bounds['thrust'][1], best_T + (bounds['thrust'][1] - bounds['thrust'][0])/config.COARSE_GRID_POINTS)
      
      CL_range = np.linspace(CL_min, CL_max, config.FINE_GRID_POINTS)
      T_range = np.linspace(T_min, T_max, config.FINE_GRID_POINTS)
      
      for CL in CL_range:
          for T in T_range:
              error, time_history, CD = evaluate_point(CL, T, params)
              if error < best_error:
                  best_error = error
                  best_CL = CL
                  best_T = T
                  best_history = time_history
                  best_CD = CD
  
  if config.DEBUG_MODE:
      print(f"Optimization complete - Error: {best_error:.3f} ft")
      print(f"Best CL: {best_CL:.3f}")
      print(f"Best CD: {best_CD:.3f}")
      print(f"Best L/D: {best_CL/best_CD:.3f}")
  
  return best_CL, best_T, best_history, best_CD