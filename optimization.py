#optimization.py Version 1.1
import numpy as np
from scipy.optimize import minimize
import config
from physics import calculate_takeoff_distance, calculate_drag_coefficient

class OptimizationError(Exception):
  """Custom exception for optimization failures."""
  pass

def evaluate_point(CL: float, T: float, params: dict) -> tuple:
  """
  Evaluate a single point in the optimization space.
  
  Args:
      CL (float): Lift coefficient
      T (float): Thrust in lbf
      params (dict): Dictionary containing simulation parameters
          Required keys: m, S, rho, sigma, mu, g, dt, t_max, power_limit,
                       target_distance, AR, epsilon, weight
  
  Returns:
      tuple: (error, time_history, CD)
          - error (float): Distance error from target
          - time_history (dict): Time history of the takeoff run
          - CD (float): Drag coefficient
  """
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

def optimize_CL_and_T(m: float, S: float, rho: float, sigma: float, mu: float, 
                   g: float, dt: float, t_max: float, power_limit: float,
                   target_distance: float, AR: float, epsilon: float, 
                   weight: float) -> tuple:
  """
  Optimize lift coefficient and thrust for minimum takeoff distance.
  
  Args:
      m (float): Aircraft mass in slugs
      S (float): Wing area in sq ft
      rho (float): Air density in slug/ft^3
      sigma (float): Density ratio
      mu (float): Ground friction coefficient
      g (float): Gravitational acceleration ft/s^2
      dt (float): Time step for simulation
      t_max (float): Maximum simulation time
      power_limit (float): Maximum power in watts
      target_distance (float): Target takeoff distance in ft
      AR (float): Wing aspect ratio
      epsilon (float): Oswald efficiency factor
      weight (float): Aircraft weight in lbf
  
  Returns:
      tuple: (best_CL, best_T, best_history, best_CD, optimization_data)
          - best_CL (float): Optimal lift coefficient
          - best_T (float): Optimal thrust
          - best_history (dict): Time history of optimal solution
          - best_CD (float): Drag coefficient at optimal solution
          - optimization_data (dict): Data from optimization process
  
  Raises:
      OptimizationError: If optimization fails to converge
  """
  bounds = config.get_optimization_bounds()
  
  # Package parameters for evaluate_point
  params = {
      'm': m, 'S': S, 'rho': rho, 'sigma': sigma, 'mu': mu,
      'g': g, 'dt': dt, 't_max': t_max, 'power_limit': power_limit,
      'target_distance': target_distance, 'AR': AR, 'epsilon': epsilon,
      'weight': weight
  }
  
  # Objective function to minimize
  def objective(x):
      CL, thrust = x
      error, _, _ = evaluate_point(CL, thrust, params)
      return error if error != float('inf') else 1e6
  
  # Constraint: takeoff must be achieved
  def constraint(x):
      CL, thrust = x
      error, time_history, _ = evaluate_point(CL, thrust, params)
      if time_history is None:
          return -1.0  # Constraint violated
      return 1.0  # Constraint satisfied
  
  # Initial guess (middle of bounds)
  x0 = [
      (bounds['CL'][0] + bounds['CL'][1]) / 2,
      (bounds['thrust'][0] + bounds['thrust'][1]) / 2
  ]
  
  # Define bounds for scipy.optimize
  bounds_list = [
      bounds['CL'],
      bounds['thrust']
  ]
  
  # Define constraint
  con = {'type': 'ineq', 'fun': constraint}
  
  try:
      # Run optimization
      result = minimize(
          objective,
          x0,
          method='SLSQP',
          bounds=bounds_list,
          constraints=con,
          options={
              'ftol': 1e-6,
              'maxiter': 100,
              'disp': config.DEBUG_MODE
          }
      )
      
      if result.success:
          best_CL, best_T = result.x
          _, best_history, best_CD = evaluate_point(best_CL, best_T, params)
          
          # Package optimization data
          optimization_data = {
              'iterations': result.nit,
              'function_calls': result.nfev,
              'final_error': result.fun,
              'success': result.success,
              'message': result.message
          }
          
          return best_CL, best_T, best_history, best_CD, optimization_data
      else:
          raise OptimizationError(f"Optimization failed: {result.message}")
          
  except Exception as e:
      raise OptimizationError(f"Optimization failed with error: {str(e)}")