#analysis.py
import numpy as np
import config
from optimization import optimize_CL_and_T
from plotting import plot_takeoff_trajectory, plot_optimization_results
from data_handling import save_results_to_csv

def run_chord_sweep():
  chord_values = config.get_chord_range()
  results = []
  
  if config.DEBUG_MODE:
      print(f"Starting chord sweep analysis from {chord_values[0]} to {chord_values[-1]} ft")
  
  for chord in chord_values:
      if config.DEBUG_MODE:
          print(f"\nAnalyzing chord = {chord:.2f} ft")
          
      S = chord * config.WINGSPAN  # wing area
      AR = config.WINGSPAN / chord  # aspect ratio
      m = config.WEIGHT / config.GRAVITY  # mass in slugs
      
      CL, T, time_history, CD = optimize_CL_and_T(
          m=m,
          S=S,
          rho=config.RHO,
          sigma=config.SIGMA,
          mu=config.MU,
          g=config.GRAVITY,
          dt=config.DT,
          t_max=config.T_MAX,
          power_limit=config.WATT_LIMIT,
          target_distance=config.TARGET_TAKEOFF_DISTANCE,
          AR=AR,
          epsilon=config.EPSILON,
          weight=config.WEIGHT
      )
      
      results.append({
          'chord': chord,
          'CL': CL,
          'CD': CD,
          'LD_ratio': CL/CD if CD else None,
          'thrust': T,
          'time_history': time_history
      })
      
      if config.SAVE_PLOTS:
          plot_takeoff_trajectory(time_history, chord)
          
      if config.DEBUG_MODE:
          print(f"Results - CL: {CL:.3f}, CD: {CD:.3f}, L/D: {CL/CD:.3f}, Thrust: {T:.2f} lbf")
  
  if config.SAVE_DATA:
      save_results_to_csv(results)
  
  plot_optimization_results(results)
  return results

def run_single_configuration(params):
  if config.DEBUG_MODE:
      print(f"Running single configuration analysis with chord = {params['chord']} ft")
      
  chord = params['chord']
  S = chord * params['wingspan']
  AR = params['wingspan'] / chord
  m = params['weight'] / config.GRAVITY
  
  CL, T, time_history = optimize_CL_and_T(
      m=m,
      S=S,
      rho=config.RHO,
      sigma=params['sigma'],
      mu=params['mu'],
      g=config.GRAVITY,
      dt=params['dt'],
      t_max=params['t_max'],
      power_limit=params['watt_limit'],
      target_distance=params['target_takeoff_distance'],
      AR=AR,
      epsilon=params['epsilon'],
      weight=params['weight']
  )
  
  if config.SAVE_PLOTS:
      plot_takeoff_trajectory(time_history, chord)
  
  if config.DEBUG_MODE:
      print(f"Analysis complete - CL: {CL:.3f}, Thrust: {T:.2f} lbf")
      
  return {'CL': CL, 'thrust': T, 'time_history': time_history}