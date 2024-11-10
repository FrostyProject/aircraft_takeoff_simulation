#main.py
import config
from analysis import run_chord_sweep, run_single_configuration
import matplotlib.pyplot as plt
import os

def main():
  # Initialize configuration and create directories
  config.initialize()
  
  if config.DEBUG_MODE:
      print("Starting aircraft takeoff analysis...")
  
  if config.CHORD_SWEEP:
      results = run_chord_sweep()
  else:
      # Run single configuration with default chord
      params = {
          'wingspan': config.WINGSPAN,
          'chord': config.MIN_CHORD,
          'weight': config.WEIGHT,
          'sigma': config.SIGMA,
          'mu': config.MU,
          'dt': config.DT,
          't_max': config.T_MAX,
          'watt_limit': config.WATT_LIMIT,
          'target_takeoff_distance': config.TARGET_TAKEOFF_DISTANCE,
          'epsilon': config.EPSILON
      }
      results = run_single_configuration(params)
  
  plt.style.use(config.PLOT_STYLE)
  if config.DEBUG_MODE:
      print("Analysis complete.")
      print(f"Results saved in {config.OUTPUT_DIR}")

if __name__ == "__main__":
  main()