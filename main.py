"""Main execution file for aircraft takeoff analysis."""

from config import (DEFAULT_MU, DEFAULT_EPSILON, DEFAULT_DT, DEFAULT_T_MAX)
from analysis import run_single_configuration, run_chord_sweep

def main():
  print("\n=== Aircraft Takeoff Analysis Tool ===")
  print("1. Single Configuration Analysis")
  print("2. Chord Sweep Analysis")
  
  while True:
      analysis_type = input("\nSelect analysis type (1 or 2): ")
      if analysis_type in ['1', '2']:
          break
      print("Invalid selection. Please choose 1 or 2.")

  # Debug mode selection
  debug_mode = input("\nEnable debug mode? (y/n): ").lower() == 'y'

  # Collect common parameters
  print("\n=== Common Parameters ===")
  watt_limit = float(input("Enter electric motor watt limit: "))
  wingspan = float(input("Enter wingspan (ft): "))
  weight = float(input("Enter aircraft weight (lbs): "))
  target_takeoff_distance = float(input("Enter target takeoff distance (ft): "))
  sigma = float(input("Enter atmospheric density ratio (sigma): "))
  prop_diameter = float(input("Enter propeller diameter (inches): "))
  prop_pitch = float(input("Enter propeller pitch (inches): "))
  max_rpm = float(input("Enter maximum motor RPM: "))

  if analysis_type == "1":
      chord = float(input("\nEnter chord (ft): "))
      params = (watt_limit, wingspan, chord, weight, target_takeoff_distance,
               sigma, prop_diameter, prop_pitch, max_rpm, DEFAULT_MU,
               DEFAULT_EPSILON, DEFAULT_DT, DEFAULT_T_MAX)
      run_single_configuration(params, debug_mode)
  else:
      print("\n=== Chord Sweep Parameters ===")
      min_chord = float(input("Enter minimum chord length (ft): "))
      max_chord = float(input("Enter maximum chord length (ft): "))
      step_size = float(input("Enter chord step size (ft): "))
      
      params = (watt_limit, wingspan, weight, target_takeoff_distance,
               sigma, prop_diameter, prop_pitch, max_rpm, DEFAULT_MU,
               DEFAULT_EPSILON, DEFAULT_DT, DEFAULT_T_MAX,
               min_chord, max_chord, step_size)
      run_chord_sweep(params, debug_mode)

  print("\nAnalysis complete.")

if __name__ == "__main__":
  main()