import sys
from analysis import run_single_configuration, run_chord_sweep
from config import get_default_params

def get_user_input():
  """Get analysis parameters from user input."""
  print("\n=== Aircraft Takeoff Analysis Tool ===")
  print("1. Single Configuration Analysis")
  print("2. Chord Sweep Analysis")
  
  while True:
      analysis_type = input("\nSelect analysis type (1 or 2): ")
      if analysis_type in ['1', '2']:
          break
      print("Invalid selection. Please choose 1 or 2.")

  debug_mode = input("\nEnable debug mode? (y/n): ").lower() == 'y'
  
  # Get common parameters
  print("\n=== Common Parameters ===")
  params = get_default_params()
  
  params['watt_limit'] = float(input("Enter electric motor watt limit: "))
  params['wingspan'] = float(input("Enter wingspan (ft): "))
  params['weight'] = float(input("Enter aircraft weight (lbs): "))
  params['target_takeoff_distance'] = float(input("Enter target takeoff distance (ft): "))
  params['sigma'] = float(input("Enter atmospheric density ratio (sigma): "))
  params['prop_diameter'] = float(input("Enter propeller diameter (inches): "))
  params['prop_pitch'] = float(input("Enter propeller pitch (inches): "))
  params['max_rpm'] = float(input("Enter maximum motor RPM: "))
  
  if analysis_type == "1":
      params['chord'] = float(input("\nEnter chord (ft): "))
  else:
      # Get chord sweep parameters
      print("\n=== Chord Sweep Parameters ===")
      params['min_chord'] = float(input("Enter minimum chord (ft): "))
      params['max_chord'] = float(input("Enter maximum chord (ft): "))
      params['chord_step'] = float(input("Enter chord step size (ft): "))
      
      # Validate chord sweep parameters
      if params['min_chord'] >= params['max_chord']:
          raise ValueError("Minimum chord must be less than maximum chord")
      if params['chord_step'] <= 0:
          raise ValueError("Chord step size must be positive")
      if params['chord_step'] > (params['max_chord'] - params['min_chord']):
          raise ValueError("Chord step size too large for chord range")
  
  return analysis_type, params, debug_mode

def main():
  """Main program execution."""
  try:
      analysis_type, params, debug_mode = get_user_input()
      
      if analysis_type == "1":
          run_single_configuration(params, debug_mode)
      else:
          run_chord_sweep(params, debug_mode)
          
  except KeyboardInterrupt:
      print("\nAnalysis interrupted by user.")
      sys.exit(0)
  except Exception as e:
      print(f"\nError: {str(e)}")
      sys.exit(1)

if __name__ == "__main__":
  main()