import csv
import json
import os
from datetime import datetime

def create_run_folder(params=None):
  """
  Create a folder for the current run with timestamp and configuration details.
  
  Args:
      params (dict): Dictionary containing analysis parameters
      
  Returns:
      str: Path to the created folder
  """
  # Create base folder name with timestamp
  timestamp = datetime.now().strftime("%m-%d-%y_%H%M")
  
  # Add configuration details to folder name
  if params is None:
      folder_name = f"run_{timestamp}"
  elif 'max_chord' in params:  # Chord sweep
      folder_name = f"run_{timestamp}_sweep_{params['min_chord']:.2f}-{params['max_chord']:.2f}"
  else:  # Single configuration
      folder_name = f"run_{timestamp}_chord_{params['chord']:.2f}"
  
  # Create folder if it doesn't exist
  if not os.path.exists(folder_name):
      os.makedirs(folder_name)
  
  return folder_name

def save_results_to_csv(results, filename, params=None):
  """
  Save analysis results to a CSV file in a dedicated run folder.
  
  Args:
      results (list): List of dictionaries containing analysis results
      filename (str): Base name of the output file
      params (dict): Dictionary containing analysis parameters
  """
  if not results:
      return
  
  # Create run folder
  folder_path = create_run_folder(params)
  
  # Create full file path
  file_path = os.path.join(folder_path, filename)
  
  # Get all unique keys from all result dictionaries
  fieldnames = set()
  for result in results:
      fieldnames.update(result.keys())
  fieldnames = sorted(list(fieldnames))
  
  try:
      with open(file_path, 'w', newline='') as csvfile:
          writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
          writer.writeheader()
          for result in results:
              writer.writerow(result)
      print(f"\nResults saved to: {file_path}")
  except Exception as e:
      print(f"\nError saving results: {str(e)}")

def save_convergence_data(convergence_history, params=None):
  """
  Save optimization convergence history to a JSON file in a dedicated run folder.
  
  Args:
      convergence_history (list): List of dictionaries containing convergence data
      params (dict): Dictionary containing analysis parameters
  """
  if not convergence_history:
      return
  
  # Create run folder
  folder_path = create_run_folder(params)
  
  # Create filename
  filename = 'convergence_history.json'
  file_path = os.path.join(folder_path, filename)
  
  try:
      with open(file_path, 'w') as f:
          json.dump(convergence_history, f, indent=4)
      print(f"\nConvergence history saved to: {file_path}")
  except Exception as e:
      print(f"\nError saving convergence history: {str(e)}")