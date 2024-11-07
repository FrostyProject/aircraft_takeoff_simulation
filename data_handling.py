"""Data handling and CSV operations."""

import csv
from datetime import datetime
import os

def ensure_output_directory():
  """Create output directory if it doesn't exist."""
  if not os.path.exists('output'):
      os.makedirs('output')

def save_data_to_csv(data_dict, filename):
  """Save dictionary data to CSV file."""
  ensure_output_directory()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  full_filename = f"output/{filename}_{timestamp}.csv"
  
  with open(full_filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(data_dict.keys())
      writer.writerow(data_dict.values())
  return full_filename

def save_time_series_to_csv(data_dict, filename):
  """Save time series data to CSV file."""
  ensure_output_directory()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  full_filename = f"output/{filename}_{timestamp}.csv"
  
  with open(full_filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(data_dict.keys())
      for i in range(len(next(iter(data_dict.values())))):
          row = [data_dict[key][i] for key in data_dict.keys()]
          writer.writerow(row)
  return full_filename

def save_convergence_data(iteration_history):
  """Save optimization convergence data to CSV."""
  convergence_data = {
      'Iteration': [x['iteration'] for x in iteration_history],
      'Start_CL': [x['start_CL'] for x in iteration_history],
      'CL': [x['CL'] for x in iteration_history],
      'Thrust(lbf)': [x['T'] for x in iteration_history],
      'Error': [x['error'] for x in iteration_history]
  }
  return save_time_series_to_csv(convergence_data, 'optimization_convergence')