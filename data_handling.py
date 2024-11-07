import pandas as pd
import os
import config

def save_results_to_csv(results):
  data = []
  for result in results:
      # Extract final values from time history
      final_distance = result['time_history']['distance'][-1]
      final_velocity = result['time_history']['velocity'][-1]
      takeoff_time = result['time_history']['time'][-1]
      
      # Create data entry
      data_entry = {
          'chord': result['chord'],
          'CL': result['CL'],
          'CD': result['CD'],
          'L/D_ratio': result['LD_ratio'],
          'thrust': result['thrust'],
          'final_distance': final_distance,
          'final_velocity': final_velocity,
          'takeoff_time': takeoff_time
      }
      
      # Add time history data if configured
      if config.SAVE_PLOTS:
          for t, d, v in zip(result['time_history']['time'],
                           result['time_history']['distance'],
                           result['time_history']['velocity']):
              data_entry.update({
                  f'time_{t:.1f}': t,
                  f'distance_{t:.1f}': d,
                  f'velocity_{t:.1f}': v
              })
      
      data.append(data_entry)
  
  # Create DataFrame and save to CSV
  df = pd.DataFrame(data)
  output_path = os.path.join(config.DATA_DIR, 'takeoff_results.csv')
  df.to_csv(output_path, index=False)
  
  if config.DEBUG_MODE:
      print(f"Results saved to {output_path}")
      print(f"Saved data for {len(results)} configurations")