#plotting.py Version 1.0 Beta
import matplotlib.pyplot as plt
import numpy as np
import os
import config

def setup_plot_style():
  """Set up plotting style with error handling"""
  try:
      plt.style.use(config.PLOT_STYLE)
  except:
      # If specified style fails, fall back to default
      plt.style.use('default')
  
def plot_takeoff_trajectory(time_history, chord):
  plt.figure(figsize=config.PLOT_FIGSIZE)
  setup_plot_style()
  
  # Plot distance vs velocity
  plt.subplot(2, 1, 1)
  plt.plot(time_history['distance'], time_history['velocity'], 
           color=config.PLOT_COLORS['primary'], 
           label='Velocity vs Distance')
  plt.grid(True, color=config.PLOT_COLORS['grid'])
  plt.xlabel('Distance (ft)')
  plt.ylabel('Velocity (ft/s)')
  plt.title(f'Takeoff Performance (Chord = {chord:.2f} ft)')
  plt.legend()
  
  # Plot time history
  plt.subplot(2, 1, 2)
  plt.plot(time_history['time'], time_history['distance'], 
           color=config.PLOT_COLORS['secondary'], 
           label='Distance vs Time')
  plt.grid(True, color=config.PLOT_COLORS['grid'])
  plt.xlabel('Time (s)')
  plt.ylabel('Distance (ft)')
  plt.legend()
  
  plt.tight_layout()
  
  if config.SAVE_PLOTS:
      filename = f'takeoff_chord_{chord:.2f}.png'
      filepath = os.path.join(config.PLOT_DIR, filename)
      plt.savefig(filepath, dpi=config.PLOT_DPI)
      if config.DEBUG_MODE:
          print(f"Saved trajectory plot to {filepath}")
  plt.close()

def plot_optimization_results(results):
  plt.figure(figsize=config.PLOT_FIGSIZE)
  setup_plot_style()
  
  chords = [r['chord'] for r in results]
  CLs = [r['CL'] for r in results]
  thrusts = [r['thrust'] for r in results]
  
  # Plot CL vs chord
  plt.subplot(2, 1, 1)
  plt.plot(chords, CLs, color=config.PLOT_COLORS['primary'], 
           marker='o', label='CL vs Chord')
  plt.grid(True, color=config.PLOT_COLORS['grid'])
  plt.xlabel('Chord (ft)')
  plt.ylabel('Lift Coefficient (CL)')
  plt.title('Optimization Results')
  plt.legend()
  
  # Plot thrust vs chord
  plt.subplot(2, 1, 2)
  plt.plot(chords, thrusts, color=config.PLOT_COLORS['secondary'], 
           marker='o', label='Thrust vs Chord')
  plt.grid(True, color=config.PLOT_COLORS['grid'])
  plt.xlabel('Chord (ft)')
  plt.ylabel('Thrust (lbf)')
  plt.legend()
  
  plt.tight_layout()
  
  if config.SAVE_PLOTS:
      filepath = os.path.join(config.PLOT_DIR, 'optimization_results.png')
      plt.savefig(filepath, dpi=config.PLOT_DPI)
      if config.DEBUG_MODE:
          print(f"Saved optimization results plot to {filepath}")
  plt.close()

# plotting.py
def plot_optimization_grid(optimization_data, best_CL, best_T):
  """
  Create visualization of the optimization grid search process.
  
  Args:
      optimization_data: Dictionary containing grid search results
      best_CL: Final optimized CL value
      best_T: Final optimized thrust value
  """
  plt.figure(figsize=(15, 6))
  setup_plot_style()
  
  # Create subplots for coarse and fine grid searches
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
  
  # Plot coarse grid search
  coarse_CL = np.array(optimization_data['coarse']['CL'])
  coarse_T = np.array(optimization_data['coarse']['T'])
  coarse_errors = np.array(optimization_data['coarse']['errors'])
  
  # Reshape data for contour plot
  unique_CL = np.unique(coarse_CL)
  unique_T = np.unique(coarse_T)
  Z_coarse = coarse_errors.reshape(len(unique_CL), len(unique_T))
  
  # Create contour plot for coarse grid
  contour1 = ax1.contour(unique_T, unique_CL, Z_coarse, levels=20, cmap='viridis')
  ax1.contourf(unique_T, unique_CL, Z_coarse, levels=20, cmap='viridis', alpha=0.7)
  plt.colorbar(contour1, ax=ax1, label='Error')
  
  # Plot fine grid search
  fine_CL = np.array(optimization_data['fine']['CL'])
  fine_T = np.array(optimization_data['fine']['T'])
  fine_errors = np.array(optimization_data['fine']['errors'])
  
  # Reshape data for contour plot
  unique_CL_fine = np.unique(fine_CL)
  unique_T_fine = np.unique(fine_T)
  Z_fine = fine_errors.reshape(len(unique_CL_fine), len(unique_T_fine))
  
  # Create contour plot for fine grid
  contour2 = ax2.contour(unique_T_fine, unique_CL_fine, Z_fine, levels=20, cmap='viridis')
  ax2.contourf(unique_T_fine, unique_CL_fine, Z_fine, levels=20, cmap='viridis', alpha=0.7)
  plt.colorbar(contour2, ax=ax2, label='Error')
  
  # Plot optimal point
  ax1.plot(best_T, best_CL, 'r*', markersize=15, label='Optimal Point')
  ax2.plot(best_T, best_CL, 'r*', markersize=15, label='Optimal Point')
  
  # Set labels and titles
  ax1.set_xlabel('Thrust (lbf)')
  ax1.set_ylabel('Lift Coefficient (CL)')
  ax1.set_title('Coarse Grid Search')
  ax1.legend()
  
  ax2.set_xlabel('Thrust (lbf)')
  ax2.set_ylabel('Lift Coefficient (CL)')
  ax2.set_title('Fine Grid Search')
  ax2.legend()
  
  plt.tight_layout()
  
  if config.SAVE_PLOTS:
      filepath = os.path.join(config.PLOT_DIR, 'optimization_grid.png')
      plt.savefig(filepath, dpi=config.PLOT_DPI)
      if config.DEBUG_MODE:
          print(f"Saved optimization grid plot to {filepath}")
  
  plt.close()