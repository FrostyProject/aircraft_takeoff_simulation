#plotting.py
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