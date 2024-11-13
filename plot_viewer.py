from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                          QLabel, QScrollArea, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import matplotlib.pyplot as plt
from pathlib import Path

class PlotViewer(QWidget):
  """Enhanced plot viewer widget"""
  
  def __init__(self):
      super().__init__()
      self.plots_history = []
      self.current_plot_index = 0
      self.setup_ui()

  def setup_ui(self):
      """Setup the plot viewer interface"""
      layout = QVBoxLayout(self)
      
      # Toolbar
      toolbar = QHBoxLayout()
      
      self.prev_btn = QPushButton("Previous")
      self.prev_btn.clicked.connect(self.show_previous_plot)
      toolbar.addWidget(self.prev_btn)
      
      self.next_btn = QPushButton("Next")
      self.next_btn.clicked.connect(self.show_next_plot)
      toolbar.addWidget(self.next_btn)
      
      self.clear_btn = QPushButton("Clear")
      self.clear_btn.clicked.connect(self.clear_plots)
      toolbar.addWidget(self.clear_btn)
      
      layout.addLayout(toolbar)
      
      # Plot display area
      self.scroll = QScrollArea()
      self.scroll.setWidgetResizable(True)
      self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
      
      self.plot_container = QWidget()
      self.plot_layout = QGridLayout(self.plot_container)
      self.scroll.setWidget(self.plot_container)
      
      layout.addWidget(self.scroll)
      
      self.update_navigation_buttons()

  def update_plots(self, plot_files: list):
      """Update displayed plots"""
      # Save plot files to history
      self.plots_history.extend(plot_files)
      self.current_plot_index = len(self.plots_history) - 1
      
      self.display_current_plot()
      self.update_navigation_buttons()

  def display_current_plot(self):
      """Display the current plot"""
      # Clear existing plots
      for i in reversed(range(self.plot_layout.count())): 
          self.plot_layout.itemAt(i).widget().setParent(None)

      if self.plots_history:
          plot_widget = QLabel()
          pixmap = QPixmap(self.plots_history[self.current_plot_index])
          scaled_pixmap = pixmap.scaled(
              800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
          plot_widget.setPixmap(scaled_pixmap)
          self.plot_layout.addWidget(plot_widget, 0, 0)

  def show_previous_plot(self):
      """Show previous plot in history"""
      if self.current_plot_index > 0:
          self.current_plot_index -= 1
          self.display_current_plot()
          self.update_navigation_buttons()

  def show_next_plot(self):
      """Show next plot in history"""
      if self.current_plot_index < len(self.plots_history) - 1:
          self.current_plot_index += 1
          self.display_current_plot()
          self.update_navigation_buttons()

  def clear_plots(self):
      """Clear all plots"""
      self.plots_history = []
      self.current_plot_index = 0
      self.display_current_plot()
      self.update_navigation_buttons()

  def update_navigation_buttons(self):
      """Update navigation button states"""
      self.prev_btn.setEnabled(self.current_plot_index > 0)
      self.next_btn.setEnabled(
          self.current_plot_index < len(self.plots_history) - 1)