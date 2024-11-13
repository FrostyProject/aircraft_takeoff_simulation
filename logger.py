import logging
from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt, pyqtSignal

class QTextEditLogger(logging.Handler):
  """Custom logger that writes to QTextEdit"""
  
  append_signal = pyqtSignal(str)

  def __init__(self, widget: QTextEdit):
      super().__init__()
      self.widget = widget
      self.append_signal.connect(self.widget.append)
      self.setFormatter(logging.Formatter(
          '%(asctime)s - %(levelname)s - %(message)s'))

  def emit(self, record):
      msg = self.format(record)
      self.append_signal.emit(msg)