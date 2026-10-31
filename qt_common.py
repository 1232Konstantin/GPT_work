"""Qt presentation utilities. The OCR, editing and duplicate algorithms remain offline."""
from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal, Qt
from PySide6.QtGui import QImage, QPixmap


STYLE = """
QMainWindow,QDialog,QWidget { background: #121821; color: #e7edf6; font-family: 'Segoe UI'; font-size: 13px; }
QToolBar { background: #1b2430; border: none; border-bottom: 1px solid #2c3847; spacing: 6px; padding: 10px; }
QToolBar QToolButton { background: #263241; border: 1px solid transparent; border-radius: 8px; padding: 8px 12px; font-weight: 550; }
QToolBar QToolButton:hover { background: #34475b; }
QToolBar QToolButton:pressed { background: #465f75; }
QPushButton { background: #283749; border: 1px solid #3a4d60; padding: 9px 12px; border-radius: 8px; }
QPushButton:hover { background: #35485e; border-color: #54708b; }
QPushButton:pressed { background: #405771; }
QPushButton:disabled { color: #8290a0; background: #1f2935; border-color: #2d3947; }
QPushButton[primary="true"] { background: #317ac6; border-color: #4c9be5; color: #ffffff; font-weight: 600; }
QPushButton[primary="true"]:hover { background: #418ed9; }
QListWidget,QTreeWidget,QPlainTextEdit,QLineEdit,QComboBox,QSpinBox { background: #17212c; color: #e7edf6; border: 1px solid #334354; border-radius: 7px; padding: 5px; selection-background-color: #326da5; }
QListWidget::item { border-radius: 6px; padding: 5px; }
QListWidget::item:selected { background: #28517b; color: #fff; }
QCheckBox,QRadioButton { spacing: 7px; padding: 4px; }
QCheckBox::indicator { width: 17px; height: 17px; }
QGroupBox { border: 1px solid #344455; border-radius: 10px; margin-top: 17px; padding: 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #a9cae9; }
QSplitter::handle { background: #2b3744; width: 4px; }
QScrollArea { border: none; background: #121821; }
QScrollBar:vertical { background: #18222d; width: 11px; }
QScrollBar::handle:vertical { background: #405367; border-radius: 5px; min-height: 24px; }
QScrollBar:horizontal { background: #18222d; height: 11px; }
QScrollBar::handle:horizontal { background: #405367; border-radius: 5px; min-width: 24px; }
QStatusBar { background: #1b2430; color: #aebdce; padding: 3px; }
QLabel[muted="true"] { color: #a4b2c4; }
QProgressBar { color: white; background: #253443; border: none; border-radius: 5px; text-align: center; height: 12px; }
QProgressBar::chunk { background: #3c9ed3; border-radius: 5px; }
"""


def pixmap_from_pil(image: Image.Image) -> QPixmap:
    rgba = image.convert('RGBA')
    raw = rgba.tobytes('raw', 'RGBA')
    # Copy before the Python bytes buffer is released.
    qt = QImage(raw, rgba.width, rgba.height, rgba.width * 4, QImage.Format.Format_RGBA8888).copy()
    return QPixmap.fromImage(qt)


class TaskSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(object)


class Task(QRunnable):
    """Pure background calculation; callbacks always reach UI via Qt queued signals."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.signals = TaskSignals()
        self.setAutoDelete(False)

    def run(self):
        try:
            self.signals.result.emit(self.fn(self.signals.progress.emit))
        except Exception as exc:
            self.signals.error.emit(f'{type(exc).__name__}: {exc}')

# Compact control bar for the picture-first Win7-style viewer.
STYLE += """
QMainWindow QToolBar { background: #1d252f; border: none; border-top: 1px solid #39414e;
 padding: 3px; spacing: 2px; }
QMainWindow QToolBar QToolButton { padding: 5px 8px; margin: 1px; border-radius: 5px; }
QGraphicsView { background: #111820; border: none; }
QPlainTextEdit { selection-background-color: #376f9c; }
"""