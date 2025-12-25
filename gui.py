from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import json

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import correlate

from main import BOTDASweep, RFParams, SweepSettings
from plotter import save_traces


# ======================================================
# --- Lorentzian Fitting Functions ---
# ======================================================
def lorentzian(f, f0, gamma, A):
    """Normalized Lorentzian function."""
    return A * (0.5 * gamma) ** 2 / ((f - f0) ** 2 + (0.5 * gamma) ** 2)


def fit_trace_xcorr(freqs, trace, gamma=0.015, normalize=True):
    """
    Estimate peak frequency (f0) by cross-correlating the BOTDA trace with
    a Lorentzian reference kernel.
    """
    y = trace - np.min(trace)
    if normalize:
        y = y / np.max(y)

    f_ref = np.linspace(-0.1, 0.1, len(freqs))
    ref_lor = lorentzian(f_ref, 0, gamma, 1.0)
    ref_lor = ref_lor / np.max(ref_lor)

    corr = correlate(y, ref_lor, mode='same')
    peak_idx = np.argmax(corr)
    f0_est = freqs[peak_idx]

    return f0_est


def process_all_traces_xcorr(processed_traces, freqs, spacing, gamma):
    """
    Fits all traces sequentially using cross-correlation.
    Returns distance array and estimated peak frequencies.
    """
    n_traces = processed_traces.shape[1]
    dist = np.arange(n_traces) * spacing
    P = []

    for i in range(n_traces):
        trace = processed_traces[:, i]
        f0_est = fit_trace_xcorr(freqs, trace, gamma=gamma)
        P.append(f0_est)

    return dist, np.array(P)


class SweepWorker(QtCore.QThread):
    status = QtCore.Signal(str)
    progress = QtCore.Signal(int, int)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, rf_params: RFParams, sweep_settings: SweepSettings, personal_comments: str = "") -> None:
        super().__init__()
        self.rf_params = rf_params
        self.sweep_settings = sweep_settings
        self.personal_comments = personal_comments

    def run(self) -> None:
        try:
            sweep = BOTDASweep(self.rf_params, self.sweep_settings)
            results = sweep.run(
                status_callback=self.status.emit,
                progress_callback=self.progress.emit,
                plot=False,
                personal_comments=self.personal_comments,
            )
            self.finished.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class PlotCanvas(FigureCanvas):
    """Single plot canvas."""
    
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        self.figure = Figure(figsize=(6, 4))
        super().__init__(self.figure)
        self.setParent(parent)

    def plot_3d(self, traces: np.ndarray, freqs: np.ndarray, distance_scale_m: float = 10) -> None:
        """Plot 3D BGS surface."""
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection="3d")
        
        n_steps, n_samples = traces.shape
        distance_m = np.arange(n_samples) * distance_scale_m
        D, F = np.meshgrid(distance_m, freqs)

        ax.plot_surface(D, F, traces, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Frequency (GHz)")
        ax.set_zlabel("Amplitude")
        ax.set_title("3D BGS (Brillouin Gain Spectrum)")

        self.figure.tight_layout()
        self.draw()

    def plot_peak_fit(self, dist: np.ndarray, peaks: np.ndarray) -> None:
        """Plot peak frequency vs distance."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.plot(dist, peaks, 'r.-', markersize=3, linewidth=1, label="Fitted Peak Frequency")
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Peak Frequency (GHz)")
        ax.set_title("BGS Peak Frequency vs Distance")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.draw()

    def plot_slice(self, freqs: np.ndarray, trace: np.ndarray, f0: float, 
                   gamma: float, actual_distance: float) -> None:
        """Plot measured trace and Lorentzian fit at specific distance."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        A = np.max(trace) - np.min(trace)
        c0 = np.min(trace)
        fit_curve = c0 + lorentzian(freqs, f0, gamma, A)

        ax.plot(freqs, trace, 'k-', lw=1.5, label=f"Measured Trace ({actual_distance:.1f} m)")
        ax.plot(freqs, fit_curve, 'r--', lw=2, label=f"Lorentzian Fit (f₀={f0:.4f} GHz)")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.set_title(f"BGS Lorentzian Fit at {actual_distance:.1f} m")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.draw()


class BotdaGui(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DSS BOTDA - Developer Edition")

        # Set application icon
        icon_path = Path(__file__).parent / "dss_icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        # Set minimum size to ensure usability on small screens
        self.setMinimumSize(1024, 768)

        # Start maximized for better screen adaptation
        self.showMaximized()

        self._worker: Optional[SweepWorker] = None
        self._last_results: Optional[Dict[str, Any]] = None
        self._fitted_data: Optional[Dict[str, Any]] = None
        self._config_presets: Dict[str, Dict] = {}
        self._current_theme = "light"  # Default theme
        self._load_presets()

        # Create menu bar first (before applying theme)
        self._create_menu_bar()

        # Apply default theme
        self._apply_theme(self._current_theme)

        # Create toolbar
        self._create_toolbar()

        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        tabs = QtWidgets.QTabWidget()
        tabs.setDocumentMode(True)
        main_layout.addWidget(tabs)

        self._init_config_tab(tabs)
        self._init_live_tab(tabs)
        self._init_results_tab(tabs)
        self._init_slice_tab(tabs)

        # Create status bar
        self._create_status_bar()

    def _apply_theme(self, theme: str) -> None:
        """Apply light or dark theme."""
        self._current_theme = theme
        
        if theme == "dark":
            # Dark theme colors
            bg_main = "#2b2b2b"
            bg_widget = "#3c3f41"
            bg_input = "#45494a"
            text_main = "#ffffff"
            text_secondary = "#bbbbbb"
            border_color = "#555555"
            accent_color = "#4a9eff"
            hover_color = "#3d6a99"
            header_bg = "#1e1e1e"
            menu_bg = "#2b2b2b"
            
            self.setStyleSheet(f"""
                QMainWindow {{
                    background-color: {bg_main};
                    color: {text_main};
                }}
                QWidget {{
                    color: {text_main};
                }}
                QTabWidget::pane {{
                    border: 1px solid {border_color};
                    background-color: {bg_widget};
                    border-radius: 3px;
                }}
                QTabBar::tab {{
                    background-color: {bg_input};
                    color: {text_main};
                    border: 1px solid {border_color};
                    padding: 8px 20px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }}
                QTabBar::tab:selected {{
                    background-color: {bg_widget};
                    border-bottom-color: {bg_widget};
                    font-weight: bold;
                }}
                QTabBar::tab:hover:!selected {{
                    background-color: {hover_color};
                }}
                QGroupBox {{
                    border: 2px solid {border_color};
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                    background-color: {bg_widget};
                    color: {text_main};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: {accent_color};
                }}
                QLabel {{
                    color: {text_main};
                }}
                QPushButton {{
                    background-color: {accent_color};
                    color: {text_main};
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QPushButton:pressed {{
                    background-color: #2d5a88;
                }}
                QPushButton:disabled {{
                    background-color: {border_color};
                    color: {text_secondary};
                }}
                QLineEdit, QSpinBox, QDoubleSpinBox {{
                    border: 1px solid {border_color};
                    border-radius: 3px;
                    padding: 5px;
                    background-color: {bg_input};
                    color: {text_main};
                    max-width: 450px;
                }}
                QSpinBox::up-button, QDoubleSpinBox::up-button {{
                    width: 20px;
                    height: 14px;
                }}
                QSpinBox::down-button, QDoubleSpinBox::down-button {{
                    width: 20px;
                    height: 14px;
                }}
                QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                    width: 12px;
                    height: 12px;
                }}
                QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                    width: 12px;
                    height: 12px;
                }}
                QTextEdit {{
                    border: 1px solid {border_color};
                    border-radius: 3px;
                    padding: 5px;
                    background-color: {bg_input};
                    color: {text_main};
                }}
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
                    border: 2px solid {accent_color};
                }}
                QComboBox {{
                    border: 1px solid {border_color};
                    border-radius: 3px;
                    padding: 5px;
                    background-color: {bg_input};
                    color: {text_main};
                }}
                QComboBox::drop-down {{
                    border: none;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {bg_input};
                    color: {text_main};
                    selection-background-color: {accent_color};
                }}
                QProgressBar {{
                    border: 1px solid {border_color};
                    border-radius: 3px;
                    text-align: center;
                    background-color: {bg_input};
                    color: {text_main};
                }}
                QProgressBar::chunk {{
                    background-color: #27ae60;
                    border-radius: 2px;
                }}
                QMessageBox {{
                    background-color: {bg_widget};
                    color: {text_main};
                }}
                QMessageBox QLabel {{
                    color: {text_main};
                }}
                QMessageBox QPushButton {{
                    background-color: {accent_color};
                    color: {text_main};
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }}
                QMessageBox QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QDialog {{
                    background-color: {bg_widget};
                    color: {text_main};
                }}
                QDialog QLabel {{
                    color: {text_main};
                }}
                QListWidget {{
                    background-color: {bg_input};
                    color: {text_main};
                    border: 1px solid {border_color};
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: {text_main};
                }}
                QInputDialog {{
                    background-color: {bg_widget};
                    color: {text_main};
                }}
            """)

            # Dark menu bar styling
            self.menuBar().setStyleSheet(f"""
                QMenuBar {{
                    background-color: {header_bg};
                    color: {text_main};
                    padding: 4px;
                }}
                QMenuBar::item {{
                    background-color: transparent;
                    padding: 6px 12px;
                    color: {text_main};
                }}
                QMenuBar::item:selected {{
                    background-color: {hover_color};
                }}
                QMenu {{
                    background-color: {menu_bg};
                    color: {text_main};
                    border: 1px solid {border_color};
                }}
                QMenu::item {{
                    padding: 6px 30px;
                    color: {text_main};
                }}
                QMenu::item:selected {{
                    background-color: {accent_color};
                    color: {text_main};
                }}
            """)
            
            # Dark status bar
            self.statusBar().setStyleSheet(f"""
                QStatusBar {{
                    background-color: {header_bg};
                    color: {text_main};
                    border-top: 2px solid {border_color};
                }}
                QLabel {{
                    color: {text_main};
                    padding: 2px 10px;
                }}
            """)

            # Dark toolbar styling
            if hasattr(self, 'toolbar'):
                self.toolbar.setStyleSheet(f"""
                    QToolBar {{
                        background-color: {bg_widget};
                        border-bottom: 2px solid {border_color};
                        spacing: 10px;
                        padding: 5px;
                    }}
                    QToolButton {{
                        background-color: transparent;
                        border: none;
                        padding: 5px;
                        color: {text_main};
                    }}
                    QToolButton:hover {{
                        background-color: {hover_color};
                        border-radius: 3px;
                    }}
                """)
            
        else:  # Light theme
            # Light theme colors
            bg_main = "#f5f5f5"
            bg_widget = "#ffffff"
            bg_input = "#ffffff"
            text_main = "#2c3e50"
            text_secondary = "#7f8c8d"
            border_color = "#cccccc"
            accent_color = "#3498db"
            hover_color = "#2980b9"
            header_bg = "#2c3e50"
            
            self.setStyleSheet(f"""
                QMainWindow {{
                    background-color: {bg_main};
                    color: {text_main};
                }}
                QWidget {{
                    color: {text_main};
                }}
                QTabWidget::pane {{
                    border: 1px solid {border_color};
                    background-color: {bg_widget};
                    border-radius: 3px;
                }}
                QTabBar::tab {{
                    background-color: #e0e0e0;
                    color: {text_main};
                    border: 1px solid {border_color};
                    padding: 8px 20px;
                    margin-right: 2px;
                    border-top-left-radius: 4px;
                    border-top-right-radius: 4px;
                }}
                QTabBar::tab:selected {{
                    background-color: {bg_widget};
                    border-bottom-color: {bg_widget};
                    font-weight: bold;
                }}
                QTabBar::tab:hover:!selected {{
                    background-color: #eeeeee;
                }}
                QGroupBox {{
                    border: 2px solid #d0d0d0;
                    border-radius: 5px;
                    margin-top: 10px;
                    font-weight: bold;
                    background-color: {bg_widget};
                    color: {text_main};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                    color: {text_main};
                }}
                QLabel {{
                    color: {text_main};
                }}
                QPushButton {{
                    background-color: {accent_color};
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }}
                QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QPushButton:pressed {{
                    background-color: #1f6391;
                }}
                QPushButton:disabled {{
                    background-color: #bdc3c7;
                    color: white;
                }}
                QLineEdit, QSpinBox, QDoubleSpinBox {{
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    padding: 5px;
                    background-color: {bg_input};
                    color: {text_main};
                    max-width: 450px;
                }}
                QSpinBox::up-button, QDoubleSpinBox::up-button {{
                    width: 20px;
                    height: 14px;
                }}
                QSpinBox::down-button, QDoubleSpinBox::down-button {{
                    width: 20px;
                    height: 14px;
                }}
                QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
                    width: 12px;
                    height: 12px;
                }}
                QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
                    width: 12px;
                    height: 12px;
                }}
                QTextEdit {{
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    padding: 5px;
                    background-color: {bg_input};
                    color: {text_main};
                }}
                QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
                    border: 2px solid {accent_color};
                }}
                QComboBox {{
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    padding: 5px;
                    background-color: {bg_input};
                    color: {text_main};
                }}
                QComboBox::drop-down {{
                    border: none;
                }}
                QComboBox QAbstractItemView {{
                    background-color: {bg_input};
                    color: {text_main};
                    selection-background-color: {accent_color};
                }}
                QProgressBar {{
                    border: 1px solid #bdc3c7;
                    border-radius: 3px;
                    text-align: center;
                    background-color: #ecf0f1;
                    color: {text_main};
                }}
                QProgressBar::chunk {{
                    background-color: #27ae60;
                    border-radius: 2px;
                }}
                QMessageBox {{
                    background-color: {bg_widget};
                    color: {text_main};
                }}
                QMessageBox QLabel {{
                    color: {text_main};
                }}
                QMessageBox QPushButton {{
                    background-color: {accent_color};
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                    min-width: 80px;
                }}
                QMessageBox QPushButton:hover {{
                    background-color: {hover_color};
                }}
                QDialog {{
                    background-color: {bg_widget};
                    color: {text_main};
                }}
                QDialog QLabel {{
                    color: {text_main};
                }}
                QListWidget {{
                    background-color: {bg_input};
                    color: {text_main};
                    border: 1px solid #bdc3c7;
                }}
                QListWidget::item:selected {{
                    background-color: {accent_color};
                    color: white;
                }}
                QInputDialog {{
                    background-color: {bg_widget};
                    color: {text_main};
                }}
            """)

            # Light menu bar styling
            self.menuBar().setStyleSheet(f"""
                QMenuBar {{
                    background-color: {header_bg};
                    color: white;
                    padding: 4px;
                }}
                QMenuBar::item {{
                    background-color: transparent;
                    padding: 6px 12px;
                    color: white;
                }}
                QMenuBar::item:selected {{
                    background-color: #34495e;
                }}
                QMenu {{
                    background-color: white;
                    color: {text_main};
                    border: 1px solid #bdc3c7;
                }}
                QMenu::item {{
                    padding: 6px 30px;
                    color: {text_main};
                }}
                QMenu::item:selected {{
                    background-color: {accent_color};
                    color: white;
                }}
            """)
            
            # Light status bar
            self.statusBar().setStyleSheet(f"""
                QStatusBar {{
                    background-color: {header_bg};
                    color: white;
                    border-top: 2px solid #1a252f;
                }}
                QLabel {{
                    color: white;
                    padding: 2px 10px;
                }}
            """)

            # Light toolbar styling
            if hasattr(self, 'toolbar'):
                self.toolbar.setStyleSheet(f"""
                    QToolBar {{
                        background-color: #ecf0f1;
                        border-bottom: 2px solid #bdc3c7;
                        spacing: 10px;
                        padding: 5px;
                    }}
                    QToolButton {{
                        background-color: transparent;
                        border: none;
                        padding: 5px;
                        color: {text_main};
                    }}
                    QToolButton:hover {{
                        background-color: #d5d8dc;
                        border-radius: 3px;
                    }}
                """)

    def _create_menu_bar(self) -> None:
        """Create professional menu bar."""
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #2c3e50;
                color: white;
                padding: 4px;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 6px 12px;
            }
            QMenuBar::item:selected {
                background-color: #34495e;
            }
            QMenu {
                background-color: white;
                border: 1px solid #bdc3c7;
            }
            QMenu::item {
                padding: 6px 30px;
            }
            QMenu::item:selected {
                background-color: #3498db;
                color: white;
            }
        """)

        # File Menu
        file_menu = menubar.addMenu("&File")
        
        new_action = QtGui.QAction("&New Session", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new_session)
        file_menu.addAction(new_action)

        load_action = QtGui.QAction("&Load Data...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_data)
        file_menu.addAction(load_action)

        save_action = QtGui.QAction("&Save Data...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_data)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("Export")
        export_csv = QtGui.QAction("Export as CSV", self)
        export_csv.triggered.connect(lambda: self._export_data("csv"))
        export_menu.addAction(export_csv)

        export_npz = QtGui.QAction("Export as NPZ", self)
        export_npz.triggered.connect(lambda: self._export_data("npz"))
        export_menu.addAction(export_npz)

        file_menu.addSeparator()

        exit_action = QtGui.QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings Menu
        settings_menu = menubar.addMenu("&Settings")
        
        # Theme submenu
        theme_menu = settings_menu.addMenu("🎨 Theme")
        
        self.light_theme_action = QtGui.QAction("☀ Light Theme", self)
        self.light_theme_action.setCheckable(True)
        self.light_theme_action.setChecked(True)
        self.light_theme_action.triggered.connect(lambda: self._change_theme("light"))
        theme_menu.addAction(self.light_theme_action)
        
        self.dark_theme_action = QtGui.QAction("🌙 Dark Theme", self)
        self.dark_theme_action.setCheckable(True)
        self.dark_theme_action.triggered.connect(lambda: self._change_theme("dark"))
        theme_menu.addAction(self.dark_theme_action)
        
        settings_menu.addSeparator()
        
        presets_action = QtGui.QAction("Manage &Presets", self)
        presets_action.triggered.connect(self._manage_presets)
        settings_menu.addAction(presets_action)

        save_preset_action = QtGui.QAction("&Save Current as Preset", self)
        save_preset_action.setShortcut("Ctrl+Shift+S")
        save_preset_action.triggered.connect(self._save_current_preset)
        settings_menu.addAction(save_preset_action)

        # Tools Menu
        tools_menu = menubar.addMenu("&Tools")
        
        compare_action = QtGui.QAction("&Compare Measurements", self)
        compare_action.triggered.connect(self._compare_measurements)
        tools_menu.addAction(compare_action)

        # Help Menu
        help_menu = menubar.addMenu("&Help")
        
        docs_action = QtGui.QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.triggered.connect(self._show_documentation)
        help_menu.addAction(docs_action)

        about_action = QtGui.QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _create_toolbar(self) -> None:
        """Create toolbar with quick actions."""
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setMovable(False)

        # Quick actions
        run_action = QtGui.QAction("▶ Run", self)
        run_action.triggered.connect(self._run_sweep)
        self.toolbar.addAction(run_action)

        self.toolbar.addSeparator()

        save_action = QtGui.QAction("💾 Save", self)
        save_action.triggered.connect(self._save_data)
        self.toolbar.addAction(save_action)

        self.toolbar.addSeparator()

        # Preset selector
        self.toolbar.addWidget(QtWidgets.QLabel("Preset:"))
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItem("Default")
        self.preset_combo.currentTextChanged.connect(self._load_preset)
        self.toolbar.addWidget(self.preset_combo)

    def _create_status_bar(self) -> None:
        """Create informative status bar."""
        self.statusBar().setStyleSheet("""
            QStatusBar {
                background-color: #2c3e50;
                color: white;
                border-top: 2px solid #1a252f;
            }
            QLabel {
                color: white;
                padding: 2px 10px;
            }
        """)

        # Connection status indicators
        self.rf_status_label = QtWidgets.QLabel("RF: ●")
        self.rf_status_label.setStyleSheet("QLabel { color: #e74c3c; }")
        self.statusBar().addPermanentWidget(self.rf_status_label)

        self.adq_status_label = QtWidgets.QLabel("ADQ: ●")
        self.adq_status_label.setStyleSheet("QLabel { color: #e74c3c; }")
        self.statusBar().addPermanentWidget(self.adq_status_label)

        # Info label
        self.status_info = QtWidgets.QLabel("Ready")
        self.statusBar().addWidget(self.status_info)

    def _init_config_tab(self, tabs: QtWidgets.QTabWidget) -> None:
        tab = QtWidgets.QWidget()
        tabs.addTab(tab, "⚙ Configuration")
        layout = QtWidgets.QGridLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # RF Sweep Settings
        rf_group = QtWidgets.QGroupBox("RF Sweep Settings")
        rf_layout = QtWidgets.QFormLayout(rf_group)
        rf_layout.setSpacing(6)
        rf_layout.setContentsMargins(10, 15, 10, 10)
        
        self.start_freq = self._double_box("GHz", 0, 100, 10.7, decimals=3)
        self.start_freq.setToolTip("Starting frequency for RF sweep")
        
        self.stop_freq = self._double_box("GHz", 0, 100, 11.0, decimals=3)
        self.stop_freq.setToolTip("Ending frequency for RF sweep")
        
        self.step_mhz = self._double_box("MHz", 0.1, 1000, 1.0, decimals=3)
        self.step_mhz.setToolTip("Frequency step size")
        
        self.power_dbm = self._double_box("dBm", -50, 20, -8.0, decimals=3)
        self.power_dbm.setToolTip("RF output power")
        
        self.com_port = QtWidgets.QLineEdit("COM19")
        self.com_port.setToolTip("COM port for RF controller")

        rf_layout.addRow("Start Frequency:", self.start_freq)
        rf_layout.addRow("Stop Frequency:", self.stop_freq)
        rf_layout.addRow("Step Size:", self.step_mhz)
        rf_layout.addRow("Power:", self.power_dbm)
        rf_layout.addRow("COM Port:", self.com_port)

        # Acquisition Settings
        acq_group = QtWidgets.QGroupBox("Acquisition Settings")
        acq_layout = QtWidgets.QFormLayout(acq_group)
        acq_layout.setSpacing(6)
        acq_layout.setContentsMargins(10, 15, 10, 10)
        
        self.fiber_len = self._double_box("m", 1, 1e7, 2000, decimals=3)
        self.fiber_len.setToolTip("Total fiber length to measure")
        
        self.dwell_time = self._double_box("s", 0, 60, 10, decimals=3)
        self.dwell_time.setToolTip("Settling time at each frequency")
        
        self.sample_rate = self._double_box("MS/s", 1, 1000, 200, decimals=3)
        self.sample_rate.setToolTip("ADQ sampling rate")
        
        self.averages = self._int_box("", 1, 1000000, 1000)
        self.averages.setToolTip("Number of averages per measurement")
        
        self.pulse_width = self._int_box("ns", 1, 10000, 100)
        self.pulse_width.setToolTip("Pulse width (determines spatial resolution)")
        
        self.save_dir = QtWidgets.QLineEdit(r"D:\Subbash\botda_exp\saved_plots")
        browse_btn = QtWidgets.QPushButton("Browse...")
        browse_btn.clicked.connect(self._choose_save_dir)

        acq_layout.addRow("Fiber Length:", self.fiber_len)
        acq_layout.addRow("Dwell Time:", self.dwell_time)
        acq_layout.addRow("Sample Rate:", self.sample_rate)
        acq_layout.addRow("Averages:", self.averages)
        acq_layout.addRow("Pulse Width:", self.pulse_width)

        save_dir_row = QtWidgets.QHBoxLayout()
        save_dir_row.setSpacing(5)
        save_dir_row.addWidget(self.save_dir)
        save_dir_row.addWidget(browse_btn)
        acq_layout.addRow("Save Directory:", save_dir_row)

        # Fitting parameters
        fitting_group = QtWidgets.QGroupBox("Lorentzian Fitting Parameters")
        fitting_layout = QtWidgets.QFormLayout(fitting_group)
        fitting_layout.setSpacing(6)
        fitting_layout.setContentsMargins(10, 15, 10, 10)
        
        self.gamma = self._double_box("MHz", 1, 1000, 35, decimals=3)
        self.gamma.setToolTip("Expected linewidth for fitting (FWHM)")
        fitting_layout.addRow("Linewidth (γ):", self.gamma)

        # Comments section
        comments_group = QtWidgets.QGroupBox("Pre-Run Comments")
        comments_layout = QtWidgets.QVBoxLayout(comments_group)
        comments_layout.setSpacing(5)
        comments_layout.setContentsMargins(10, 10, 10, 10)
        
        self.pre_run_comments = QtWidgets.QTextEdit()
        self.pre_run_comments.setPlaceholderText("Enter objective, setup details, laser wavelength, pump/probe settings, etc.")
        comments_layout.addWidget(self.pre_run_comments)

        layout.addWidget(rf_group, 0, 0)
        layout.addWidget(acq_group, 0, 1)
        layout.addWidget(fitting_group, 1, 0)
        layout.addWidget(comments_group, 1, 1)

    def _init_live_tab(self, tabs: QtWidgets.QTabWidget) -> None:
        tab = QtWidgets.QWidget()
        tabs.addTab(tab, "▶ Live Run")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_connect_rf = QtWidgets.QPushButton("🔌 Connect RF")
        self.btn_connect_adq = QtWidgets.QPushButton("🔌 Connect ADQ")
        self.btn_run = QtWidgets.QPushButton("▶ Run Sweep")
        self.btn_run.setStyleSheet("QPushButton { background-color: #27ae60; } QPushButton:hover { background-color: #229954; }")
        self.btn_abort = QtWidgets.QPushButton("⏹ Abort")
        self.btn_abort.setStyleSheet("QPushButton { background-color: #e74c3c; } QPushButton:hover { background-color: #c0392b; }")
        self.btn_abort.setEnabled(False)

        btn_layout.addWidget(self.btn_connect_rf)
        btn_layout.addWidget(self.btn_connect_adq)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_abort)
        btn_layout.addStretch()

        self.status_text = QtWidgets.QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setFont(QtGui.QFont("Consolas", 9))

        self.progress = QtWidgets.QProgressBar()
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p% - %v/%m steps")

        layout.addLayout(btn_layout)
        layout.addWidget(QtWidgets.QLabel("Status / Live Output:"))
        layout.addWidget(self.status_text)
        layout.addWidget(self.progress)

        self.btn_connect_rf.clicked.connect(self._connect_rf)
        self.btn_connect_adq.clicked.connect(self._connect_adq)
        self.btn_run.clicked.connect(self._run_sweep)
        self.btn_abort.clicked.connect(self._abort_sweep)

    def _init_results_tab(self, tabs: QtWidgets.QTabWidget) -> None:
        tab = QtWidgets.QWidget()
        tabs.addTab(tab, "📊 Results & Logs")
        main_layout = QtWidgets.QVBoxLayout(tab)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Plots in separate group boxes
        plots_layout = QtWidgets.QHBoxLayout()
        plots_layout.setSpacing(8)
        
        # 3D BGS Plot
        plot_3d_group = QtWidgets.QGroupBox("3D Brillouin Gain Spectrum")
        plot_3d_layout = QtWidgets.QVBoxLayout(plot_3d_group)
        plot_3d_layout.setContentsMargins(5, 15, 5, 5)
        self.plot_canvas_3d = PlotCanvas(self)
        plot_3d_layout.addWidget(self.plot_canvas_3d)
        
        # Peak Frequency Plot
        plot_peak_group = QtWidgets.QGroupBox("Peak Frequency vs Distance")
        plot_peak_layout = QtWidgets.QVBoxLayout(plot_peak_group)
        plot_peak_layout.setContentsMargins(5, 15, 5, 5)
        self.plot_canvas_peak = PlotCanvas(self)
        plot_peak_layout.addWidget(self.plot_canvas_peak)
        
        plots_layout.addWidget(plot_3d_group)
        plots_layout.addWidget(plot_peak_group)
        
        main_layout.addLayout(plots_layout)

        # Log view
        log_label = QtWidgets.QLabel("Log Output:")
        self.log_view = QtWidgets.QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumHeight(150)
        self.log_view.setFont(QtGui.QFont("Consolas", 8))

        # Post-run comments section
        comments_group = QtWidgets.QGroupBox("Post-Run Comments")
        comments_layout = QtWidgets.QVBoxLayout(comments_group)
        comments_layout.setSpacing(5)
        comments_layout.setContentsMargins(10, 10, 10, 10)
        
        self.post_run_comments = QtWidgets.QTextEdit()
        self.post_run_comments.setPlaceholderText("Enter observations, analysis notes, anomalies noticed in the plot, etc.")
        comments_layout.addWidget(self.post_run_comments)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_save_csv = QtWidgets.QPushButton("💾 Save CSV")
        self.btn_open_log = QtWidgets.QPushButton("📄 Open Log")
        self.btn_save_comments = QtWidgets.QPushButton("📝 Save Comments")
        self.btn_save_csv.setEnabled(False)
        self.btn_open_log.setEnabled(False)
        self.btn_save_comments.setEnabled(False)

        btn_layout.addWidget(self.btn_save_csv)
        btn_layout.addWidget(self.btn_open_log)
        btn_layout.addWidget(self.btn_save_comments)
        btn_layout.addStretch()

        main_layout.addWidget(log_label)
        main_layout.addWidget(self.log_view)
        main_layout.addWidget(comments_group)
        main_layout.addLayout(btn_layout)

        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_open_log.clicked.connect(self._open_log)
        self.btn_save_comments.clicked.connect(self._save_post_comments)

    def _init_slice_tab(self, tabs: QtWidgets.QTabWidget) -> None:
        """Initialize the 2D Slice Analysis tab."""
        tab = QtWidgets.QWidget()
        tabs.addTab(tab, "🔍 2D Slice Analysis")
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # Control panel
        control_group = QtWidgets.QGroupBox("Slice Selection")
        control_layout = QtWidgets.QHBoxLayout(control_group)
        control_layout.setSpacing(8)
        control_layout.setContentsMargins(10, 15, 10, 10)
        
        control_layout.addWidget(QtWidgets.QLabel("Distance (m):"))
        self.slice_distance = QtWidgets.QDoubleSpinBox()
        self.slice_distance.setRange(0, 100000)
        self.slice_distance.setValue(0)
        self.slice_distance.setDecimals(3)
        self.slice_distance.setSingleStep(10)
        self.slice_distance.setMinimumWidth(120)
        self.slice_distance.setMaximumWidth(150)
        self.slice_distance.setToolTip("Select distance point for detailed analysis")
        control_layout.addWidget(self.slice_distance)
        
        self.btn_plot_slice = QtWidgets.QPushButton("📈 Plot Slice")
        self.btn_plot_slice.setEnabled(False)
        self.btn_plot_slice.setMaximumWidth(120)
        self.btn_plot_slice.clicked.connect(self._plot_slice)
        control_layout.addWidget(self.btn_plot_slice)
        
        control_layout.addStretch()
        
        layout.addWidget(control_group)

        # Plot area
        plot_group = QtWidgets.QGroupBox("BGS at Selected Distance")
        plot_layout = QtWidgets.QVBoxLayout(plot_group)
        plot_layout.setContentsMargins(5, 15, 5, 5)
        self.plot_canvas_slice = PlotCanvas(self)
        plot_layout.addWidget(self.plot_canvas_slice)
        
        layout.addWidget(plot_group)

        # Info display
        self.slice_info = QtWidgets.QLabel("Run a sweep to enable slice analysis")
        self.slice_info.setStyleSheet("QLabel { color: gray; font-style: italic; padding: 5px; }")
        layout.addWidget(self.slice_info)

    def _double_box(self, suffix: str, minimum: float, maximum: float, value: float, decimals: int = 3) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setSuffix(f" {suffix}" if suffix else "")
        box.setRange(minimum, maximum)
        box.setValue(value)
        box.setDecimals(decimals)
        return box

    def _int_box(self, suffix: str, minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        box = QtWidgets.QSpinBox()
        box.setSuffix(f" {suffix}" if suffix else "")
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    def _choose_save_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if directory:
            self.save_dir.setText(directory)

    def _connect_rf(self) -> None:
        rf_params = self._collect_rf_params()
        self._append_status("Connecting RF controller...")
        try:
            from rf_controller import RFController
            controller = RFController(rf_params.com_port)
            controller.connect()
            controller.disconnect()
            self._append_status("✅ RF controller connection OK.")
            self.rf_status_label.setStyleSheet("QLabel { color: #27ae60; }")
            self.rf_status_label.setText("RF: ✓")
        except Exception as exc:
            self._append_status(f"❌ RF connection failed: {exc}")
            self.rf_status_label.setStyleSheet("QLabel { color: #e74c3c; }")

    def _connect_adq(self) -> None:
        sweep_settings = self._collect_sweep_settings()
        self._append_status("Connecting ADQ controller...")
        try:
            from adq_controller import ADQController
            controller = ADQController(
                BOTDASweep.compute_adq_config(
                    sweep_settings.fiber_len_m,
                    averages=sweep_settings.averages,
                    sample_rate=sweep_settings.sample_rate_hz,
                    pulselength=sweep_settings.pulselength_ns,
                )
            )
            controller.connect()
            controller.close()
            self._append_status("✅ ADQ controller connection OK.")
            self.adq_status_label.setStyleSheet("QLabel { color: #27ae60; }")
            self.adq_status_label.setText("ADQ: ✓")
        except Exception as exc:
            self._append_status(f"❌ ADQ connection failed: {exc}")
            self.adq_status_label.setStyleSheet("QLabel { color: #e74c3c; }")

    def _run_sweep(self) -> None:
        if self._worker and self._worker.isRunning():
            self._append_status("⚠️ Sweep already running.")
            return

        self._last_results = None
        self._fitted_data = None
        self.progress.setValue(0)
        self.btn_abort.setEnabled(True)
        self.btn_run.setEnabled(False)

        rf_params = self._collect_rf_params()
        sweep_settings = self._collect_sweep_settings()
        personal_comments = self.pre_run_comments.toPlainText().strip()

        self._append_status("Starting sweep...")
        self.status_info.setText("Running sweep...")
        
        self._worker = SweepWorker(rf_params, sweep_settings, personal_comments)
        self._worker.status.connect(self._append_status)
        self._worker.progress.connect(self._update_progress)
        self._worker.finished.connect(self._on_sweep_finished)
        self._worker.failed.connect(self._on_sweep_failed)
        self._worker.start()

    def _abort_sweep(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._append_status("⚠️ Sweep terminated by user.")
            self.status_info.setText("Sweep aborted")
            self.btn_abort.setEnabled(False)
            self.btn_run.setEnabled(True)

    def _update_progress(self, current: int, total: int) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(current)
        self.status_info.setText(f"Acquiring: {current}/{total} steps")

    def _on_sweep_finished(self, results: dict) -> None:
        self._last_results = results
        self.btn_abort.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.btn_open_log.setEnabled(True)
        self.btn_save_comments.setEnabled(True)
        self.btn_plot_slice.setEnabled(True)
        self.progress.setValue(self.progress.maximum())

        self._append_status("✅ Sweep completed.")
        self.status_info.setText("Sweep completed - Processing...")

        traces = results["processed_traces"]
        freqs = results["freqs"]
        
        gamma_mhz = self.gamma.value()
        gamma_ghz = gamma_mhz / 1000.0
        
        sample_rate = self.sample_rate.value() * 1e6
        pulse_width_ns = self.pulse_width.value()
        
        group_size = int(pulse_width_ns * sample_rate * 1e-9)
        sampling_period = 1 / sample_rate
        v_fiber = 2e8
        distance_scale_m = (group_size * sampling_period * v_fiber) / 2
        
        self._append_status(f"Spatial resolution: {distance_scale_m:.3f} m/point (group_size={group_size})")
        
        self.plot_canvas_3d.plot_3d(traces, freqs, distance_scale_m)
        
        self._append_status("Performing Lorentzian fitting...")
        dist, peaks = process_all_traces_xcorr(traces, freqs, spacing=distance_scale_m, gamma=gamma_ghz)
        self.plot_canvas_peak.plot_peak_fit(dist, peaks)
        self._append_status("✅ Fitting completed.")
        
        self._fitted_data = {
            'traces': traces,
            'freqs': freqs,
            'dist': dist,
            'peaks': peaks,
            'gamma': gamma_ghz,
            'distance_scale_m': distance_scale_m
        }
        
        max_dist = dist[-1]
        self.slice_distance.setRange(0, max_dist)
        self.slice_distance.setSingleStep(distance_scale_m)
        self.slice_info.setText(f"Available distance range: 0 to {max_dist:.1f} m (resolution: {distance_scale_m:.3f} m)")
        self.slice_info.setStyleSheet("QLabel { color: green; padding: 5px; }")

        self.status_info.setText("Ready - Sweep complete")

        log_path = results.get("log_file")
        if log_path and Path(log_path).exists():
            for encoding in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']:
                try:
                    self.log_view.setPlainText(Path(log_path).read_text(encoding=encoding))
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                try:
                    with open(log_path, 'rb') as f:
                        content = f.read().decode('utf-8', errors='replace')
                    self.log_view.setPlainText(content)
                except Exception as e:
                    self.log_view.setPlainText(f"Error reading log file: {e}")
        else:
            self.log_view.setPlainText("No log file available.")

    def _on_sweep_failed(self, message: str) -> None:
        self.btn_abort.setEnabled(False)
        self.btn_run.setEnabled(True)
        self._append_status(f"❌ Sweep failed: {message}")
        self.status_info.setText("Sweep failed")

    def _plot_slice(self) -> None:
        """Plot 2D BGS slice at specified distance."""
        if not self._fitted_data:
            self._append_status("⚠️ No fitted data available.")
            return
        
        target_distance = self.slice_distance.value()
        
        traces = self._fitted_data['traces']
        freqs = self._fitted_data['freqs']
        dist = self._fitted_data['dist']
        peaks = self._fitted_data['peaks']
        gamma = self._fitted_data['gamma']
        
        idx = np.argmin(np.abs(dist - target_distance))
        actual_distance = dist[idx]
        trace = traces[:, idx]
        f0 = peaks[idx]
        
        self.plot_canvas_slice.plot_slice(freqs, trace, f0, gamma, actual_distance)
        
        self._append_status(f"✅ Plotted slice at {actual_distance:.1f} m (f₀ = {f0:.4f} GHz)")

    def _save_csv(self) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.warning(self, "No Data", "No results to save.")
            return

        save_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return

        traces = self._last_results["processed_traces"]
        freqs = self._last_results["freqs"]
        t_us = self._last_results["t_us"]
        log_file = self._last_results.get("log_file")
        save_traces(traces, freqs, t_us, save_dir=save_dir, log_file=log_file)
        self._append_status(f"✅ Saved CSV to {save_dir}")
        QtWidgets.QMessageBox.information(self, "Success", f"Data saved to {save_dir}")

    def _open_log(self) -> None:
        if not self._last_results or not self._last_results.get("log_file"):
            QtWidgets.QMessageBox.warning(self, "No Log", "No log file to open.")
            return

        log_file = self._last_results["log_file"]
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(log_file))

    def _save_post_comments(self) -> None:
        """Save post-run comments to the log file."""
        if not self._last_results or not self._last_results.get("log_file"):
            QtWidgets.QMessageBox.warning(self, "No Log", "No log file to save comments to.")
            return

        comment_text = self.post_run_comments.toPlainText().strip()
        if not comment_text:
            QtWidgets.QMessageBox.warning(self, "No Comments", "Please enter comments before saving.")
            return

        log_file = self._last_results["log_file"]
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write("\n\n===== POST-RUN COMMENTS (from GUI) =====\n")
                f.write(comment_text + "\n")
                f.write("=" * 40 + "\n")
            
            self._append_status(f"✅ Comments saved to log file.")
            QtWidgets.QMessageBox.information(self, "Success", "Comments saved to log file.")
            
            for encoding in ['utf-8', 'cp1252', 'latin-1', 'iso-8859-1']:
                try:
                    self.log_view.setPlainText(Path(log_file).read_text(encoding=encoding))
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
        except Exception as e:
            self._append_status(f"❌ Failed to save comments: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save comments: {e}")

    # Menu action implementations
    def _new_session(self) -> None:
        reply = QtWidgets.QMessageBox.question(
            self, 'New Session',
            'Start a new session? Unsaved data will be lost.',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self._last_results = None
            self._fitted_data = None
            self.pre_run_comments.clear()
            self.post_run_comments.clear()
            self._append_status("New session started")

    def _load_data(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Data", "", "NPZ Files (*.npz);;All Files (*)"
        )
        if filename:
            try:
                data = np.load(filename, allow_pickle=True)
                self._append_status(f"✅ Loaded data from {filename}")
                QtWidgets.QMessageBox.information(self, "Success", "Data loaded successfully")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    def _save_data(self) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to save.")
            return
        
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "", "NPZ Files (*.npz)"
        )
        if filename:
            try:
                np.savez_compressed(
                    filename,
                    traces=self._last_results["processed_traces"],
                    freqs=self._last_results["freqs"],
                    t_us=self._last_results["t_us"]
                )
                self._append_status(f"✅ Saved data to {filename}")
                QtWidgets.QMessageBox.information(self, "Success", "Data saved successfully")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save data: {e}")

    def _export_data(self, format_type: str) -> None:
        if not self._last_results:
            QtWidgets.QMessageBox.warning(self, "No Data", "No data to export.")
            return
        
        if format_type == "csv":
            self._save_csv()
        elif format_type == "npz":
            self._save_data()

    def _manage_presets(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Manage Presets")
        dialog.resize(400, 300)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        preset_list = QtWidgets.QListWidget()
        for preset_name in self._config_presets.keys():
            preset_list.addItem(preset_name)
        layout.addWidget(preset_list)
        
        btn_layout = QtWidgets.QHBoxLayout()
        delete_btn = QtWidgets.QPushButton("Delete Selected")
        delete_btn.clicked.connect(lambda: self._delete_preset(preset_list))
        btn_layout.addWidget(delete_btn)
        btn_layout.addStretch()
        
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        dialog.exec()

    def _delete_preset(self, preset_list: QtWidgets.QListWidget) -> None:
        current_item = preset_list.currentItem()
        if current_item:
            preset_name = current_item.text()
            del self._config_presets[preset_name]
            self._save_presets()
            preset_list.takeItem(preset_list.currentRow())
            self.preset_combo.removeItem(self.preset_combo.findText(preset_name))

    def _save_current_preset(self) -> None:
        name, ok = QtWidgets.QInputDialog.getText(self, "Save Preset", "Enter preset name:")
        if ok and name:
            self._config_presets[name] = {
                'start_freq': self.start_freq.value(),
                'stop_freq': self.stop_freq.value(),
                'step_mhz': self.step_mhz.value(),
                'power_dbm': self.power_dbm.value(),
                'fiber_len': self.fiber_len.value(),
                'sample_rate': self.sample_rate.value(),
                'averages': self.averages.value(),
                'pulse_width': self.pulse_width.value(),
                'gamma': self.gamma.value(),
            }
            self._save_presets()
            if self.preset_combo.findText(name) == -1:
                self.preset_combo.addItem(name)
            QtWidgets.QMessageBox.information(self, "Success", f"Preset '{name}' saved")

    def _load_preset(self, preset_name: str) -> None:
        if preset_name == "Default" or preset_name not in self._config_presets:
            return
        
        preset = self._config_presets[preset_name]
        self.start_freq.setValue(preset['start_freq'])
        self.stop_freq.setValue(preset['stop_freq'])
        self.step_mhz.setValue(preset['step_mhz'])
        self.power_dbm.setValue(preset['power_dbm'])
        self.fiber_len.setValue(preset['fiber_len'])
        self.sample_rate.setValue(preset['sample_rate'])
        self.averages.setValue(preset['averages'])
        self.pulse_width.setValue(preset['pulse_width'])
        self.gamma.setValue(preset['gamma'])

    def _save_presets(self) -> None:
        try:
            with open("botda_presets.json", "w") as f:
                json.dump(self._config_presets, f, indent=2)
        except Exception as e:
            print(f"Failed to save presets: {e}")

    def _load_presets(self) -> None:
        try:
            if Path("botda_presets.json").exists():
                with open("botda_presets.json", "r") as f:
                    self._config_presets = json.load(f)
        except Exception as e:
            print(f"Failed to load presets: {e}")

    def _compare_measurements(self) -> None:
        QtWidgets.QMessageBox.information(
            self, "Compare Measurements",
            "This feature allows you to overlay multiple measurement runs.\n"
            "Implementation coming soon!"
        )

    def _show_documentation(self) -> None:
        doc_text = """
        <h2>DSS BOTDA - Documentation</h2>

        <h3>Quick Start:</h3>
        <ol>
            <li>Configure RF and acquisition settings in the Configuration tab</li>
            <li>Connect RF and ADQ controllers</li>
            <li>Run the sweep</li>
            <li>Analyze results in Results & 2D Slice tabs</li>
        </ol>

        <h3>Key Features:</h3>
        <ul>
            <li><b>Presets:</b> Save/load frequently used configurations</li>
            <li><b>Auto-fitting:</b> Automatic Lorentzian peak detection</li>
            <li><b>Slice Analysis:</b> Examine individual distance points</li>
            <li><b>Export:</b> Save data in multiple formats</li>
            <li><b>Themes:</b> Switch between Light and Dark themes</li>
        </ul>

        <h3>Keyboard Shortcuts:</h3>
        <ul>
            <li><b>Ctrl+N:</b> New Session</li>
            <li><b>Ctrl+S:</b> Save Data</li>
            <li><b>Ctrl+O:</b> Load Data</li>
            <li><b>Ctrl+Q:</b> Exit</li>
            <li><b>F1:</b> Help (this dialog)</li>
        </ul>
        """

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("DSS BOTDA - Documentation")
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setText(doc_text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec()

    def _show_about(self) -> None:
        about_text = f"""
        <h2>DSS BOTDA - Developer Edition</h2>
        <p><b>Version:</b> 2.0 Developer Edition</p>
        <p><b>Date:</b> {datetime.now().strftime('%Y-%m-%d')}</p>

        <p>Distributed Sensor System (DSS) for BOTDA (Brillouin Optical Time Domain Analysis)
        measurement and analysis.</p>

        <p><b>Features:</b></p>
        <ul>
            <li>Automated RF frequency sweeping</li>
            <li>Real-time data acquisition</li>
            <li>Lorentzian peak fitting</li>
            <li>3D visualization</li>
            <li>Comprehensive logging</li>
            <li>Light/Dark theme support</li>
        </ul>

        <p><b>Developed for advanced distributed fiber optic sensing applications.</b></p>
        """

        msg = QtWidgets.QMessageBox(self)
        msg.setWindowTitle("About DSS BOTDA")
        msg.setTextFormat(QtCore.Qt.RichText)
        msg.setText(about_text)
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec()

    def _change_theme(self, theme: str) -> None:
        """Change application theme."""
        self._apply_theme(theme)
        
        # Update checkmarks
        if theme == "light":
            self.light_theme_action.setChecked(True)
            self.dark_theme_action.setChecked(False)
        else:
            self.light_theme_action.setChecked(False)
            self.dark_theme_action.setChecked(True)

    def _collect_rf_params(self) -> RFParams:
        return RFParams(
            start_freq=float(self.start_freq.value()),
            stop_freq=float(self.stop_freq.value()),
            step_mhz=float(self.step_mhz.value()),
            sweep_power_dbm=float(self.power_dbm.value()),
            com_port=self.com_port.text().strip(),
        )

    def _collect_sweep_settings(self) -> SweepSettings:
        return SweepSettings(
            fiber_len_m=float(self.fiber_len.value()),
            dwell_time=float(self.dwell_time.value()),
            arm_before_each=True,
            inter_capture_delay=0.0,
            settle_extra=0.0,
            sample_rate_hz=float(self.sample_rate.value()) * 1e6,
            averages=int(self.averages.value()),
            pulselength_ns=int(self.pulse_width.value()),
            save_dir=self.save_dir.text().strip(),
        )

    def _append_status(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.append(f"[{timestamp}] {message}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle window close event."""
        reply = QtWidgets.QMessageBox.question(
            self, 'Exit Confirmation',
            'Are you sure you want to exit?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    app.setApplicationName("DSS BOTDA - Developer Edition")
    app.setOrganizationName("DSS Research")

    # Set application icon for taskbar
    icon_path = Path(__file__).parent / "dss_icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))

    window = BotdaGui()
    # Window already maximized in __init__, no need to call show() separately
    app.exec()
