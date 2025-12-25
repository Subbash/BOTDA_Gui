# BOTDA Sweep Control - Professional Edition v2.0

## 📋 Overview
Professional BOTDA (Brillouin Optical Time Domain Analysis) measurement and analysis software with advanced GUI and data processing capabilities.

## 🎯 Key Features

### GUI Features:
- **Professional Menu Bar** - File, Settings, Tools, Help menus
- **Toolbar** - Quick access to common actions
- **Theme System** - Light/Dark mode toggle
- **Status Bar** - Real-time connection and operation status
- **Configuration Presets** - Save/load frequently used settings
- **Keyboard Shortcuts** - Efficient workflow (Ctrl+S, Ctrl+O, etc.)

### Measurement Features:
- **Automated RF Sweeping** - Configurable frequency range
- **Real-time Acquisition** - Live progress monitoring
- **Lorentzian Fitting** - Automatic BGS peak detection
- **3D Visualization** - Brillouin Gain Spectrum surface plots
- **2D Slice Analysis** - Detailed analysis at specific distances
- **Dual Plot Display** - 3D BGS and peak frequency side-by-side

### Data Management:
- **Multiple Export Formats** - CSV, NPZ
- **Auto-logging** - Comprehensive session logs
- **Pre/Post-run Comments** - Detailed experiment documentation
- **Spatial Resolution Calculation** - Automatic based on pulse width

## 📁 Files

### Core Files:
- **gui.py** - Main GUI application (Professional Edition with themes)
- **main.py** - BOTDA sweep control logic
- **processing.py** - Signal processing and trace analysis
- **plotter.py** - Plotting and data visualization
- **logger_utils.py** - Logging utilities
- **rf_controller.py** - RF synthesizer control
- **adq_controller.py** - ADQ digitizer control

## 🚀 Quick Start

### Installation:
```bash
pip install PySide6 numpy scipy matplotlib pandas tqdm
```

### Run:
```bash
python gui.py
```

## ⚙️ Configuration

### Default Settings:
- **Start Frequency**: 10.700 GHz
- **Stop Frequency**: 11.000 GHz
- **Step Size**: 1.000 MHz
- **Power**: -8.000 dBm
- **Fiber Length**: 2000.000 m
- **Sample Rate**: 200.000 MS/s
- **Averages**: 1000
- **Pulse Width**: 100 ns
- **Linewidth (γ)**: 35 MHz

### Spatial Resolution:
Automatically calculated based on:
- Pulse width
- Sample rate
- Speed of light in fiber (2×10⁸ m/s)

Formula: `resolution = (group_size × sampling_period × v_fiber) / 2`

## 🎨 Theme Options

### Light Theme (Default):
- Clean, professional appearance
- High contrast for bright environments
- Dark text on light backgrounds

### Dark Theme:
- Reduced eye strain
- Perfect for low-light conditions
- Light text on dark backgrounds

**Switch:** Settings → Theme → Light/Dark

## ⌨️ Keyboard Shortcuts

- **Ctrl+N** - New Session
- **Ctrl+O** - Load Data
- **Ctrl+S** - Save Data
- **Ctrl+Q** - Exit
- **Ctrl+Shift+S** - Save Preset
- **F1** - Help Documentation

## 📊 Workflow

1. **Configure** - Set RF and acquisition parameters
2. **Connect** - Connect RF and ADQ controllers
3. **Comment** - Add pre-run experimental notes
4. **Run** - Execute sweep measurement
5. **Analyze** - View 3D BGS and peak fits
6. **Slice** - Examine specific fiber locations
7. **Comment** - Add post-run observations
8. **Export** - Save data in desired format

## 🔧 Advanced Features

### Configuration Presets:
- Save current settings as named preset
- Quick-switch between configurations
- Presets saved to `botda_presets.json`

### 2D Slice Analysis:
- Select any distance point
- View detailed Lorentzian fit
- Compare measured vs fitted trace
- Extract peak frequency at location

### Data Export:
- **CSV**: Frequency × Distance matrix
- **NPZ**: Compressed numpy arrays
- **Logs**: Complete session documentation

## 📝 Logging

Each sweep generates:
- Timestamp
- RF parameters
- ADQ configuration
- Processing metrics
- Spatial resolution
- Group size calculation
- Pre-run comments
- Post-run comments (optional)

## 🛠️ Technical Details

### Distance Calculation:
```python
group_size = pulse_width_ns × sample_rate × 1e-9
v_fiber = 2e8  # m/s
distance_scale = (group_size × (1/sample_rate) × v_fiber) / 2
```

### Lorentzian Fitting:
- Cross-correlation method
- Reference kernel generation
- Peak frequency extraction
- Linewidth parameter: γ (MHz)

### Processing Pipeline:
1. Raw trace acquisition
2. Period reshaping
3. Spatial averaging (group_size)
4. DC offset removal
5. Lorentzian fitting
6. Distance mapping

## 🐛 Troubleshooting

### Common Issues:

**RF Connection Failed:**
- Check COM port setting
- Verify device is powered on
- Check USB cable connection

**ADQ Connection Failed:**
- Verify device drivers installed
- Check device is recognized
- Restart application

**Wrong Distance Scale:**
- Verify pulse width setting
- Check sample rate matches hardware
- Ensure correct fiber velocity (2e8 m/s)

**White Text on White Background:**
- Switch to Dark theme
- Settings → Theme → Dark Theme

## 📚 Documentation

Press **F1** in the application for built-in help documentation.

## 🔄 Version History

### v2.0 - Professional Edition
- Added theme system (Light/Dark)
- Professional menu bar
- Toolbar with quick actions
- Status bar with indicators
- Configuration presets
- Keyboard shortcuts
- Enhanced styling
- Message dialogs
- Tooltips on all controls
- Better error handling

### v1.0 - Initial Release
- Basic GUI functionality
- RF sweep control
- ADQ acquisition
- 3D plotting
- Basic logging

## 📧 Support

For issues or questions, refer to the built-in documentation (F1) or check the log files for detailed error messages.

## 📄 License

Developed for scientific research and educational purposes.

---

**Last Updated:** December 23, 2024
**Version:** 2.0 Professional Edition
