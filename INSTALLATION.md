# Installation Guide - BOTDA Sweep Control

## System Requirements

### Operating System:
- Windows 10/11 (Recommended)
- Linux (Ubuntu 20.04+)
- macOS 10.15+

### Python Version:
- Python 3.9 or higher

### Hardware Requirements:
- Windfreak SynthHD RF Synthesizer
- Teledyne SP Devices ADQ Digitizer
- USB connections for both devices

## Step-by-Step Installation

### 1. Install Python
Download and install Python from [python.org](https://www.python.org/downloads/)

**During installation:**
- ✅ Check "Add Python to PATH"
- ✅ Choose "Install for all users" (optional)

### 2. Install Required Packages

Open Command Prompt (Windows) or Terminal (Linux/Mac) and run:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install PySide6 numpy scipy matplotlib pandas tqdm
```

### 3. Install Hardware Drivers

#### Windfreak SynthHD:
```bash
pip install windfreak
```

#### ADQ Digitizer:
Install from Teledyne SP Devices:
- Download ADQ SDK from vendor website
- Install pyadq package as per vendor instructions

### 4. Verify Installation

Run this test script:

```python
import sys
print(f"Python version: {sys.version}")

try:
    import PySide6
    print("✅ PySide6 installed")
except:
    print("❌ PySide6 not found")

try:
    import numpy
    print("✅ NumPy installed")
except:
    print("❌ NumPy not found")

try:
    import scipy
    print("✅ SciPy installed")
except:
    print("❌ SciPy not found")

try:
    import matplotlib
    print("✅ Matplotlib installed")
except:
    print("❌ Matplotlib not found")

try:
    import pandas
    print("✅ Pandas installed")
except:
    print("❌ Pandas not found")

print("\nAll required packages installed!")
```

### 5. Download BOTDA Files

Download all files to a folder:
```
botda_sweep_control/
├── gui.py
├── main.py
├── processing.py
├── plotter.py
├── logger_utils.py
├── rf_controller.py
├── adq_controller.py
├── requirements.txt
└── README.md
```

### 6. Run the Application

Navigate to the folder and run:

```bash
python gui.py
```

## Configuration

### First Run Setup:

1. **Configure Save Directory:**
   - Go to Configuration tab
   - Set "Save Directory" to your preferred location
   - Click "Browse..." to select folder

2. **Set COM Port:**
   - Check Device Manager (Windows) for RF synthesizer COM port
   - Update "COM Port" field (e.g., COM19)

3. **Test Connections:**
   - Click "Connect RF" button
   - Click "Connect ADQ" button
   - Verify green checkmarks in status bar

## Troubleshooting

### PySide6 Installation Issues:

**Windows:**
```bash
pip install PySide6 --upgrade
```

**Linux:**
```bash
sudo apt-get install python3-pyside6
# or
pip install PySide6
```

### Import Errors:

If you get "ModuleNotFoundError":
```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

### Hardware Not Detected:

**Windfreak:**
- Check USB connection
- Verify COM port in Device Manager
- Try different USB port
- Restart device

**ADQ:**
- Ensure SDK is properly installed
- Check device appears in system
- Verify driver installation
- Contact vendor support

### Permission Errors:

**Windows:**
Run Command Prompt as Administrator:
```bash
pip install --user -r requirements.txt
```

**Linux:**
```bash
sudo pip3 install -r requirements.txt
# or
pip install --user -r requirements.txt
```

## Updates

To update the software:

```bash
# Update Python packages
pip install --upgrade -r requirements.txt

# Download latest versions of .py files
# Replace old files with new versions
```

## Support

For installation issues:
1. Check error messages in terminal/command prompt
2. Verify all requirements are met
3. Check vendor documentation for hardware drivers
4. Ensure Python PATH is set correctly

## Advanced Configuration

### Virtual Environment (Recommended):

```bash
# Create virtual environment
python -m venv botda_env

# Activate (Windows)
botda_env\Scripts\activate

# Activate (Linux/Mac)
source botda_env/bin/activate

# Install packages
pip install -r requirements.txt

# Run application
python gui.py
```

### Custom Python Path:

If multiple Python versions installed:

```bash
# Windows
py -3.9 -m pip install -r requirements.txt
py -3.9 gui.py

# Linux/Mac
python3.9 -m pip install -r requirements.txt
python3.9 gui.py
```

---

**Need Help?** Check the README.md for usage instructions and press F1 in the application for documentation.
