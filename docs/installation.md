# Installation Guide for Crop Yield Dashboard

## Understanding the Installation Process

Before we dive into the installation steps, let's understand what we're about to do. Installing the Crop Yield Dashboard involves setting up a Python environment with specific libraries that work together to create an interactive web application. Think of it like building a house - we need to lay the foundation (Python), set up the framework (virtual environment), and then add all the specialized components (libraries) that make the dashboard functional.

The entire installation process typically takes 10-15 minutes, depending on your internet speed and computer performance. Don't worry if you encounter some warnings during installation - we'll address common issues and explain what's happening at each step.

## System Requirements

### Minimum Requirements

Your computer needs to meet these basic requirements to run the dashboard smoothly:

**Hardware Requirements:**

- **Processor**: Any modern CPU (Intel i3/AMD Ryzen 3 or better recommended)
- **RAM**: 4GB minimum (8GB or more recommended for larger datasets)
- **Storage**: 2GB of free disk space (for Python, libraries, and temporary files)
- **Display**: 1366x768 resolution or higher for optimal visualization

**Operating System Compatibility:**

- **Windows**: Windows 10 version 1903 or later (64-bit recommended)
- **macOS**: macOS 10.14 (Mojave) or later
- **Linux**: Ubuntu 18.04 LTS, Debian 10, CentOS 7, or equivalent

**Software Prerequisites:**

- **Python**: Version 3.8, 3.9, 3.10, or 3.11 (we'll show you how to check and install)
- **Web Browser**: Chrome, Firefox, Safari, or Edge (for viewing the dashboard)
- **Internet Connection**: Required for initial setup and downloading packages

### Checking Your Current Setup

Let's start by checking what you already have installed. Open your terminal or command prompt and run these diagnostic commands:

```bash
# Check if Python is installed and its version
python --version
# or on some systems:
python3 --version

# Check if pip (Python package manager) is installed
pip --version
# or:
pip3 --version

# Check available disk space
# On Windows:
dir
# On macOS/Linux:
df -h
```

If Python isn't installed or you have an older version, don't worry - we'll walk through the installation process step by step.

## Installation Methods

We'll cover three different installation approaches, starting with the most straightforward method. Choose the one that best fits your comfort level and system setup.

### Method 1: Standard Installation with pip (Recommended for Most Users)

This method is perfect if you're new to Python or want the simplest setup process. We'll use pip, Python's standard package manager, which comes bundled with Python installations.

#### Step 1: Installing Python

If you don't have Python installed, let's get it set up:

**For Windows Users:**

1. Visit the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Click the "Download Python 3.x.x" button (the exact version number will vary)
3. Run the downloaded installer
4. **Important**: Check the box that says "Add Python to PATH" before clicking Install
5. Click "Install Now" and wait for the installation to complete
6. Restart your command prompt to ensure the changes take effect

**For macOS Users:**
Python comes pre-installed on macOS, but it's often an older version. To install the latest version:

```bash
# If you have Homebrew installed (recommended):
brew install python@3.11

# If you don't have Homebrew, install it first:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# Then run the brew install command above
```

**For Linux Users:**
Most Linux distributions come with Python, but you might need to install pip:

```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Fedora:
sudo dnf install python3 python3-pip

# CentOS/RHEL:
sudo yum install python3 python3-pip
```

#### Step 2: Cloning the Repository

Now let's get the dashboard code onto your computer. If you have Git installed, this is straightforward:

```bash
# Navigate to where you want to install the dashboard
# For example, to install in your home directory:
cd ~

# Clone the repository
git clone https://github.com/d-dziublenko/crop-yield-dashboard.git

# Enter the project directory
cd crop-yield-dashboard
```

If you don't have Git installed, you can download the repository as a ZIP file:

1. Visit the repository page on GitHub
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to your desired location
5. Open a terminal/command prompt and navigate to the extracted folder

#### Step 3: Creating a Virtual Environment

A virtual environment is like a private workspace for your project. It keeps all the dashboard's dependencies separate from other Python projects on your system, preventing conflicts. Think of it as creating a dedicated toolbox for this specific project.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# If the above doesn't work, try:
python3 -m venv venv
```

This creates a new folder called `venv` in your project directory containing a clean Python installation.

#### Step 4: Activating the Virtual Environment

Now we need to "enter" this virtual environment. The activation command differs by operating system:

**On Windows (Command Prompt):**

```bash
venv\Scripts\activate
```

**On Windows (PowerShell):**

```bash
# If you get an execution policy error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate:
venv\Scripts\Activate.ps1
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

When activated successfully, you'll see `(venv)` appear at the beginning of your command prompt. This indicates you're now working inside the virtual environment.

#### Step 5: Installing Dependencies

With the virtual environment activated, we can now install all the required packages. The `requirements.txt` file contains a list of all necessary libraries and their versions.

```bash
# Upgrade pip to the latest version first (recommended)
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

This process will download and install numerous packages including:

- **Streamlit**: The web framework for our dashboard
- **pandas**: For data manipulation
- **numpy**: For numerical computations
- **XGBoost**: The machine learning algorithm
- **SHAP**: For model interpretability
- **Plotly**: For interactive visualizations
- **Folium**: For geographic mapping

The installation might take several minutes as it downloads all packages and their dependencies. You'll see progress bars and messages as each package installs.

#### Step 6: Verifying the Installation

Let's make sure everything installed correctly:

```bash
# Check that key packages are installed
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
python -c "import xgboost; print(f'XGBoost version: {xgboost.__version__}')"
python -c "import shap; print(f'SHAP version: {shap.__version__}')"

# List all installed packages
pip list
```

#### Step 7: Running the Dashboard

Now for the exciting part - launching your dashboard:

```bash
streamlit run crop_yield_dashboard.py
```

You should see output similar to:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

Your default web browser should automatically open to the dashboard. If it doesn't, manually open a browser and navigate to `http://localhost:8501`.

### Method 2: Installation Using Docker (For Consistent Environments)

Docker provides a way to package the entire application with all its dependencies into a container. This ensures the dashboard runs identically on any system that has Docker installed. Think of Docker as a shipping container for software - everything needed is packed inside, ready to run anywhere.

#### Prerequisites for Docker Installation

First, you'll need Docker installed on your system:

**Installing Docker Desktop:**

1. Visit [https://www.docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
2. Download Docker Desktop for your operating system
3. Run the installer and follow the setup wizard
4. Restart your computer if prompted
5. Verify Docker is running by opening a terminal and typing:
   ```bash
   docker --version
   ```

#### Building and Running with Docker

Once Docker is installed, the process is remarkably simple:

```bash
# Navigate to the project directory
cd crop-yield-dashboard

# Build the Docker image (this packages everything needed)
docker build -t crop-yield-dashboard .

# Run the container
docker run -p 8501:8501 crop-yield-dashboard
```

The first build might take 5-10 minutes as Docker downloads the base Python image and installs all dependencies. Subsequent runs will be much faster.

To stop the container, press `Ctrl+C` in the terminal where it's running.

#### Using Docker Compose (Even Easier)

Docker Compose simplifies the process even further:

```bash
# Start the dashboard with a single command
docker-compose up

# To run in the background (detached mode):
docker-compose up -d

# To stop the dashboard:
docker-compose down
```

### Method 3: Installation Using Conda (For Data Science Environments)

Conda is particularly popular in the data science community because it can manage both Python packages and system-level dependencies. If you're already using Anaconda or Miniconda for other projects, this method integrates well with your existing workflow.

#### Installing Miniconda

If you don't have Conda installed:

1. Download Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
2. Choose the installer for your operating system and Python 3.x
3. Run the installer and follow the prompts
4. Restart your terminal to apply the changes

#### Creating a Conda Environment

```bash
# Create a new environment with Python 3.9
conda create -n crop-yield python=3.9

# Activate the environment
conda activate crop-yield

# Install pip in the conda environment (important!)
conda install pip

# Navigate to the project directory
cd crop-yield-dashboard

# Install the requirements using pip within the conda environment
pip install -r requirements.txt
```

## Troubleshooting Common Installation Issues

Even with careful installation, you might encounter some issues. Here's how to resolve the most common problems:

### Issue 1: "Python is not recognized as an internal or external command"

This error on Windows means Python wasn't added to your system PATH during installation.

**Solution:**

1. Reinstall Python and ensure you check "Add Python to PATH"
2. Or manually add Python to PATH:
   - Search for "Environment Variables" in Windows settings
   - Edit the System PATH variable
   - Add the Python installation directory (usually `C:\Python39` or `C:\Users\YourName\AppData\Local\Programs\Python\Python39`)

### Issue 2: "No module named 'streamlit'" after installation

This usually means you're not in the virtual environment or the installation didn't complete successfully.

**Solution:**

1. Ensure your virtual environment is activated (you should see `(venv)` in your prompt)
2. Reinstall the requirements:
   ```bash
   pip install --force-reinstall -r requirements.txt
   ```

### Issue 3: "error: Microsoft Visual C++ 14.0 is required" (Windows)

Some Python packages require compilation, which needs Visual C++ tools on Windows.

**Solution:**

1. Install Visual Studio Build Tools:

   - Download from [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Run the installer
   - Select "Desktop development with C++"
   - Install (this is large, about 6GB)

2. Alternative: Use pre-compiled wheels:
   ```bash
   pip install --only-binary :all: -r requirements.txt
   ```

### Issue 4: SSL Certificate Errors

These occur when downloading packages behind corporate firewalls or with outdated certificates.

**Solution (use with caution in corporate environments only):**

```bash
# Temporary workaround - upgrade certificates first
pip install --upgrade certifi

# If still having issues, you can bypass SSL (not recommended for production):
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue 5: "Port 8501 is already in use"

This means another application (possibly another Streamlit app) is using the default port.

**Solution:**

```bash
# Run on a different port
streamlit run crop_yield_dashboard.py --server.port 8502

# Or find and stop the process using port 8501:
# On Windows:
netstat -ano | findstr :8501
# Then kill the process using its PID

# On macOS/Linux:
lsof -ti:8501 | xargs kill
```

### Issue 6: Memory Errors During Installation

Large packages like XGBoost might fail to install on systems with limited RAM.

**Solution:**

```bash
# Install packages one at a time to reduce memory usage
pip install streamlit
pip install pandas numpy
pip install xgboost
pip install shap
# Continue with other packages...
```

## Post-Installation Setup

### Configuring Your Environment

After successful installation, you might want to customize some settings:

1. **Create a `.streamlit` directory in your project folder:**

   ```bash
   mkdir .streamlit
   ```

2. **Create a `config.toml` file for Streamlit settings:**

   ```toml
   [theme]
   primaryColor = "#2E7D32"
   backgroundColor = "#FFFFFF"
   secondaryBackgroundColor = "#F0F2F6"
   textColor = "#262730"
   font = "sans serif"

   [server]
   maxUploadSize = 200
   enableCORS = false
   ```

### Testing Your Installation

Run these tests to ensure everything is working correctly:

```python
# Create a file called test_installation.py
import sys
print(f"Python version: {sys.version}")

try:
    import streamlit
    print(f"✓ Streamlit {streamlit.__version__} installed")
except ImportError:
    print("✗ Streamlit not found")

try:
    import xgboost
    print(f"✓ XGBoost {xgboost.__version__} installed")
except ImportError:
    print("✗ XGBoost not found")

try:
    import shap
    print(f"✓ SHAP {shap.__version__} installed")
except ImportError:
    print("✗ SHAP not found")

# Run with: python test_installation.py
```

## Updating the Dashboard

To update to the latest version in the future:

```bash
# Pull the latest code
git pull origin main

# Activate your virtual environment
# (Use the appropriate command for your OS)

# Update dependencies
pip install --upgrade -r requirements.txt
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/d-dziublenko/crop-yield-dashboard/issues) page
2. Review the [Streamlit documentation](https://docs.streamlit.io/)
3. Ask for help in the [Streamlit community forum](https://discuss.streamlit.io/)
4. Create a new issue with details about your problem, including:
   - Your operating system and version
   - Python version
   - Complete error messages
   - Steps you've already tried

Remember, installation issues are often system-specific, and what works on one computer might need adjustment on another. Don't hesitate to seek help if you're stuck - the community is generally very helpful and welcoming to newcomers.

## Next Steps

- Load your agricultural data
- Train prediction models
- Interpret the results
- Export your findings

The dashboard is designed to be intuitive, but understanding its full capabilities will help you get the most value from your agricultural data analysis.

Happy predicting! 🌾
