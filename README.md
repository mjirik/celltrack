# CellTrack
Tracking of roots in microscopy images.


# Install 

1. [Install Conda](https://conda.io/miniconda.html) (for Python 3, 64-bit) and  check "Add Anaconda to my PATH environment variable" 
**during the installation**

2. Use terminal to install required packages
```bash
conda create -n celltrack -c mjirik -c conda-forge celltrack pywin32
conda activate celltrack
python -m celltrack install
python -m celltrack
```

The `pywin32` is used for icon installation. It can be skipped.

## For developpers

Clone the repo
```cmd
git clone git@github.com:mjirik/celltrack.git
```

# Run

```cmd
cd celltrack
python -m celltrack
```

# GUI
![graphics](docs/graphics/screenshot_gui03.png)
