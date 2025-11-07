---
title: "Setup"
teaching: 0
exercises: 0
---

:::::::::::::::::::::::::::::::::::::: questions 

- How do I prepare my system for this workshop?
- How do I create and activate a Python virtual environment with `uv`?
- How do I make my workshop environment available in JupyterLab?

::::::::::::::::::::::::::::::::::::::::::::::::

::::::::::::::::::::::::::::::::::::: objectives

- Create a dedicated folder for the workshop.
- Install Python 3.11.9.
- Install `uv` and use it to create and manage a virtual environment.
- Configure JupyterLab to use the workshop environment.

::::::::::::::::::::::::::::::::::::::::::::::::

This workshop provides a beginner-friendly overview of machine learning (ML) and common ML methods— including regression, classification, clustering, dimensionality reduction, ensemble methods, and a quick neural-network demo—using Python + scikit-learn. The broad coverage is designed to jump-start your ML journey and point you toward next learning steps.

::::::::::::::::::::::::::::::::::::: callout
### Prerequisites 
A basic understanding of Python. You will need to know how to write a for loop, if statement, use functions, libraries and perform basic arithmetic. 
Either of the Software Carpentry Python courses cover sufficient background.
::::::::::::::::::::::::::::::::::::: 

## Setup

Please complete the setup at least a day in advance of the workshop. If you run into issues, contact the workshop organizers by email so you're ready to begin on time. The setup steps include:

1. Set up the workshop folder  
2. Install Python 3.11.9  
3. Install `uv` and set up the virtual environment  

## 0. Set up a terminal or Git Bash if on Windows

To run the commands referenced in this setup, we recommend using:

- Terminal (Mac, Linux)
- Git Bash (Windows). Download and install from here: <https://git-scm.com/install/windows>. Git Bash provides a shell environment that closely resembles the terminal.

## 1. Set up the workshop folder

Create a folder on your desktop called `ML_workshop` for storing the workshop data and environment.

Open a terminal or Git Bash (Windows) to run the following commands. If you prefer, you can also create the folder manually on your Desktop.

```bash
cd ~/Desktop
mkdir ML_workshop
cd ML_workshop
pwd
```

## 2. Install Python 3.11.9

We recommend using Python 3.11.9 to ensure consistency across participants.

Download the appropriate installer from the [official 3.11.9 downloads page](https://www.python.org/downloads/release/python-3119/). Follow your OS-specific installation steps. **When prompted, select "Add python.exe to PATH".**

## 3. Install `uv` and set up the environment

[`uv`](https://github.com/astral-sh/uv) is a modern, fast Python package and environment manager. It's significantly faster than `pip` and simplifies reproducible setup.

### A. Install `uv`

```bash
pip install uv
```

### B. Create a `requirements.txt` file in the `ML_workshop` folder

```bash
touch requirements.txt
```

Use your preferred text editor to add the following contents into `ML_workshop/requirements.txt`. Make sure to save the file after your edits.

```text
numpy
pandas
matplotlib
opencv-python
scikit-learn
jupyterlab
seaborn
```

### C. Set up the virtual environment

```bash
pwd  # check to make sure you're still in the ML_workshop folder.
```

Create the virtual environment using Python 3.11.9:

```bash
uv venv --python=3.11.9
```

::::::::::::::::::::::::::::::::::::: callout

## What does the `.venv` folder contain?

Running `uv venv --python=3.11.9` creates a folder named `.venv/` in your `ML_workshop` directory.

Inside this folder, you'll find multiple subfolders and files:

- `bin/` (or `Scripts/` on Windows): contains the Python interpreter and executable scripts
- `lib/`: stores all installed Python packages (and their dependencies)
- `pyvenv.cfg`: tracks Python version and configuration
- `include/`: headers used to build native extensions

These components form an isolated environment, keeping your installed packages separate from your global Python setup.

::::::::::::::::::::::::::::::::::::::::::::::::

### D. Activate the environment

Run one of the OS-specific commands below:

```bash
# Mac/Linux
source .venv/bin/activate

# Git Bash on Windows
source .venv/Scripts/activate

# Windows CMD (not recommended)
.venv\Scripts\activate.bat
```

### E. Install `requirements.txt`

```bash
uv pip install -r requirements.txt
```

### F. Add the environment to JupyterLab

Run one of the OS-specific commands below.

```bash
# Mac/Linux
.venv/bin/python -m ipykernel install --user --name=.venv --display-name "ML_workshop"

# Git Bash on Windows
.venv/Scripts/python.exe -m ipykernel install --user --name=.venv --display-name "ML_workshop"

# Windows CMD (not recommended)
.venv\Scripts\python.exe -m ipykernel install --user --name=.venv --display-name "ML_workshop"
```

### G. Launch JupyterLab

```bash
jupyter lab
```

When Jupyter opens, select the `ML_workshop` kernel from the dropdown to start a new notebook.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from sklearn import datasets, model_selection, metrics
```

If this runs without error, your setup is complete!

::::::::::::::::::::::::::::::::::::: keypoints 

- Use a dedicated `ML_workshop` folder to keep all materials and the environment together.
- Install and use Python 3.11.9 so your setup matches the instructors'.
- Use `uv` to create and manage an isolated virtual environment in `.venv/`.
- Register the environment as a Jupyter kernel and select it before working through the lesson.

::::::::::::::::::::::::::::::::::::::::::::::::
