---
title: Setup
---
# Requirements

## Folder setup

Create a new directory for the workshop (e.g., Desktop/workshop-ml). We will place our code generated during the workshop in this folder.

~~~
cd Desktop
mkdir workshop-ml
~~~
{: .language-bash}

## Software

You will need a terminal (or Git Bash, Anaconda Prompt), Python 3.8+, and the ability to create Python virtual environments. You will also need the MatPlotLib, Pandas, Numpy, scikit-learn, and OpenCV packages. 

### Installing Python

Python is a popular language for scientific computing, and a frequent choice
for machine learning as well.
To install Python, follow the [Beginner's Guide](https://wiki.python.org/moin/BeginnersGuide/Download) or head straight to the [download page](https://www.python.org/downloads/).

Please set up your python environment at least a day in advance of the workshop.
If you encounter problems with the installation procedure, ask your workshop organizers via e-mail for assistance so
you are ready to go as soon as the workshop begins.

### Creating a new virtual environment
We'll install the prerequisite libraries in a virtual environment, to prevent them from cluttering up your Python environment and causing conflicts.

To create a new virtual environment ("venv") called "intro_ml" for the project, open the terminal (Mac/Linux), Git Bash (Windows), or Anaconda Prompt (Windows), and type one of the below OS-specific options:

~~~
python3 -m venv intro_ml # mac/linux
python -m venv intro_ml # windows
~~~
{: .language-bash}

> If you're on Linux and this doesn't work, you may need to install venv first. Try running `sudo apt-get install python3-venv` first, then `python3 -m venv intro_ml`
{: .info}

### Activating the environment
To activate the environment, run the following OS-specific commands in Terminal (Mac/Linux), Git Bash (Windows), or Anaconda Prompt (Windows):

* **Windows + Git Bash**: `source intro_ml/Scripts/activate`
* **Windows + Anaconda Prompt**: `intro_ml/Scripts/activate`
* **Mac/Linux**: `source intro_ml/bin/activate`

### Installing your prerequisites
Once the virtual environment is activated, install the prerequisites by running the following commands:

First, make sure you have the latest version of pip by running:

~~~
python.exe -m pip install --upgrade pip
~~~
{: .language-bash}

Then, install the required libraries.

~~~
pip install numpy pandas matplotlib opencv-python scikit-learn jupyterlab
~~~
{: .language-bash}

> Including `jupyterlab` ensures that all participants use the same environment for running and interacting with notebooks during the workshop.
{: .tip}

### Adding your virtual environment to JupyterLab
To use this virtual environment in JupyterLab, follow these steps:

1. Install the `ipykernel` package (if not already included):
   ~~~
   pip install ipykernel
   ~~~
   {: .language-bash}

2. Add the virtual environment as a Jupyter kernel:
   ~~~
   python -m ipykernel install --user --name=intro_ml --display-name "Python (intro_ml)"
   ~~~
   {: .language-bash}

3. When you launch JupyterLab, select the `Python (intro_ml)` kernel to ensure your code runs in the correct environment.

### Deactivating/activating environment
To deactivate your virtual environment, simply run `deactivate` in your terminal or prompt. If you close the terminal, Git Bash, or Conda Prompt without deactivating, the environment will automatically close as the session ends. Later, you can reactivate the environment using the "Activate environment" instructions above to continue working. If you want to keep coding in the same terminal but no longer need this environment, it’s best to explicitly deactivate it. This ensures that the software installed for this workshop doesn’t interfere with your default Python setup or other projects.

## Fallback option: cloud environment
If a local installation does not work for you, it is also possible to run this lesson in [Google colab](https://colab.research.google.com/). If you open a jupyter notebook there, the required packages are already pre-installed.

{% include links.md %}
