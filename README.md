# DS4CG-Automatic-Georeferencing
To get a better understanding of this project before using it, please watch this short video: and/or read this short paper: . This project was created by Collin Giguere and Sowmya Vasuki Jallepalli for the Data Science for the Common Good program in Summer 2021 in partnership with UMass Libraries and Department of Environmental Conservation and advising from Pixel Forensics. To find out more about the program, see ds.cs.umass.edu/ds4cg.

## Setup
Before using the application, you must set up the environment. First install python on your device by following the steps here: https://www.python.org/downloads/. Make sure you download and install **Python 3.9**. Then install GIT from here: https://git-scm.com/downloads.

Next, open the terminal (on MacOS) or Command Prompt (on Windows), navigate to the directory you want the application in, and execute the following commands:

### Windows
`git clone https://github.com/cdgiguere/DS4CG-Automatic-Georeferencing.git`

`py -m pip install --upgrade pip`

`py -m pip install --user virtualenv`

`py -m venv env`

`.\env\Scripts\activate`
##### On 32-bit systems:
`py -m pip install Setup\requirements_32.txt`
##### On 64-bit systems:
`py -m pip install Setup\requirements_64.txt`

### MacOS/UNIX

### Config
Included with each of the two pipelines is a config yaml file. These are used to specify where files are located, where to place output files, and certain other parameters to use. Instructions on how to fill out these files are in the files themselves; simply open them with any text editor.

## Use
Now that the setup has been completed and the cofigurations have been set, all you need to do to run the application is navigate to the directory of the desired pipeline (.../DS4CG-Automatic-Georeferencing/Propagation/ or .../DS4CG-Automatic-Georeferencing/Satellite/) and run

### Windows
`py Propagate.py`
or
`py Satellite.py`

### MacOS/UNIX
`python Propagate.py`
or
`python Satellite.py`
