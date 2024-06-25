#!/bin/bash


# Run the first Python script
python script1.py

# Run the MATLAB script
matlab -nodisplay -nosplash -nodesktop -r "run('script.m');exit;"

# Run the second Python script
python script2.py