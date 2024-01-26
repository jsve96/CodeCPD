#!/bin/bash
echo "Script executed from: ${PWD}"

ENV="${PWD}/venv"

echo "${ENV}"

# Change to the directory where the virtual environment will be created
cd "${ENV}"

# # Create a virtual environment using python -m venv
python3 -m venv venv


# # Activate the virtual environment
. venv/bin/activate

# # Install the packages specified in requirements.txt
pip install -r requirements.txt

# # Optional: You can add more commands or instructions here


cd ..

python3 config.py

Rscript ./methods/ocp.R
Rscript ./methods/ecp.R
Rscript ./methods/kcpa.R
Rscript ./methods/WATCH.R


