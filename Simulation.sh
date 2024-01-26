#!/bin/bash
#echo "Script executed from: ${PWD}"

# SCRIPT_DIR=$(realpath "$(dirname "$0")")

# ENV="${SCRIPT_DIR}/venv"

# #ENV="${PWD}/venv"


# #SCRIPT_DIR=$(realpath "$(dirname "$0")")

# echo "${SCRIPT_DIR}"
# #echo "${ENV}"

# # Change to the directory where the virtual environment will be created
# cd "${ENV}"

# #echo "${ENV}"

# # # Create a virtual environment using python -m venv
# #python3 -m venv venv


# # # Activate the virtual environment
# source "${ENV}"/bin/activate

# # Install the packages specified in requirements.txt
#pip install -r requirements.txt

# # Optional: You can add more commands or instructions here

#ENV="$(pwd)/venv"

#echo "${ENV}"/bin/activate

#source "${ENV}"/bin/activate


#cd ..

python3 config.py

#Rscript ./methods/ocp.R
Rscript ./methods/ecp.R
Rscript ./methods/kcpa.R
Rscript ./methods/WATCH.R


