#!/bin/bash

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "${SCRIPT_DIR}"


# Path to the virtual environment
if [ "$(pwd)" != "$SCRIPT_DIR" ]; then
    echo "Script is not executed from its directory. Exiting."
    source "cd in Project directory"
    VIRTUAL_ENV_PATH="$SCRIPT_DIR/venv"

else
     echo "Script is excecuted from ${SCRIPT_DIR}."
    #VIRTUAL_ENV_PATH="venv"
    source venv/bin/activate
fi 


echo "${VIRTUAL_ENV_PATH}"

# if [ -z "$VIRTUAL_ENV" ]; then
#     #Virtual environment is not activated, activate it
#     source "$VIRTUAL_ENV_PATH/bin/activate"
#     echo "Virtual environment activated."
# else
#     # Virtual environment is already activated
#     echo "Virtual environment is already activated."
# fi
#echo "Script executed from: ${PWD}"


python3 config.py

python3 ./methods/python/run_SWD.py
python3 ./methods/python/run_SWD10.py
python3 ./methods/python/run_SWD20.py
python3 ./methods/python/run_MMD.py


Rscript ./methods/ocp.R
Rscript ./methods/ecp.R
Rscript ./methods/kcpa.R
Rscript ./methods/WATCH.R


