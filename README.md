# CodeCPD

This is the repository for the experiments of

"Projected Change Point Detection based on the Sliced Wasserstein distance for
high-dimensional time series"

and contains:
- All datasets
- Methods
- Sampling methods

Make sure that you have an installation of Python and R. We choose to create a virtual environment for this repository called venv and some similar solution for R (rlib). All neccessary libraries for R are already in the rlib folder. Python packages must be installed manually first.

### Set up repo and virtual env to run experiments in python
Just clone repository and create an virtual environment called vevn first and create result directory

```
git clone ...
python3 -m venv path/to/repo/venv
mkdir path/to/repo/results
```
Then activate virtual environment and install required packages in requirements.txt (BOCPMS)
```
source venv/bin/activate
cd methods/python/bocpdms
pip install -r requirements.txt
```
The results directory has the same structure as datasets and will be emtpy first (also part of .gitignore file). It will be automatically filled if you run experiments.
### Run experiments 
If you want to run all experiments please run (not recommended)
```
sh Simulation.sh
```
The bash script activates the virtual environment and calls R, and Python scripts of the methods folder. Each script iterates over all datasets and stores the results in the result folder.


If you want to run specific methods on specific datasets (recommended), you can modify config.py and use
```
python3 config.py
Rscript methods/method.R
or
python3 methods/python/run_method.py
```
to update config.json and run the selected method.

For example say you want to run Bayesian Online Change Point Detection (BOCPD) just on the Apple dataset. Update datasets in config.py and set DATASETS = ['apple'], update config.json, and run ocp.R.

You can run
```
python3 summarize.py
```
to calculate and summarize the metrics (F1-score, Segmentation Covering)

