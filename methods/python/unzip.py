import sys
import json
import os
import pathlib
import zipfile
import shutil
from utils import *





tmp_path = sys.argv[1]

dataset_path = sys.argv[2]

if isZIP(dataset_path):
    output_script = True
#create tmp folder
    if not os.path.exists(os.path.join(tmp_path)):
        os.mkdir(os.path.join(tmp_path))
    with zipfile.ZipFile(os.path.join(dataset_path+'.zip'),"r") as archive:
        FILES = archive.namelist()
        for file in FILES:
            if file.endswith('.json'):
                archive.extract(file, path=tmp_path) 
#remove tmp folder
    #shutil.rmtree(os.path.join(tmp_path))
#print(tmp_path)
else:
    output_script = False


print(output_script)
