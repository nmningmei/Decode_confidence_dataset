#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:26:17 2019

@author: nmei
"""
import os
import pandas as pd
import numpy as np

from glob import glob
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils           import shuffle as util_shuffle
from shutil                  import copyfile

templates           = glob('4.1*RNN*cross_domain.py')

node                = 1
core                = 16
mem                 = 2 * core * node
cput                = 24 * core * node

add = """from shutil import copyfile
copyfile('../utils.py','utils.py')

"""
bash_folder = 'CD_bash'
with open('../.gitignore','r') as f:
    check_bash_folder_name = [bash_folder not in line for line in f]
    f.close()
if all(check_bash_folder_name):
    with open('../.gitignore','a')  as f:
        f.write(f'\n{bash_folder}/')
if not os.path.exists(bash_folder):
    os.mkdir(bash_folder)
    os.mkdir(os.path.join(bash_folder,'outputs'))
copyfile('utils.py',os.path.join(bash_folder,'utils.py'))

add_on = """from shutil import copyfile
copyfile('../utils.py','utils.py')

"""

if not os.path.exists(f'{bash_folder}/outputs'):
    os.mkdir(f'{bash_folder}/outputs')

collection = []
for template in templates:
    for ii,con in enumerate(['confidence','adequacy']):
        new_script_name = os.path.join(bash_folder,template.replace('.py',f'_{con}.py'))
        with open(new_script_name,'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if '../' in line:
                        line = line.replace('../','../../')
                    elif 'add here' in line:
                        line = add_on
                    elif "from glob" in line:
                        line = line + '\n' + add + '\n'
                    elif "verbose             =" in line:
                        line = "verbose             = 0\n"
                    elif "experiment          = " in line:
                        line = line.replace('confidence',con)
                    new_file.write(line)
                old_file.close()
            new_file.close()
        collection.append(new_script_name)

with open(f'{bash_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")
    f.close()

with open(f'{bash_folder}/qsub_jobs.py','a') as f:
    for ii,row in enumerate(collection):
        new_batch_script_name = os.path.join(bash_folder,f'{row.split("_")[2]}_{row.split("_")[-1].split(".")[0]}')
        content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N {row.split("_")[2]}_{row.split("_")[-1].split(".")[0]}
#PBS -o outputs/out_{ii+1}.txt
#PBS -e outputs/err_{ii+1}.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo {row.split("_")[2]}_{row.split("_")[-1].split(".")[0]}

python "{row.split('/')[-1]}"
    """
        with open(new_batch_script_name,'w') as ff:
            ff.write(content)
            ff.close()
        f.write(f'\nos.system("qsub {row.split("_")[2]}_{row.split("_")[-1].split(".")[0]}")\n')
    f.close()

