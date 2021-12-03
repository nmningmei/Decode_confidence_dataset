#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:26:17 2019

@author: nmei
"""
import os
import pandas as pd
from shutil import copyfile

template            = '3.1.5.simple_SVM_LOSO_confidence_split.py'

experiment          = ['confidence','LOO',template.split('_')[1]]
data_dir            = '../data'
working_df_name     = os.path.join(data_dir,experiment[0],experiment[1],'all_data.csv')
df_def              = pd.read_csv(working_df_name,)
node                = 1
core                = 16
mem                 = 5 * core * node
cput                = 24 * core * node

bash_folder = '{}_{}_{}'.format(*experiment)
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

add = """from shutil import copyfile
copyfile('../utils.py','utils.py')

"""

with open(f'{bash_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")
for _key in ['past','recent']:
    collections = []
    for ii,((filename),df_sub) in enumerate(df_def.groupby(["filename"])):
        new_script_name = os.path.join(bash_folder,template.replace('.py',f'_{_key}_{ii+1}.py'))
        with open(new_script_name,'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "filename = '../data/4-point/data_Bang_2019_Exp1.csv'" in line:
                        line = f"filename = '{filename}'\n"
                    elif 'experiment          = ' in line:
                        line = line.replace('past',_key)
                    elif '../' in line:
                        line = line.replace('../','../../')
                    elif "from glob import glob" in line:
                        line = line + '\n' + add
                    elif "verbose             =" in line:
                        line = "verbose             = 0\n"
                    elif "experiment          = " in line:
                        line = line.replace("confidence",experiment[0])
                    elif "debug               = " in line:
                        line = line.replace("True","False")
                    new_file.write(line)
                old_file.close()
            new_file.close()
        
        new_batch_script_name = os.path.join(bash_folder,f'LOO{_key}{ii+1}')
        content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N LOSO_{experiment[-1]}{_key[0]}{ii+1}
#PBS -o outputs/out_{experiment[-1]}_{_key}_{ii+1}.txt
#PBS -e outputs/err_{experiment[-1]}_{_key}_{ii+1}.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd
echo {filename}

python "{template.replace('.py',f'_{_key}_{ii+1}.py')}"
"""
        with open(new_batch_script_name,'w') as f:
            f.write(content)
            f.close()
        
        collections.append("qsub LOO{_key}{ii+1}")
    
    
    
    with open(f'{bash_folder}/qsub_jobs.py','a') as f:
        for ii,line in enumerate(collections):
            if ii == 0:
                f.write(f'\nos.system("qsub LOO{_key}{ii+1}")\n')
            else:
                f.write(f'time.sleep(.3)\nos.system("qsub LOO{_key}{ii+1}")\n')
        f.close()
            

