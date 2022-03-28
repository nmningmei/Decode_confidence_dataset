#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:26:17 2019

@author: nmei
"""
import os
import pandas as pd
from shutil import copyfile,rmtree

template            = '3.2.1.CD_no-split.py'

bash_folder = 'CD_bash_folder'
with open('../.gitignore','r') as f:
    check_bash_folder_name = [bash_folder not in line for line in f]
    f.close()
if all(check_bash_folder_name):
    with open('../.gitignore','a')  as f:
        f.write(f'\n{bash_folder}/')

if os.path.exists(bash_folder):
    rmtree(bash_folder)

os.mkdir(bash_folder)
os.mkdir(os.path.join(bash_folder,'outputs'))
copyfile('utils.py',os.path.join(bash_folder,'utils.py'))

with open(f'{bash_folder}/qsub_jobs.py','w') as f:
    f.write("""import os\nimport time""")

collections = []
for mm,folder_name in enumerate([#'confidence',
                                 #'accuracy',
                                 #'confidence-accuracy',
                                 #'RT',
                                 #'confidence-RT',
                                 'all',
                                 ]):
    for ll,model_name in enumerate(['SVM','RF']):
        for kk,reg_clf in enumerate(['classification','regression']):
            experiment          = [folder_name,'CD',model_name]
            data_dir            = '../data'
            node                = 1
            core                = 12
            mem                 = 4
            cput                = 24
            new_script_name = os.path.join(bash_folder,template.replace('.py',f'_{folder_name}_{model_name}_{reg_clf}.py'))
            with open(new_script_name,'w') as new_file:
                with open(template,'r') as old_file:
                    for line in old_file:
                        if "# change model name" in line:
                            line = f"        model_name          = '{model_name}'\n"
                        elif "# change type" in line:
                            line = f"        reg_clf             = '{reg_clf}'\n"
                        elif "change attributes"  in line:
                            line = f"target_attributes = '{folder_name}'\n"
                        elif '../' in line:
                            line = line.replace('../','../../')
                        elif "verbose             = " in line:
                            line = "        verbose             = 0\n"
                        elif "debug               = " in line:
                            line = line.replace("True","False")
                        new_file.write(line)
                    old_file.close()
                new_file.close()
            
            new_batch_script_name = os.path.join(bash_folder,f'CD{mm+1}{ll+1}{kk+1}')
#             content = f"""#!/bin/bash
# #SBATCH --partition=regular
# #SBATCH --job-name={mm+1}{ll+1}{kk+1}
# #SBATCH --cpus-per-task={core}
# #SBATCH --nodes={node}
# #SBATCH --ntasks-per-node=1
# #SBATCH --time={cput}:00:00
# #SBATCH --mem-per-cpu={mem}G
# #SBATCH --output=outputs/out_{mm+1}{ll+1}{kk+1}.txt
# #SBATCH --error=outputs/err_{mm+1}{ll+1}{kk+1}.txt
# #SBATCH --mail-user=nmei@bcbl.eu

# source /scratch/ningmei/.bashrc
# conda activate bcbl
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/ningmei/anaconda3/lib
# module load FSL/6.0.0-foss-2018b
# cd $SLURM_SUBMIT_DIR

# pwd
# python3 "{new_script_name.split('/')[-1]}"
# """
            content = f"""#!/bin/bash
#$ -cwd
#$ -o outputs/out_{mm+1}{ll+1}{kk+1}.txt
#$ -e outputs/err_{mm+1}{ll+1}{kk+1}.txt
#$ -m be
#$ -M nmei@bcbl.eu
#$ -N S{mm+1}{ll+1}{kk+1}
#$ -q long.q
#$ -S /bin/bash

module load python/python3.6 fsl/6.0.0

python3 "{new_script_name.split('/')[-1]}"
"""
            with open(new_batch_script_name,'w') as f:
                f.write(content)
                f.close()
                
            collections.append(f"sbatch CD{mm+1}{ll+1}{kk+1}")
            
with open(f'{bash_folder}/qsub_jobs.py','a') as f:
    for ii,line in enumerate(collections):
        if ii == 0:
            f.write(f'\nos.system("{line}")\n')
        else:
            f.write(f'time.sleep(1)\nos.system("{line}")\n')
    f.close()


