#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 14:26:17 2019

@author: nmei
"""
import os
import pandas as pd
from shutil import copyfile,rmtree

template            = '3.1.1.SVM_LOO_no-split.py'
for jj,folder_name in enumerate(['confidence','accuracy','confidence-accuracy']):
    experiment          = [folder_name,'LOO','SVM']
    data_dir            = '../data'
    working_df_name     = os.path.join(data_dir,experiment[0],experiment[1],'all_data.csv')
    df_def              = pd.read_csv(working_df_name,)
    node                = 1
    core                = 16
    mem                 = 4
    cput                = 24
    
    bash_folder = '{}_{}_{}'.format(*experiment)
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
    
    
    collections = []
    for ii,((filename),df_sub) in enumerate(df_def.groupby(["filename"])):
        new_script_name = os.path.join(bash_folder,template.replace('.py',f'_{jj+1}_{ii+1}.py'))
        with open(new_script_name,'w') as new_file:
            with open(template,'r') as old_file:
                for line in old_file:
                    if "change file name" in line:
                        line = f"filename = '{filename}'\n"
                    elif "change folder name"  in line:
                        line = f"target_attributes   = '{folder_name}'\n"
                    elif '../' in line:
                        line = line.replace('../','../../')
                    elif "verbose             =" in line:
                        line = "verbose             = 0\n"
                    elif "experiment          = " in line:
                        line = line.replace("confidence",experiment[0])
                    elif "debug               = " in line:
                        line = line.replace("True","False")
                    new_file.write(line)
                old_file.close()
            new_file.close()
        
        new_batch_script_name = os.path.join(bash_folder,f'LOO{jj+1}{ii+1}')
        content = f"""#!/bin/bash
#SBATCH --partition=regular
#SBATCH --job-name=RSA{ii}
#SBATCH --cpus-per-task={core}
#SBATCH --nodes={node}
#SBATCH --ntasks-per-node=1
#SBATCH --time={cput}:00:00
#SBATCH --mem-per-cpu={mem}G
#SBATCH --output=outputs/out_{ii}.txt
#SBATCH --error=outputs/err_{ii}.txt
#SBATCH --mail-user=nmei@bcbl.eu

source /scratch/ningmei/.bashrc
conda activate bcbl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/scratch/ningmei/anaconda3/lib
module load FSL/6.0.0-foss-2018b
cd $SLURM_SUBMIT_DIR

pwd
echo {filename}
python3 "{new_script_name}"
"""

        with open(new_batch_script_name,'w') as f:
            f.write(content)
            f.close()
        
        collections.append("qsub LOO{jj+1}{ii+1}")
    
    with open(f'{bash_folder}/qsub_jobs.py','w') as f:
        f.write("""import os\nimport time""")
    
    with open(f'{bash_folder}/qsub_jobs.py','a') as f:
        for ii,line in enumerate(collections):
            if ii == 0:
                f.write(f'\nos.system("qsub LOO{jj+1}{ii+1}")\n')
            else:
                f.write(f'time.sleep(.3)\nos.system("qsub LOO{jj+1}{ii+1}")\n')
        f.close()
            

