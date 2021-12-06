#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 20:35:07 2021

@author: ning
"""

import os
from glob import glob
from shutil import copyfile,rmtree

scripts = [item for item in glob('4.*cross_domain*py') if ("bash" not in item) and ("_split" not in item)]

node                = 1
core                = 16
mem                 = 5 * core * node
cput                = 24 * core * node

bash_folder = 'cross_domain_bash'
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

add = """from shutil import copyfile
copyfile('../utils.py','utils.py')

"""
collections = []
for ii,script in enumerate(scripts):
    _temp = script.split('_')
    decoder = _temp[1]
    experiment = _temp[-1].replace('.py','')
    with open(os.path.join(bash_folder,script),'w') as new_file:
        with open(script,'r') as old_file:
            for line in old_file:
                if "from glob import glob" in line:
                    line = line + '\n' + add
                elif '../' in line:
                    line = line.replace('../','../../')
                new_file.write(line)
            old_file.close()
        new_file.close()
    new_batch_script_name = os.path.join(bash_folder,f'CD{ii+1}')
    content = f"""#!/bin/bash
#PBS -q bcbl
#PBS -l nodes={node}:ppn={core}
#PBS -l mem={mem}gb
#PBS -l cput={cput}:00:00
#PBS -N {decoder}_{experiment}
#PBS -o outputs/out_{experiment}_{ii+1}.txt
#PBS -e outputs/err_{experiment}_{ii+1}.txt

cd $PBS_O_WORKDIR
export PATH="/scratch/ningmei/anaconda3/bin:/scratch/ningmei/anaconda3/condabin:$PATH"
source activate keras-2.1.6_tensorflow-2.0.0
pwd

python "{script}"
"""
    with open(new_batch_script_name,'w') as f:
        f.write(content)
        f.close()
        
        collections.append(f"qsub CD{ii+1}")
    
    with open(f'{bash_folder}/qsub_jobs.py','w') as f:
        f.write("""import os\nimport time""")
    
    with open(f'{bash_folder}/qsub_jobs.py','a') as f:
        for ii,line in enumerate(collections):
            if ii == 0:
                f.write(f'\nos.system("{line}")\n')
            else:
                f.write(f'time.sleep(.3)\nos.system("{line}")\n')
        f.close()




