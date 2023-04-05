# 
# MODEL SAVE STATE SEARCH FUNCTION 
# BY : CAMERON KAMINSKI
# DESCRIPTION : FUNCTION FOR SEARCHING THROUGH A DIR (AND ITS SUB-DIR) TO
# PERFORM SOME GIVEN FUNCTION ON THE MODEL SAVES.
# 

import os

def process_modelstate(root_dir : str, function):
"""
MODEL STATE PROCESSOR, PROPAGATES THROUGH A DIR FOR PYTORCH MODEL STATES, 
THEN PERFORMS THE SPECIFIED FUNCTION TO EACH OF THE STATES.
:param root_dir: ROOT DIRECTORY 
:param function: SOME ARBITRARY FUNCTION TO PERFORM ON THE MODEL STATES.
:returns NONE:
"""
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                file_path = os.path.join(dirpath, filename)
                function(file_path)
        for dirname in dirnames:
        subdir_path = os.path.join(dirpath, dirname)
        process_files_with_extension(subdir_path, function)
    
