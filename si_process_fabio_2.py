def si_process_fabio_2(base_folder, mouse, date,dst_folder,job_kwargs):
    
    ephys_folder = base_folder + mouse + '/ephys/' + date +'/'
    

    import shutil
    import os

    folders_to_move = ['probe0',
                    'probe1',
                    'probe0_preprocessed',
                    'probe1_preprocessed']
##
#
    for folder in folders_to_move:
        # construct the destination path
        destination = os.path.join(ephys_folder, folder)
        # copy the folder to the destination
        shutil.copytree(dst_folder+folder, destination)
#
    