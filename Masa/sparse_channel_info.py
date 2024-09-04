import json

base_folder = 'Z:\\ibn-vision\\DATA\\SUBJECTS\\'
mouse = 'M23034'
date = '20230805'
ephys_folder = base_folder + mouse + '\\ephys\\' + date +'\\'

# Specify the file path
probe0_ks3_sparsity_path = ephys_folder+'probe0\\waveform\\kilosort3\\'+'sparsity.json'

# Load the JSON file
with open(probe0_ks3_sparsity_path, 'r') as file:
    probe0_ks3_sparsity = json.load(file)

# Access the dictionary
size = len(probe0_ks3_sparsity['unit_id_to_channel_ids'])
print(size)
