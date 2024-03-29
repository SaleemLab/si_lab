%% Unitmatch DT implementation
base_folder = 'Z:\ibn-vision\DATA\SUBJECTS\';
mouse = 'M23034';
date = ['20230805'];
no_probe = 1;
ephys_folder = fullfile(base_folder,mouse,'ephys',date);

UMparam.KSDir = cell(1,2);
UMparam.AllDecompPaths = cell(1,2);
UMparam.AllChannelPos = cell(1,2);
UMparam.SaveDir = fullfile(ephys_folder,['probe',num2str(no_probe)-1],'sorters','unit_match_sorter_comparison'); 

clusinfo = struct; % Note, this can be kilosort input, 
clusinfo.cluster_id = [];
clusinfo.Good_ID = [];
clusinfo.ProbeID = [];
clusinfo.RecSesID=[];
mkdir(UMparam.SaveDir);

sorters = {'kilosort2_5';'kilosort3'};
for iSorter = 1:2

%UMparam.SaveDir = fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1],'unit_match'); % Recommended to use end this path with \Probe0\IMRO_1\ if more probes/IMRO tables were used or \AllProbes\AllIMRO\ otherwise


UMparam.KSDir{iSorter} = fullfile(ephys_folder,['probe',num2str(no_probe)-1],'sorters',sorters{iSorter,:},'sorter_output');  % This is a cell array with a path, in the path there should be a subfolder called 'RawWaveforms'. 
% N.B. if you want to use the functional score evaluation of UnitMatch, 'KSDir' should also contain typical 'Kilosort output', (e.g. spike times etc.)

%% N.B. the following user input can also be automatically extracted and prepared/cleaned up using UMparam = ExtractKilosortData(KiloSortPaths, UMparam) for Kilosorted data of SpikeGLX recorded data (see next section);
%UMparam.RawDataPaths = {'\\path\to\firstrecording','\\path\to\secondrecording','\\path\to\nthrecording'};  % This is a cell array with info on where to find the compressed recording (.cbin files OR .bin files)

UMparam.AllDecompPaths{iSorter} = fullfile(['probe',num2str(no_probe)-1,'preprocessed']);  % This is a cell array with info on where to find the decompressed recording (.bin files) --> Necessary when you want UnitMatch to do waveform extraction

UMparam.AllChannelPos{iSorter} = readNPY(fullfile(ephys_folder,['probe',num2str(no_probe)-1],['sorters\',sorters{iSorter,:},'\sorter_output\channel_positions.npy'])); % These are coordinates of every recording channel on the probe (e.g. nRecordingChannels x 2)


%% convert spikeinterface waveforms to unit match version in kilosort folder

% Specify the file path
probe0_ks3_sparsity_path = fullfile(ephys_folder,['probe',num2str(no_probe)-1],['waveform\',sorters{iSorter,:},'\sparsity.json']);
probe0_ks3_waveform_path = fullfile(ephys_folder,['probe',num2str(no_probe)-1],['waveform\',sorters{iSorter,:},'\waveforms\']);
probe0_ks3_raw_waveform_path =fullfile(UMparam.KSDir{iSorter},'RawWaveforms');
mkdir(probe0_ks3_raw_waveform_path);
% Load the JSON file
fileID = fopen(probe0_ks3_sparsity_path);
rawData = fread(fileID, inf);
strData = char(rawData');
fclose(fileID);
no_channels = size(UMparam.AllChannelPos{iSorter},1);
probe0_ks3_sparsity = jsondecode(strData);

unit_ids = probe0_ks3_sparsity.unit_ids;

for iUnit = 1:length(unit_ids)
    waveform_channels = probe0_ks3_sparsity.unit_id_to_channel_ids.(['x',num2str(unit_ids(iUnit))]);
    % Assume 'cellArray' is your cell array
    
    
    % Initialize an empty matrix of the same size as the cell array
    waveform_channel_ids = zeros(size(waveform_channels));
    
    % Loop over the cell array
    for i = 1:numel(waveform_channels)
        % Extract the number from the string using regexp

        temp_channel_indices = strcmp(probe0_ks3_sparsity.channel_ids,waveform_channels{i});
        waveform_channel_ids(i) = find(temp_channel_indices ==1);
    end
    unit_waveform_path = fullfile(probe0_ks3_waveform_path,['waveforms_',num2str(unit_ids(iUnit)),'.npy']);
    unit_waveform = readNPY(unit_waveform_path);
    unit_waveform = permute(unit_waveform,[2 3 1]);
    spikeMap = zeros(size(unit_waveform,1),no_channels,size(unit_waveform,3));
    
    spikeMap(:,waveform_channel_ids,:) = unit_waveform;
    spikeMapAvg = zeros(size(unit_waveform,1),no_channels,2);
    nwavs = size(spikeMap,3);
    for cv = 1:2
            if cv==1
                wavidx = floor(1:nwavs/2);
            else
                wavidx = floor(nwavs/2+1:nwavs);
            end
            spikeMapAvg(:,:,cv) = nanmedian(spikeMap(:,:,wavidx),3);
     end
        spikeMap = spikeMapAvg;
        
    %fetch the waveforms of the unit
    writeNPY(spikeMap, [UMparam.KSDir{iSorter},'\RawWaveforms\','Unit',num2str(unit_ids(iUnit)),'_RawSpikes.npy']);
end
%


clusinfo.cluster_id = [clusinfo.cluster_id;unit_ids];
clusinfo.Good_ID = [clusinfo.Good_ID;zeros(size(unit_ids))];
clusinfo.ProbeID = [clusinfo.ProbeID;ones(size(unit_ids)).*no_probe];
clusinfo.RecSesID = [clusinfo.RecSesID;ones(size(unit_ids))*iSorter];
end
UMparam = DefaultParametersUnitMatch(UMparam);
UMparam.GoodUnitsOnly = 0;
UMparam.spikeWidth = 105;
%%
[UniqueIDConversion, MatchTable, WaveformInfo, UMparam] = UnitMatch(clusinfo, UMparam);
if UMparam.AssignUniqueID
    AssignUniqueID(UMparam.SaveDir);
end