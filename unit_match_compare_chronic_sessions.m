%% Unitmatch DT implementation



base_folder = 'Z:\ibn-vision\DATA\SUBJECTS\';
mouse = 'M23087';
date = {['20231207'];['20231208'];['20231212'];['20231214_OpenField'];['20231215_OpenField']};
ephys_folder = cell(1,size(date,1));
UMparam.KSDir = cell(1,size(date,1));
UMparam.AllDecompPaths = cell(1,size(date,1));
UMparam.AllChannelPos = cell(1,size(date,1));
UMparam.SaveDir = fullfile(base_folder,mouse,'ephys','unit_match_output'); 
clusinfo = struct; % Note, this can be kilosort input, 
clusinfo.cluster_id = [];
clusinfo.Good_ID = [];
clusinfo.ProbeID = [];
clusinfo.RecSesID=[];
mkdir(UMparam.SaveDir);
no_probe = 1;
for iDate = 1:size(date,1)
ephys_folder{iDate} = fullfile(base_folder,mouse,'ephys',date{iDate,:});

%UMparam.SaveDir = fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1],'unit_match'); % Recommended to use end this path with \Probe0\IMRO_1\ if more probes/IMRO tables were used or \AllProbes\AllIMRO\ otherwise


UMparam.KSDir{iDate} = fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1],'sorters','kilosort3','sorter_output');  % This is a cell array with a path, in the path there should be a subfolder called 'RawWaveforms'. 
% N.B. if you want to use the functional score evaluation of UnitMatch, 'KSDir' should also contain typical 'Kilosort output', (e.g. spike times etc.)

%% N.B. the following user input can also be automatically extracted and prepared/cleaned up using UMparam = ExtractKilosortData(KiloSortPaths, UMparam) for Kilosorted data of SpikeGLX recorded data (see next section);
%UMparam.RawDataPaths = {'\\path\to\firstrecording','\\path\to\secondrecording','\\path\to\nthrecording'};  % This is a cell array with info on where to find the compressed recording (.cbin files OR .bin files)

UMparam.AllDecompPaths{iDate} = fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1,'preprocessed']);  % This is a cell array with info on where to find the decompressed recording (.bin files) --> Necessary when you want UnitMatch to do waveform extraction

UMparam.AllChannelPos{iDate} = readNPY(fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1],'sorters\kilosort3\sorter_output\channel_positions.npy')); % These are coordinates of every recording channel on the probe (e.g. nRecordingChannels x 2)


%% convert spikeinterface waveforms to unit match version in kilosort folder

% Specify the file path
probe0_ks3_sparsity_path = fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1],'waveform\kilosort3\sparsity.json');
probe0_ks3_waveform_path = fullfile(ephys_folder{iDate},['probe',num2str(no_probe)-1],'waveform\kilosort3\waveforms\');
probe0_ks3_raw_waveform_path =fullfile(UMparam.KSDir{iDate},'RawWaveforms');
mkdir(probe0_ks3_raw_waveform_path);
% Load the JSON file
fileID = fopen(probe0_ks3_sparsity_path);
rawData = fread(fileID, inf);
strData = char(rawData');
fclose(fileID);
no_channels = size(UMparam.AllChannelPos{iDate},1);
probe0_ks3_sparsity = jsondecode(strData);

unit_ids = probe0_ks3_sparsity.unit_ids;
% 
% for iUnit = 1:length(unit_ids)
%     waveform_channels = probe0_ks3_sparsity.unit_id_to_channel_ids.(['x',num2str(unit_ids(iUnit))]);
%     % Assume 'cellArray' is your cell array
%     
%     
%     % Initialize an empty matrix of the same size as the cell array
%     waveform_channel_ids = zeros(size(waveform_channels));
%     
%     % Loop over the cell array
%     for i = 1:numel(waveform_channels)
%         % Extract the number from the string using regexp
% 
%         temp_channel_indices = strcmp(probe0_ks3_sparsity.channel_ids,waveform_channels{i});
%         waveform_channel_ids(i) = find(temp_channel_indices ==1);
%     end
%     unit_waveform_path = fullfile(probe0_ks3_waveform_path,['waveforms_',num2str(unit_ids(iUnit)),'.npy']);
%     unit_waveform = readNPY(unit_waveform_path);
%     unit_waveform = permute(unit_waveform,[2 3 1]);
%     spikeMap = zeros(size(unit_waveform,1),no_channels,size(unit_waveform,3));
%     
%     spikeMap(:,waveform_channel_ids,:) = unit_waveform;
%     spikeMapAvg = zeros(size(unit_waveform,1),no_channels,2);
%     nwavs = size(spikeMap,3);
%     for cv = 1:2
%             if cv==1
%                 wavidx = floor(1:nwavs/2);
%             else
%                 wavidx = floor(nwavs/2+1:nwavs);
%             end
%             spikeMapAvg(:,:,cv) = nanmedian(spikeMap(:,:,wavidx),3);
%      end
%         spikeMap = spikeMapAvg;
%         
%     %fetch the waveforms of the unit
%     writeNPY(spikeMap, [UMparam.KSDir{iDate},'\RawWaveforms\','Unit',num2str(unit_ids(iUnit)),'_RawSpikes.npy']);
% end



clusinfo.cluster_id = [clusinfo.cluster_id;unit_ids];
clusinfo.Good_ID = [clusinfo.Good_ID;zeros(size(unit_ids))];
clusinfo.ProbeID = [clusinfo.ProbeID;ones(size(unit_ids)).*no_probe];
clusinfo.RecSesID = [clusinfo.RecSesID;ones(size(unit_ids))*iDate];
end
UMparam = DefaultParametersUnitMatch(UMparam);
UMparam.GoodUnitsOnly = 0;
UMparam.spikeWidth = 105;
%%
[UniqueIDConversion, MatchTable, WaveformInfo, UMparam] = UnitMatch(clusinfo, UMparam);
if UMparam.AssignUniqueID
    AssignUniqueID(UMparam.SaveDir);
end
MatchTable.ID1 = MatchTable.ID1+1;
MatchTable.ID2 = MatchTable.ID2+1;
no_session = length(unique(MatchTable.RecSes1));
match_ids = cell(no_session,1);
unit_id_no = zeros([no_session 1]);

for iSes = 1:no_session
    session_idx = MatchTable.RecSes1 == iSes & MatchTable.RecSes2 == iSes;
    
        unit_id = unique(MatchTable.ID1(session_idx));

        unit_id_no(iSes) = length(unit_id);

        unique_id_idx = (sum(unit_id_no(1:iSes))+1-unit_id_no(iSes)):sum(unit_id_no(1:iSes));

        unique_id = UniqueIDConversion.UniqueID(unique_id_idx);

        MatchProb = reshape(MatchTable.MatchProb(session_idx),[length(unit_id) length(unit_id)]);

        upperTri = triu(MatchProb, 1);

        lowerTri = tril(MatchProb, -1);

        a= lowerTri';

        avgMatrix = (upperTri + lowerTri') / 2;

        self_match_prob = spdiags(MatchProb,0);

        unstable_id = self_match_prob < 0.5;

        merged_id = unit_id; %pre-allocate the unit id to the merged id
        
        for id_count = 1:length(unit_id)
            id = unit_id(id_count);

            match_id = find(avgMatrix(id_count,:) >= 0.85);
            if ~isempty(match_id)
                if merged_id(id_count) == id
                    merged_id(match_id) = id;
                else
                    merged_id(match_id) = merged_id(id_count);
                end
            end

        end
        original_id = unit_id;
        unique_merged_ids = sort(unique(merged_id));

[id_counts, edges] = histcounts(merged_id,[unique_merged_ids; unique_merged_ids(end)+1]-0.5); % find ids that occur more than once

ids_to_merge = unique_merged_ids(id_counts > 1); % these ids have other original ids merged to this id so will only look at these clusters

for number_id = 1:length(ids_to_merge)

    id_temp = ids_to_merge(number_id);
    
    temp_merge_index = find(merged_id == id_temp); % find the original ids of the merged ones

    unique_id(temp_merge_index) = unique_id(temp_merge_index(1));  %convert cluster ids to merged ids
   
end

        match_ids{iSes} = [original_id,merged_id,unstable_id,unique_id'];
end

save('Z:\ibn-vision\DATA\SUBJECTS\M23087\analysis\all_unit_match.mat','match_ids') %['20231207';'20231208';'20231212';'20231214_OpenField';'20231215_OpenField'];