addpath(genpath('C:\Users\adam.tong\Documents\GitHub\UnitMatch'))
base_folder = 'Z:\ibn-vision\DATA\SUBJECTS\';
mouse = 'M23034';
date = ['20230806'];
ephys_folder = fullfile(base_folder,mouse,'ephys',date);
no_probe = 1;
KS3_dir = fullfile(ephys_folder,['probe',num2str(no_probe)-1],'sorters','kilosort3','sorter_output');
load(fullfile(KS3_dir,'UnitMatch','UnitMatch.mat'));

unit_id = unique(MatchTable.ID1);
MatchProb = reshape(MatchTable.MatchProb,[length(unit_id) length(unit_id)]);
upperTri = triu(MatchProb, 1);
lowerTri = tril(MatchProb, -1);
a= lowerTri';
avgMatrix = (upperTri + lowerTri') / 2;
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
savepath = fullfile(base_folder,mouse,'analysis',date,['probe',num2str(no_probe)-1,'um_merge_suggestion.mat']);

save(savepath,'original_id','merged_id');