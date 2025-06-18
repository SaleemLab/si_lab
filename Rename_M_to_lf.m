% Define the root directory containing all data folders
rootDir = 'Z:\ibn-vision\DATA\SUBJECTS\M24018\ephys\20240719\20240719_3\';  % <-- change to your actual path
% dataFolders = dir(fullfile(rootDir, 'M24018_20240719_3_g*'));
dataFolders = dir(fullfile(rootDir, '*_g*'));

for i = 1:length(dataFolders)
    gPath = fullfile(rootDir, dataFolders(i).name);
    
    % Look for imec0 and imec1 subfolders
    imecFolders = dir(fullfile(gPath, '*imec*'));
    
    for j = 1:length(imecFolders)
        imecPath = fullfile(gPath, imecFolders(j).name);
        
        % Find .ap.bin file
        apFile = dir(fullfile(imecPath, '*imec*.ap.bin'));
        if isempty(apFile)
            fprintf('No .ap.bin file in %s\n', imecPath);
            continue;
        end
        
        [~, apBaseName, ~] = fileparts(apFile(1).name);
        newBaseName = strrep(strrep(apBaseName, 't0', 'tcat'), '.ap', '.lf');
        
        % New filenames
        newBin = fullfile(imecPath, [newBaseName, '.bin']);
        newMeta = fullfile(imecPath, [newBaseName, '.meta']);

        % Filter strictly M followed by digits using regexp
        allFiles = dir(imecPath);
        for k = 1:length(allFiles)
            fname = allFiles(k).name;

            if regexp(fname, '^M\d+\.bin$')
                movefile(fullfile(imecPath, fname), newBin);
                fprintf('Renamed: %s -> %s\n', fname, newBin);
            elseif regexp(fname, '^M\d+\.meta$')
                movefile(fullfile(imecPath, fname), newMeta);
                fprintf('Renamed: %s -> %s\n', fname, newMeta);
            end
        end
    end
end
