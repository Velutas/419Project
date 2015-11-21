%POSITIVE DATA

% Get path to all of the file names in the positive review folder
positiveFileListPath=fullfile('Data','TrainingData','positive')
%positiveFileList=getAllFiles(positiveFileListPath)

for i=1:1000
%     filelist(i) = strrep(filelist(i), '.', '');
    %FullName = strcat(FileStart, filelist(i));
    
    
%     check = 'U:\419Project\positive\file1.txt';
    %file = fopen('U:\419Project\positive\file1.txt', 'r');
    positiveFilePath=strcat(positiveFileListPath,'\file',int2str(i),'.txt')
    positiveFile = fopen(positiveFilePath, 'r');
    
    % Scan in the data from the .txt file and store in data cell, Data{1}
    positiveData(i) = textscan(positiveFile, '%s');
    sizeData = size(positiveData{i});

    % Strip invalid characters from the .txt files
    positiveData{i} = (regexprep(positiveData{i},'[\"123\]\[4567890]', ''));
    
    fclose(positiveFile);
end