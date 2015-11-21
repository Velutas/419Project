%POSITIVE DATA

% Get path to all of the file names in the positive review folder
positiveFileListPath=fullfile('Data','TrainingData','positiveStemmedTrainingData')

for i=1:1000

    positiveFilePath=strcat(positiveFileListPath,'\file',int2str(i),'.txt')
    positiveFile = fopen(positiveFilePath, 'r');
    
    % Scan in the data from the .txt file and store in data cell, Data{1}
    positiveData(i) = textscan(positiveFile, '%s');
    sizeData = size(positiveData{i});

    % Strip invalid characters from the .txt files
    %positiveData{i} = (regexprep(positiveData{i},'[\"123\]\[4567890]', ''));
    
    fclose(positiveFile);
end