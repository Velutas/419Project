%NEGATIVE DATA

% Get path to all of the file names in the negative review folder
negativeFileListPath=fullfile('Data','TrainingData','negativeStemmedTrainingData')

for i=1:1000

    negativeFilePath=strcat(negativeFileListPath,'\file',int2str(i),'.txt')
    negativeFile = fopen(negativeFilePath, 'r');
    
    % Scan in the data from the .txt file and store in data cell, Data{1}
    negativeData(i) = textscan(negativeFile, '%s');
    sizeData = size(negativeData{i});

    % Strip invalid characters from the .txt files
    %negativeData{i} = (regexprep(negativeData{i},'[\"123\]\[4567890]', ''));
    
    fclose(negativeFile);
end