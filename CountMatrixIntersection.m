%% Count Matrix Using Intersection
% Anahita Berenji
% This code implements a matrix representing the number of times each
% negative word is repeated in a review. Rows represent the 1200 reviews
% and the columns represent the 2960 possuble negative words

%Load training data
load('TrainingData.mat');
sizeTrainingData=size(TrainingData,1);

%Load testing data
load('TestData.mat');
sizeTestData=size(TestData,1);

% %load positive vocabulary words
% positiveVocabPath=fullfile('Data','TextODS','positive.ods');
% positiveVocabulary=readtable(positiveVocabPath);

%load negative vocabulary words
[num,txt,raw] = xlsread('negative.xls');
negativeVocab=txt;
sizeNegativeVocab=size(negativeVocab,1);
CMatrix=zeros(sizeTrainingData,sizeNegativeVocab);
Rev=TrainingData(:,1);
for i=1:sizeTrainingData;
    [CC,iNeg,iRev]=intersect(negativeVocab,Rev{i});      %Rev is a cell array hence the {}
    CMatrix(i,iNeg)=1;
end
