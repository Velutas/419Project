
% Get list of all of the file names in the positive review folder
% filelist = getAllFiles('.\positive');
% filelist = cellfun(@(x)(x(2:end)), filelist, 'uni', false);

% % Read in the data for all positive reviews
% loadPositiveData
% 
% % Read in the data for all negative reviews
% loadNegativeData

%Load training data
load('TrainingData.mat')
sizeTrainingData=size(TrainingData,1)
%Load testing data
load('TestData.mat')
sizeTestData=size(TestData,1)
%load positive vocabulary words
positiveVocabPath=fullfile('Data','TextODS','positive.ods');
positiveVocabulary=readtable(positiveVocabPath);

%load negative vocabulary words
negativeVocabPath=fullfile('Data','TextODS','negative.ods');
negativeVocabulary=readtable(negativeVocabPath);

negativeVocabulary.Properties.VariableNames{'x2_face'} = 'words';
positiveVocabulary.Properties.VariableNames{'a_'} = 'words';
label_train=zeros(sizeTrainingData,1)
label_test=zeros(sizeTestData,1)
error=0
%strcmp(s1,s2) - compares string values 
for (i=1:sizeTrainingData)   
    review=TrainingData{i}
    review=cell2table(review)
    sizeReview=size(review,1)
    review.Properties.VariableNames{'review'} = 'words';
    negative=intersect(review,negativeVocabulary)
    positive=intersect(review,positiveVocabulary)
    if (size(negative,1)>=size(positive,1))
        label_train(i)=-1
    else label_train(i)=1
    end
    
    if(label_train(i)~=TrainingData{i,2})
        error=error+1
    end   
end
fprintf('error %d', error)

errorTest=0
for (j=1:sizeTestData)   
    review=TestData{j}
    review=cell2table(review)
    sizeReview=size(review,1)
    review.Properties.VariableNames{'review'} = 'words';
    negative=intersect(review,negativeVocabulary)
    positive=intersect(review,positiveVocabulary)
    if (size(negative,1)>=size(positive,1))
        label_test(j)=2
    else label_test(j)=1
    end
    
    if(label_test(j)~=TestData{j,2})
        errorTest=errorTest+1
    end   
end
fprintf('error Test%d', errorTest)
% for i=1:sizeData
%     Data{1}(i)
% end

% % do_kernel_perceptron.m
% 
% MAX_EPOCH=100;
% ETA=0.00001;
% NORMALIZE=0;
% 
% % Kernel parameters
% K_TYPE = 'gaussian';
% K_PARAMS = {5};
% 
% % Positive class
% POS=1;
% 
% % Load data
% load('email.mat');
% 
% Ntrain = size(Ftrain,1);
% 
% % Class POS versus the rest.
% % This sets up L as +/-1 for the two classes.
% L = (Ltrain == POS) - (Ltrain~= POS);
% 
% if NORMALIZE
%   tr_mean = mean(Ftrain,1);
%   tr_std = std(Ftrain,1,1) + eps;
%   Ftrain = Ftrain - repmat(tr_mean, [Ntrain 1]);
%   Ftrain = Ftrain ./ repmat(tr_std, [Ntrain 1]);
% end
% 
% % Train kernel perceptron
% % Compute gram matrix
% K = gramMatrix(Ftrain,Ftrain,K_TYPE,K_PARAMS);
% 
% % Run stochastic gradient descent
% Alpha = zeros(Ntrain,1);
% 
% for epoch=1:MAX_EPOCH
%   rp = randperm(Ntrain);
%   for n_i=rp
%     % TO DO:: Stochastic gradient update.
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%     
%   end
% 
%   % Debug: print out total error.
%   Fn = sign((Alpha') * K)';
%   nerr = sum(Fn ~= L);
%   fprintf('Epoch %d: error %d\n', epoch, nerr);
% 
%   if nerr <=0
%     break
%   end
% end
