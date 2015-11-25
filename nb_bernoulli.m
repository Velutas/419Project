clear % Loading train data and setting number of words and labels
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
vocab = vertcat(negativeVocabulary,positiveVocabulary);
no_of_words = size(vocab,1);
labels ={1,2} ;
no_of_labels = 2;

% Loading training data
train_data = TrainingData(:,1);
train_labels = TrainingData(:,2);
no_of_docs = length(train_labels);

% setting laplacian coefficient = 1
laplacian = 1;

%Calculating priors
priors = zeros(no_of_labels,1);
train_labels=cell2mat(train_labels);
labels=cell2mat(labels);
for i=1:no_of_labels
   priors(i) = ((length(find(train_labels(:,1)== labels(i))))/(length(train_labels))); 
end

%Building (Words) vs (Labels) matrix
words_labels = zeros(no_of_words,no_of_labels);

for i=1:no_of_labels
   docIds = find(train_labels == labels(i));
   for j=1:numel(docIds)
       index = docIds(j);
       words_labels(index,i) = words_labels(index,i) + 1;
   end
end

%load test datartest_data(:,2)
test_data = TestData(:,1);
test_labels = TestData(:,2);
test_docs_length = length(test_labels);


%Calculating posteriors
posteriors_combined = zeros(test_docs_length,no_of_labels);
test_labels=cell2mat(test_labels);
for i = 1:test_docs_length
      
    wordIds = test_data{i};
    %indices = find(test_data(:,1) == i);  
    for j=1:no_of_labels
        posteriors_combined(i,j) = sum(log((words_labels(i,j) + laplacian)/(numel(find(train_labels(:,1) == labels(j))) + (laplacian * 2))) + log(priors(j))) ;
        %posteriors_combined(i,j) = sum(log((words_labels(wordIds,j) + laplacian)/(numel(find(train_labels == j)) + (laplacian * no_of_words))) + log(priors(j))) ;
        %posteriors_combined(i,j) = sum(((words_labels(wordIds,j) + laplacian)/(numel(find(train_labels == j)) + (laplacian * no_of_words))) * (priors(j))) ;
    end
end


z = 1:length(test_labels);
[dummy,posteriors] = max((posteriors_combined(z,:)),[],2);

errors = posteriors(:)-test_labels(:);
errors(errors == 0) = [];
no_of_errors = length(errors);

classifier_accuracy = ((1-(no_of_errors/length(test_labels)))*100);

%fprintf('Classifier accuracy is %d',classifier_accuracy);
fprintf('Classifier accuracy is %.2f\n',classifier_accuracy);










