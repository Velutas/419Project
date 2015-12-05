load NegWordReview
load toyexample
load TrainingData

Temp1 = zeros(2959,600);
Temp2 = zeros(2959,600);

labels1 = zeros(600,1);
labels2 = zeros(600,1);

for i=1:600
    Temp1(1:2959,i) = NegWordReviews(1:2959,i);
    Temp2(1:2959,i) = NegWordReviews(1:2959,i+600);
end

TRNfeatures = Temp1;
TSTfeatures = Temp2;

for i=1:600
    if TrainingData{i,2} == 0
        labels1(i) = -1;
    elseif TrainingData{i,2} == 1
        labels1(i) = 1;
    end
    
    if TrainingData{i+600,2} == 0
        labels2(i) = -1;
    elseif TrainingData{i+600,2} == 1
        labels2(i) = 1;
    end
end
    
TRNlabels = labels1;
TSTlabels = labels2;

ans = 1;
save('Newexample.mat', 'TRNfeatures', 'TRNlabels', 'TSTfeatures', 'TSTlabels', 'ans');