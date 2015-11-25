% Load data to sort into training and test data
loadPositiveData;
loadNegativeData;

% Set variables for Training and Test Data
PosNum = 1;
NegNum = 1;
TrainingData{1200,2} = [];
TestData{800,2} = [];

% 1200 Training data
x = randi([0 1],1200,1);
for i=1:1200 
    if x(i) == 1
        TrainingData{i,1} = positiveData{PosNum};
        TrainingData{i,2} = 1;
        PosNum = PosNum + 1;
    else
        TrainingData{i,1} = negativeData{NegNum};
        TrainingData{i,2} = 2;
        NegNum = NegNum + 1;
    end
end
        
% 800 Test data
for i=1:800
    if (x(i) == 1) && (PosNum <= 1000) 
        TestData{i,1} = positiveData{PosNum};
        TestData{i,2} = 1;
        PosNum = PosNum + 1;
    else
        if NegNum <= 1000
            TestData{i,1} = negativeData{NegNum};
            TestData{i,2} = 2;
            NegNum = NegNum + 1;
        else
            TestData{i,1} = positiveData{PosNum};
            TestData{i,2} = 1;
            PosNum = PosNum + 1;
        end
    end
end


save('TestData.mat', 'TestData')
save('TrainingData.mat', 'TrainingData')        
