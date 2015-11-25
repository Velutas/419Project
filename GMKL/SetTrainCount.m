load('TrainingData.mat')
% Words = readtable('negative.xls');
Words = textread('positive.csv', '%s', 'whitespace', ',');
% Words = textread('negative.csv', '%s', 'whitespace', ',');

Words = Words(1:1280,1);

Vocab = 1280;
Reviews = 1200;

TempFeatures = zeros(1280,Reviews);

for ReviewNum = 1:Reviews
    ReviewWords = size(TrainingData{ReviewNum});
    
    for VocabWord = 1:Vocab
        for Word = 1:ReviewWords(1)
            
%             if strcmp(TrainingData{ReviewNum}(Word) , Words{VocabWord,1}) == 1;
%                 TempFeatures(VocabWord,ReviewNum) = TempFeatures(VocabWord,ReviewNum) + 1; 
%             end
            if strcmp(TrainingData{ReviewNum}(Word) , Words(VocabWord)) == 1;
                TempFeatures(VocabWord,ReviewNum) = TempFeatures(VocabWord,ReviewNum) + 1; 
            end
            
        end
    end
    sprintf('FinishedOneReview %d', ReviewNum)
end
