load Newexample;


for i = 1:2959
    x = 0;
    for j = 1:600
        if TRNfeatures(i,j) >=1
            x = x + 1;
        end
    end
    for k = 1:600
        TRNfeatures(i,k) = TRNfeatures(i,k)*x;
        TSTfeatures(i,k) = TSTfeatures(i,k)*x;
%       Alt versions
%       if TRNfeatures(i,k) >= 1
%           TRNfeatures(i,k) = x;
%       end
% 
    end
end



save('Newexample2.mat', 'TRNfeatures', 'TRNlabels', 'TSTfeatures', 'TSTlabels');