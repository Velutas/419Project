% Initalize Ints
% FileStart = 'U:\419Project'
% 
% Read in the data for all positive reviews
% 
% % Get list of all of the file names in the positive review folder
% filelist = getAllFiles('.\positive');
% filelist = cellfun(@(x)(x(2:end)), filelist, 'uni', false);

loadPositiveData
loadNegativeData


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
