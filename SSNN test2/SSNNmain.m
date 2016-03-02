

hiddenSize1 = 587;
hiddenSize2 = 19;
hiddenSize3 = 315;
inputSize = 3522;
sampledSize = 6;
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term    

%load data

wdms = load( 'wdms_1478.mat');
wdms = wdms.P2;
non = load('NonWDMs10000.mat');
non = non.data;

train_x = [wdms(1:1000,:);non(1:7000,:)];

train_y = [ones(1000,1); zeros(7000,1)];

train_x = mapminmax(train_x,0,1)';
train_y = train_y';
[~,m] = size(train_x);

%----------------------Train The First Ply---------------

%initial the paragraments
ssnn1Theta = sampledinitializeParameters(hiddenSize1,inputSize);

% %cost function
% [cost,grad] = sampledsparseAutoencoderCost(ssnn1Theta,inputSize,hiddenSize1,sampledSize,lambda,sparsityParam,beta,train_x);
% 
% %gradient checking
% numgrad = computeNumericalGradient( @(x) sparseAutoencoderCost(x, inputSize, ...
%                                                   hiddenSize1, sampledSize,lambda, ...
%                                                   sparsityParam, beta, ...
%                                                   t), ssnn1Theta);
% 
% % Use this to visually compare the gradients side by side
% disp([numgrad grad]); 
% % Compare numerically computed gradients with the ones obtained from backpropagation
% diff = norm(numgrad-grad)/norm(numgrad+grad);
% disp(diff); % Should be small. In our implementation, these values are
%             % usually less than 1e-9.
% 
%             
            
            
%train the first ply
addpath minFunc/
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'on';
[OptTheta1, loss] = minFunc( @(p) sampledsparseAutoencoderCost(p, ...
      inputSize,hiddenSize1,sampledSize,lambda,sparsityParam,beta,train_x), ...
      ssnn1Theta, options);
  
  
  
  

  
%----------------------Train The Second Ply---------------

feather2 = sampledPredict(OptTheta1,train_x,inputSize,hiddenSize1,sampledSize);

ssnn2Theta = initializeParameters(hiddenSize2,hiddenSize1);
  
  
options.Method = 'lbfgs'; 
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[OptTheta2, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSize1, hiddenSize2, ...
                                   lambda, sparsityParam, ...
                                   beta, feather2), ...
                              ssnn2Theta, options);
                          
          
                          
%----------------------Train The Third Ply---------------------
 [feather3] = feedForwardAutoencoder(OptTheta2, hiddenSize2, ...
                                        hiddenSize1, feather2);    
                                    
                                    
ssnn3Theta = initializeParameters(hiddenSize2,hiddenSize3);
  
  
options.Method = 'lbfgs'; 
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[OptTheta3, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSize2, hiddenSize3, ...
                                   lambda, sparsityParam, ...
                                   beta, feather3), ...
                              ssnn3Theta, options);                                  
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                           
%----------------------Train The Fourth Ply---------------------
 [feather4] = feedForwardAutoencoder(OptTheta3, hiddenSize3, ...
                                        hiddenSize2, feather3);                                      
 weights = rand(1,316)';                                   
[err,OptTheta4] = batch_gradient_descent([feather4;ones(1,m)],train_y,weights,0.1,2000);        






%----------------------Finetune The Model---------------------
theta = [OptTheta1(1:inputSize);OptTheta2(1:hiddenSize1*hiddenSize2);OptTheta2(2*hiddenSize2*hiddenSize1+1:2*hiddenSize2*hiddenSize1+hiddenSize2);
    OptTheta3(1:hiddenSize2*hiddenSize3);OptTheta3(2*hiddenSize3*hiddenSize2+1:2*hiddenSize3*hiddenSize2+hiddenSize3);OptTheta4(:)];
options.Method = 'lbfgs'; 
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[OptTheta, cost] = minFunc( @(p) SSNNCost(p, ...
                                   train_x, inputSize, ...
                                   hiddenSize1, hiddenSize2,hiddenSize3, ...
                                   train_y,sampledSize), ...
                              theta, options);
                 





%----------------------Predict---------------------      
test_x = [wdms(1001:1400,:);non(7001:10000,:)];
test_y = [ones(400,1); zeros(3000,1)];
test_x = mapminmax(test_x,0,1)';

test_y = test_y';


result = Predict(OptTheta,test_x,inputSize,hiddenSize1,hiddenSize2,hiddenSize3,sampledSize);
for i = 1:size(result,2)
    if(result(i)<0.5)
        result(i) = 0;
    else
        result(i) = 1;
    end;
end;
acc = sum(result == test_y)./size(result,2);
fprintf(' Test Accuracy: %0.3f%%\n', acc * 100);
