inputSize = 5;
hiddenSize1 = 4;
hiddenSize2 = 3;
maxIterations = 400;
sparsityParam = 0.1;   % desired average activation of the hidden units.
                       % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		               %  in the lecture notes). 
lambda = 3e-3;         % weight decay parameter       
beta = 3;              % weight of sparsity penalty term    
addpath minFunc/
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
tic;
%----------------------Train The First Ply---------------
Theta12 = initializeParameters(inputSize,hiddenSize1);

options.Method = 'lbfgs'; 
options.maxIter = maxIterations;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[OptTheta12, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSize1, ...
                                   lambda, sparsityParam, ...
                                   beta, train_x), ...
                                   Theta12, options);
                               
                               
                               
                               
%----------------------Train The Second Ply---------------\
[feather2] = feedForwardAutoencoder(OptTheta12, hiddenSize1, ...
                                        inputSize, train_x);    
                                    
                                    
Theta23 = initializeParameters(hiddenSize2,hiddenSize1);
  
  
options.Method = 'lbfgs'; 
options.maxIter = maxIterations;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[OptTheta23, ~] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSize1, hiddenSize2, ...
                                   lambda, sparsityParam, ...
                                   beta, feather2), ...
                                   Theta23, options);   
                               
                               
%----------------------Train The Third Ply---------------------

 [feather3] = feedForwardAutoencoder(OptTheta23, hiddenSize2, ...
                                        hiddenSize1, feather2);    
 weights = rand(1,hiddenSize2 + 1)';                                   
[err,OptTheta34] = batch_gradient_descent([feather3;ones(1,m)],train_y,weights,0.1,4000);      


%----------------------Finetune The Model---------------------
theta = [OptTheta12(1:inputSize*hiddenSize1);OptTheta12(2*inputSize*hiddenSize1+1:2*inputSize*hiddenSize1+hiddenSize1);
        OptTheta23(1:hiddenSize1*hiddenSize2);OptTheta23(2*hiddenSize1*hiddenSize2+1:2*hiddenSize1*hiddenSize2+hiddenSize2);
        OptTheta34(:)];
    
options.Method = 'lbfgs'; 
options.maxIter = maxIterations;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[OptTheta, cost] = minFunc( @(p) neturalNetworkCost(p, ...
                                   train_x, inputSize, ...
                                   hiddenSize1, hiddenSize2, ...
                                   train_y), ...
                                   theta, options);
                               
                               
  time = toc;                             
                               
                               
%----------------------Predict The Model---------------------                               
test_x = [wdms(1001:1400,:);non(7001:10000,:)];
test_y = [ones(400,1); zeros(3000,1)];
test_x = mapminmax(test_x,0,1)';

test_y = test_y';


                       
result_test  = neturalPredict(OptTheta,test_x,inputSize,hiddenSize1,hiddenSize2);        
for i = 1:size(result_test,2)
    if(result_test(i)<0.5)
        result_test(i) = 0;
    else
        result_test(i) = 1;
    end;
end;
acc = sum(result_test == test_y)./size(result_test,2);
fprintf(' Test Accuracy: %0.3f%%\n', acc * 100);


 result_train  = neturalPredict(OptTheta,train_x,inputSize,hiddenSize1,hiddenSize2);        
for i = 1:size(result_train,2)
    if(result_train(i)<0.5)
        result_train(i) = 0;
    else
        result_train(i) = 1;
    end;
end;
acc = sum(result_train == train_y)./size(result_train,2);
fprintf(' Train Accuracy: %0.3f%%\n', acc * 100);
fprintf(' Train Time: %0.1f\n', time);

