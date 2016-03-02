function [cost,grad,features] = sparseAutoencoderLinearCost(theta, visibleSize, hiddenSize, ...
                                                            lambda, sparsityParam, beta, data)
% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Copy sparseAutoencoderCost in sparseAutoencoderCost.m from your
%   earlier exercise onto this file, renaming the function to
%   sparseAutoencoderLinearCost, and changing the autoencoder to use a
%   linear decoder.
% -------------------- YOUR CODE HERE --------------------                                    

% W1 is a hiddenSize * visibleSize matrix
W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
% W2 is a visibleSize * hiddenSize matrix
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
% b1 is a hiddenSize * 1 vector
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
% b2 is a visible * 1 vector
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

numCases = size(data, 2);

% forward propagation
z2 = W1 * data + repmat(b1, 1, numCases);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2, 1, numCases);
a3 = z3;

% error
sqrerror = (data - a3) .* (data - a3);
error = sum(sum(sqrerror)) / (2 * numCases);
% weight decay
wtdecay = (sum(sum(W1 .* W1)) + sum(sum(W2 .* W2))) / 2;
% sparsity
rho = sum(a2, 2) ./ numCases;
divergence = sparsityParam .* log(sparsityParam ./ rho) + (1 - sparsityParam) .* log((1 - sparsityParam) ./ (1 - rho));
sparsity = sum(divergence);

cost = error + lambda * wtdecay + beta * sparsity;

% delta3 is a visibleSize * numCases matrix
delta3 = -(data - a3);
% delta2 is a hiddenSize * numCases matrix
sparsityterm = beta * (-sparsityParam ./ rho + (1-sparsityParam) ./ (1-rho));
delta2 = (W2' * delta3 + repmat(sparsityterm, 1, numCases)) .* sigmoiddiff(z2);

W1grad = delta2 * data' ./ numCases + lambda * W1;
b1grad = sum(delta2, 2) ./ numCases;

W2grad = delta3 * a2' ./ numCases + lambda * W2;
b2grad = sum(delta3, 2) ./ numCases;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function sigmdiff = sigmoiddiff(x)

    sigmdiff = sigmoid(x) .* (1 - sigmoid(x));
end