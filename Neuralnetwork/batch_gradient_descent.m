function [ loss, parameters ] = batch_gradient_descent(x,y,init_parameter, rate, maxiter)
% sub function to compute gradient

w = init_parameter;
m = length(y);
total_loss = [];
for i=1:maxiter
    h = sigmoid(w' * x);
    J = -sum(y.*log(h)+(1-y).*log(1-h))/m ;
    total_loss = [total_loss J];
    % gradient is now in a different expression
    gradient = x*(h-y)'./m;%+ lambda.*w; % sum all in each iteration, it's a batch gradient
    w = w - rate.*gradient;
end

loss = total_loss;
parameters = w;

end

