function [Jcost,grad] = neturalNetworkCost(theta,data,inputSize,hiddenSize1,hiddenSize2,lable)
W1 = reshape(theta(1:hiddenSize1*inputSize), hiddenSize1, inputSize);
b1 = theta(hiddenSize1*inputSize+1:hiddenSize1*inputSize+hiddenSize1);
W2 = reshape(theta(hiddenSize1*inputSize+hiddenSize1+1:hiddenSize1*inputSize+hiddenSize1+hiddenSize1*hiddenSize2), hiddenSize2, hiddenSize1);
b2 = theta(hiddenSize1*inputSize+hiddenSize1+hiddenSize1*hiddenSize2+1:hiddenSize1*inputSize+hiddenSize1+hiddenSize1*hiddenSize2+hiddenSize2);
W3 = theta(hiddenSize1*inputSize+hiddenSize1+hiddenSize1*hiddenSize2+hiddenSize2+1:end-1);
b3 = theta(end);



[~,m] = size(data);



%result
a2 = sigmoid(W1*data+repmat(b1,1,m));
a3 = sigmoid(W2*a2+repmat(b2,1,m));
a4 = sigmoid(W3'*a3+repmat(b3,1,m));


Jcost = (0.5/m)*sum((a4-lable).^2);

d4 = -(lable-a4).*(a4.*(1-a4));%sigmoidInv(z3);
d3 = (W3*d4).*(a3.*(1-a3));
d2 = (W2'*d3).*(a2.*(1-a2));


W3grad = (1/m).*(d4*a3');
b3grad = (1/m)*sum(d4,2);
W2grad = (1/m).*(d3*a2');
b2grad = (1/m)*sum(d3,2);
W1grad = (1/m).*(d2*data');
b1grad = (1/m)*sum(d2,2);


grad = [W1grad(:) ; b1grad(:) ; W2grad(:) ; b2grad(:) ; W3grad(:) ; b3grad(:)];

end
