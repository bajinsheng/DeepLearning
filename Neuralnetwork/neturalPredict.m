function result = neturalPredict(theta,data,inputSize,hiddenSize1,hiddenSize2)
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
result = a4;