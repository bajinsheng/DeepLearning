function result = Predict(theta,data,inputSize,hiddenSize1,hiddenSize2,hiddenSize3,sampledSize)
W1 = theta(1:inputSize);
W2 = reshape(theta(inputSize+1:inputSize+hiddenSize1*hiddenSize2),hiddenSize2,hiddenSize1);
b2 = theta(inputSize+hiddenSize1*hiddenSize2+1:inputSize+hiddenSize1*hiddenSize2+hiddenSize2);
W3 = reshape(theta(inputSize+hiddenSize1*hiddenSize2+hiddenSize2+1:inputSize+hiddenSize1*hiddenSize2+hiddenSize2+hiddenSize2*hiddenSize3),hiddenSize3,hiddenSize2);
b3 = theta(inputSize+hiddenSize1*hiddenSize2+hiddenSize2+hiddenSize2*hiddenSize3+1:inputSize+hiddenSize1*hiddenSize2+hiddenSize2+hiddenSize2*hiddenSize3+hiddenSize3);
W4 = theta(inputSize+hiddenSize1*hiddenSize2+hiddenSize2+hiddenSize2*hiddenSize3+hiddenSize3+1:end-1);
b4 = theta(end);
[~,m] = size(data);
a2 = zeros(587,m);

%result
for i = 1:hiddenSize1
    a2(i,:) = sigmoid(W1(sampledSize*i-5:sampledSize*i)'*data(sampledSize*i-5:sampledSize*i,:));
end;
a3 = sigmoid(W2*a2+repmat(b2,[1,size(a2,2)]));
a4 = sigmoid(W3*a3+repmat(b3,[1,size(a3,2)]));
result = sigmoid(W4'*a4 + repmat(b4,[1,size(a4,2)]));

                                         