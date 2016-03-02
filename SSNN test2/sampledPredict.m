function a2 = sampledPredict(theta,data,visibleSize, hiddenSize,sampledSize)
W1 = theta(1:visibleSize);
[~,m] = size(data);
a2 = zeros(587,m);
for i = 1:hiddenSize
    a2(i,:) = sigmoid(W1(sampledSize*i-5:sampledSize*i)'*data(sampledSize*i-5:sampledSize*i,:));
end;