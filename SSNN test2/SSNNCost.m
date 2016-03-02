function [Jcost grad] = SSNNCost(theta,data,inputSize,hiddenSize1,hiddenSize2,hiddenSize3,lable,sampledSize)
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


Jcost = (0.5/m)*sum((result-lable).^2);

d5 = -(lable-result).*(result.*(1-result));%sigmoidInv(z3);
d4 = (W4*d5).*(a4.*(1-a4));
d3 = (W3'*d4).*(a3.*(1-a3));
d2 = (W2'*d3).*(a2.*(1-a2));


W4grad = (1/m).*(a4*d5');
b4grad = (1/m)*sum(d5,2);
W3grad = (1/m).*(d4*a3');
b3grad = (1/m)*sum(d4,2);
W2grad = (1/m).*(d3*a2');
b2grad = (1/m)*sum(d3,2);
c = kron(d2,[1;1;1;1;1;1]);                                 %把残差行为单位平铺6行，对应乘到输入数据得到梯度
W1grad = (1/m)*sum(c.*data,2);%+lambda*W1


grad = [W1grad(:) ; W2grad(:) ; b2grad(:) ; W3grad(:) ; b3grad(:) ;W4grad(:) ; b4grad(:)];
