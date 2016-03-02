function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize,sampledSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units 
% hiddenSize: the number of hidden units 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = theta(1:visibleSize);
W2 = reshape(theta(visibleSize+1:visibleSize+visibleSize*hiddenSize),visibleSize,hiddenSize);
b2 = theta(visibleSize+visibleSize*hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 

W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b2grad = zeros(size(b2));


% Jcost = 0;
% Jweight = 0;
% Jsparse = 0
[~,m] = size(data);%m为样本的个数，~为样本的特征数
  
%前向算法计算各神经网络节点的线性组合值和active值

a2 = zeros(587,m);
for i = 1:hiddenSize
    a2(i,:) = sigmoid(W1(sampledSize*i-5:sampledSize*i)'*data(sampledSize*i-5:sampledSize*i,:));
end;

% z3 = W2*a2+repmat(b2,1,m);
% a3 = sigmoid(z3);
a3 = sigmoid(W2*a2+repmat(b2,1,m));

% 计算预测产生的误差
Jcost = (0.5/m)*sum(sum((a3-data).^2));%直接误差
  
%计算权值惩罚项
Jweight = (1/2)*(sum(sum(W1.^2))+sum(sum(W2.^2)));%权值惩罚
  
%计算稀释性正则项
% rho = (1/m).*sum(a2,2);%求出第一个隐含层的平均值向量
% Jsparse = sum(sparsityParam.*log(sparsityParam./rho)+ ...
%         (1-sparsityParam).*log((1-sparsityParam)./(1-rho)));%稀疏性惩罚
  
%损失函数的总表达式
cost = Jcost+lambda*Jweight;%+beta*Jsparse;
  
%反向传导算法求出每个节点的误差值
d3 = -(data-a3).*(a3.*(1-a3));%sigmoidInv(z3);
%sterm = beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho));%因为加入了稀疏正则项，所以
                                                             %计算偏导时需要引入该项
% d2 = (W2'*d3+repmat(sterm,1,m)).*sigmoidInv(z2);
d2 = (W2'*d3).*(a2.*(1-a2));%+repmat(sterm,1,m)
  
%计算W1grad 
c = kron(d2,[1;1;1;1;1;1]);                                 %把残差行为单位平铺6行，对应乘到输入数据得到梯度
W1grad = W1grad+(1/m)*sum(c.*data,2);%+lambda*W1

  
%计算W2grad 
W2grad = W2grad+d3*a2';
W2grad = (1/m).*W2grad;%+lambda*W2;
  
%计算b2grad
b2grad = b2grad+sum(d3,2);
b2grad = (1/m)*b2grad;






%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

