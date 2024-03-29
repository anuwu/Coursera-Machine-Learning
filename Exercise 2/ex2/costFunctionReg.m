function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J1 = log(sigmoid(X*theta)) ;
J1 = transpose(y) * J1 ;

J2 = log(ones(m , 1) - sigmoid(X*theta)) ;
J2 = transpose (ones(m,1) - y) * J2 ;

J = (-1/m) * (J1 + J2) ;
J = J + (lambda/(2*m))*(sum (theta .* theta)) ;
J = J(1,1) - (lambda/(2*m)) * theta(1)^2 ;

grad = (1/m)*(transpose(X) * (sigmoid(X*theta) - y)) + (lambda*theta/m) ;
grad(1) = grad(1) - lambda*theta(1)/m ;

% =============================================================

end
