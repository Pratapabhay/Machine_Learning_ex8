function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)


X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

s=0;
[a b] =  size(R);

s =  sum(sum(((X*Theta').* R - (Y .* R) ).^ 2)); 

a = sum(sum(Theta .^ 2));
b = sum(sum(X .^ 2));



J =  0.5 * ( s + lambda * (a+b));

[a b] = size(X);

for i = 1:a
	idx = find(R(i,:)==1);

	Thetatemp = Theta(idx,:);

	Ytemp = Y(i, idx);
	
	X_grad(i, :) = ((X(i, :) * Thetatemp') - Ytemp ) * Thetatemp;
end

X_grad = X_grad + lambda * X;

[a b] = size(Theta);


for i = 1:a
	idx = find(R(:,i)==1);
	Xtemp = X(idx,:);
	Ytemp = Y(idx, i);
	Theta_grad(i, :) = ((Xtemp * Theta(i,:)') - Ytemp )' * Xtemp;
end

Theta_grad = Theta_grad + lambda * Theta;
grad = [X_grad(:); Theta_grad(:)];

end
