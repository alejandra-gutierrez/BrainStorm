function yhat = nonLinModelFun(beta, x)
    N = 4;
    L = size(x, 1);
    b = zeros(N, L);
    b(1, :) = beta(1: L);
    b(2, :) = beta(L+1 : 2*L);
    b(3, :) = beta(2*L+1 : 3*L);
    b(4, :) = beta(3*L+1 : 4*L); % additional spaces if need higher ranked 

    yhat = b(1,:)*x+b(2, :)*x.^3;
end