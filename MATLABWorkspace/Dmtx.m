function D = Dmtx(N)
% generates D matrix according to input dimension N
[D_temp_X, D_temp_Y] = meshgrid(1:2:(2*N-1),0:(N-1));
D_temp = D_temp_X.*D_temp_Y*pi()/2/N;
D = sqrt(2/N)*cos(D_temp);
D(1,:) = sqrt(1/N);
end