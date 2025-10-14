mat = load('A.txt');   % should have n*n entries

A = mat;
D = diag(diag(A));
L = tril(A, -1);
U = triu(A, 1);

w = 1.4;
Gs = (D + w*L) \ ((1 - w)*D - w*U);
Gj = D \ (D - A);

% ANALYSIS
% eigen vector of a G
Eig = eig(Gj);

% spectral radius and convergence rate
specRad = max(abs(Eig));
convRate = -log(specRad);
omega = 2 / (1 + sqrt(1 - specRad*specRad));