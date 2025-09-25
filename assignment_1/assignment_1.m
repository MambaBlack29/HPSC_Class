clear; clc;

% load data from Kmat into A
mat = load("Kmat.txt");
vec = load("Fvec.txt");
n = sqrt(size(mat, 1));
A = reshape(mat, n, n);

% split A
D = diag(diag(A));
L = tril(A, -1);
U = triu(A, 1);

% Jacobi iteration matrix -D^-1 * (D - A)
Gj = D \ (D - A);
cj = D \ vec;

% Gauss Seidel iteration stuff (D + L)^-1 * U
Gg = -(D + L) \ U;
cg = (D + L) \ vec;

% SOR iteration thing
w = 1.9;
Gs = (D + w*L) \ ((1 - w)*D - w*U);
cs = w * ((D + w*L) \ vec);

% ANALYSIS
% eigen vector of a G
Eig = eig(Gs);

% spectral radius and convergence rate
specRad = max(abs(Eig));
convRate = -log(specRad);

% % iterative steps
% function [x, k] = run_fp(G, c, x0)
% %RUN_FP  Minimal fixed-point iterator for x_{k+1} = G*x_k + c.
% % Usage:
% %   [x, k] = run_fp(G, c)         % starts from zeros(size(c))
% %   [x, k] = run_fp(G, c, x0)     % custom initial guess
% %
% % Stopping rule (fixed):  ||x^{k+1}-x^k||_inf <= 1e-8 * (1 + ||x^{k+1}||_inf)
% % Hard cap on iterations (fixed): 1e6

%     if nargin < 3, x0 = randn(size(c)); end
%     tol   = 1e-8;
%     maxit = 1e8;

%     x = x0;
%     for k = 1:maxit
%         x_new = G*x + c;
%         if norm(x_new - x, inf) <= tol * (1 + norm(x_new, inf))
%             x = x_new;
%             return
%         end
%         x = x_new;
%     end
%     % If we get here, maxit was hit; x holds the last iterate and k = maxit.
% end

% sm = 0;
% num_iter = 3000;
% for i = 1:num_iter
%     [x, k] = run_fp(Gs, cs);
%     sm = sm + k;
% end

% sm = sm / num_iter;