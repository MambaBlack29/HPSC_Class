% Read flat vector from file
mat = load('x.txt');   % should have n*n entries

% Infer dimension n
n = sqrt(length(mat));
if n ~= floor(n)
    error('Length of x.txt is not a perfect square!');
end

% Reshape into n x n array (column-major by default in MATLAB)
A = reshape(mat, n, n)';

% Plot as heatmap
figure;
imagesc([0 1], [0 1], A);   % map array domain to [0,1]x[0,1]
colormap jet;               % nice color scheme
colorbar;                   % show color scale
axis equal tight;           % equal aspect ratio, remove gaps
axis xy;

xlabel('x');
ylabel('y');
title('Heatmap of solution x to the Laplacian Equation');