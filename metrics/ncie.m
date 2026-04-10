function res = NCIE(im1, im2, fim)

% =========================
% NORMALIZE (inline normalize1)
% =========================
im1 = normalize_inline(im1);
im2 = normalize_inline(im2);
fim = normalize_inline(fim);

b = 256;
K = 3;

% =========================
% NCC computations
% =========================
NCCxy = NCC_inline(im1, im2);
NCCxf = NCC_inline(im1, fim);
NCCyf = NCC_inline(im2, fim);

% =========================
% Correlation matrix
% =========================
R = [1      NCCxy  NCCxf;
     NCCxy  1      NCCyf;
     NCCxf  NCCyf  1];

% Eigenvalues
r = eig(R);

% =========================
% NCIE
% =========================
HR = sum(r .* log2(r ./ K) / K);
HR = -HR / log2(b);

NCIE = 1 - HR;

res = NCIE;

end

% ============================================================
% INLINE NORMALIZE (from normalize1.m)
% ============================================================
function RES = normalize_inline(data)

data = double(data);

da = max(data(:));
xiao = min(data(:));

if (da == 0 && xiao == 0)
    RES = data;
else
    newdata = (data - xiao) / (da - xiao);
    RES = round(newdata * 255);
end

end

% ============================================================
% INLINE NCC (from NCC.m)
% ============================================================
function res = NCC_inline(im1, im2)

im1 = double(im1);
im2 = double(im2);

[hang, lie] = size(im1);
N = 256;
b = 256;

% =========================
% Joint histogram
% =========================
h = zeros(N, N);

for i = 1:hang
    for j = 1:lie
        x = im1(i,j) + 1;
        y = im2(i,j) + 1;
        h(x, y) = h(x, y) + 1;
    end
end

% =========================
% Normalize histogram -> probability
% =========================
h = h ./ sum(h(:));

% =========================
% Marginal distributions
% =========================
im1_marg = sum(h);
im2_marg = sum(h');

% =========================
% Entropy
% =========================
H_x = -sum(im1_marg .* log2(im1_marg + (im1_marg == 0)));
H_y = -sum(im2_marg .* log2(im2_marg + (im2_marg == 0)));

H_xy = -sum(sum(h .* log2(h + (h == 0))));

% Normalize
H_xy = H_xy / log2(b);
H_x = H_x / log2(b);
H_y = H_y / log2(b);

% =========================
% NCC (Nonlinear Correlation)
% =========================
res = H_x + H_y - H_xy;

end