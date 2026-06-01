function [cores, ranks, info] = tt_svd_full(T, max_rank, tol)
% TT_SVD_FULL  Standard TT-SVD on a full tensor (Oseledets 2011).
%
% Inputs:
%   T        - full tensor (any shape, here 2x2x...x2 with J modes)
%   max_rank - cap on TT-ranks
%   tol      - singular value truncation threshold
%
% Outputs:
%   cores - cell array of J cores
%   ranks - (J+1) x 1 vector of TT-ranks
%   info  - storage in parameters, sweep info

if nargin < 3, tol = 1e-12; end
if nargin < 2, max_rank = Inf; end

sz = size(T);
J = length(sz);

cores = cell(J, 1);
ranks = zeros(J+1, 1);
ranks(1) = 1;
ranks(J+1) = 1;

% Prepend r_0 = 1 to T's shape conceptually
T_current = reshape(T, 1, []);  % 1 x (prod sz)
T_current = reshape(T_current, [1, sz]);

for j = 1:J-1
    s = size(T_current);
    r_prev = s(1);
    n_j = s(2);
    rest = prod(s(3:end));
    
    M = reshape(T_current, r_prev * n_j, rest);
    [U, S, V] = svd(M, 'econ');
    sv = diag(S);
    r_new = sum(sv > tol);
    r_new = min(r_new, max_rank);
    
    cores{j} = reshape(U(:, 1:r_new), [r_prev, n_j, r_new]);
    ranks(j+1) = r_new;
    
    T_current = S(1:r_new, 1:r_new) * V(:, 1:r_new)';
    T_current = reshape(T_current, [r_new, s(3:end)]);
end

% Last core
s = size(T_current);
cores{J} = reshape(T_current, [s(1), s(2), 1]);

% Storage
storage = 0;
for j = 1:J
    storage = storage + numel(cores{j});
end

info.storage = storage;
info.tol = tol;
info.max_rank = max_rank;
end
