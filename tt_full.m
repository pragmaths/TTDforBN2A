function T = tt_full(cores)
% TT_FULL  Reconstruct full tensor from TT cores (for validation only).
%
% cores - cell array of J cores, each r_{j-1} x 2 x r_j
%
% Returns 2x2x...x2 (J modes) tensor.

J = length(cores);

% Start by reshaping first core to a matrix (mode dim x trailing)
c1 = cores{1};
sz1 = size(c1);
if length(sz1) < 3, sz1(3) = 1; end
% c1 is 1 x 2 x r_1
T = reshape(c1, 2, sz1(3));

for j = 2:J
    cj = cores{j};
    sz = size(cj);
    if length(sz) < 3, sz(3) = 1; end
    % cj is r_{j-1} x 2 x r_j
    % Contract: T (prev x r_{j-1}) * cj (r_{j-1} x 2*r_j)
    cj_mat = reshape(cj, sz(1), 2 * sz(3));
    T = T * cj_mat;
    T = reshape(T, [], sz(3));    % collapse for next iteration
end

% Final reshape to 2x2x...x2
T = reshape(T, repmat(2, 1, J));
end
