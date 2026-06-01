function v = tt_eval(cores, pattern)
% TT_EVAL  Evaluate a TT tensor at index pattern.
%
% cores   - cell array of J cores, each r_{j-1} x 2 x r_j
% pattern - 1xJ vector of indices in {0, 1}
%
% Returns scalar value.

J = length(cores);
% Start with identity (1x1)
v = 1;
for j = 1:J
    c = cores{j};
    sz = size(c);
    if length(sz) < 3
        sz(3) = 1;
    end
    slice = squeeze(c(:, pattern(j) + 1, :));
    % Reshape to matrix r_{j-1} x r_j
    slice = reshape(slice, sz(1), sz(3));
    v = v * slice;
end
v = squeeze(v);
end
