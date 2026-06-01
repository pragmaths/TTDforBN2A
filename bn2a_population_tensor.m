function [T, oracle, info] = bn2a_population_tensor(Q, params, p_alpha)
% BN2A_POPULATION_TENSOR  Exact population tensor and oracle for BN2A/RRUM model
%
% Computes the EXACT (not sampled) joint distribution tensor
%   T[r_1,...,r_J] = P(R_1=r_1, ..., R_J=r_J)
% by marginalizing over all 2^K latent attribute patterns:
%   T[r] = sum_{alpha} p_alpha(alpha) * prod_j P(R_j=r_j | alpha)
%
% Inputs:
%   Q        - (J x K) Q-matrix (binary)
%   params   - struct with fields:
%              .leak    (J x 1) leak probabilities q_{j,0}
%              .penalty (J x K) penalty parameters q_{j,k}
%   p_alpha  - (2^K x 1) prior over attribute patterns
%              (default: uniform 1/2^K)
%
% Outputs:
%   T       - (2 x 2 x ... x 2, J modes) full population tensor
%             Indexing convention: T(i_1, ..., i_J) where i_j in {1,2}
%             corresponds to response r_j = i_j - 1
%   oracle  - function handle: p = oracle(r_pattern)
%             r_pattern is a 1xJ binary vector (response values 0 or 1)
%             Returns the population probability for that pattern WITHOUT
%             touching T.  This is what TT-cross will query.
%             Calls are counted internally; reset with oracle('reset')
%             and queried with oracle('count')
%   info    - struct with diagnostics

[J, K] = size(Q);

if nargin < 3 || isempty(p_alpha)
    p_alpha = ones(2^K, 1) / (2^K);
end

% Enumerate all attribute patterns
% alpha_patterns(a, k) = k-th attribute of a-th pattern (0/1)
alpha_patterns = zeros(2^K, K);
for a = 0:(2^K - 1)
    bits = bitget(a, K:-1:1);  % MSB-first
    alpha_patterns(a+1, :) = bits;
end

% --- Precompute P(R_j = 1 | alpha) for every (j, alpha) ---
% p_correct(j, a) = leak_j * prod_k penalty_{j,k}^{q_{j,k}(1 - alpha_k)}
p_correct = zeros(J, 2^K);
for j = 1:J
    for a = 1:(2^K)
        alpha = alpha_patterns(a, :);
        p = params.leak(j);
        for k = 1:K
            if Q(j, k) == 1 && alpha(k) == 0
                p = p * params.penalty(j, k);
            end
        end
        p_correct(j, a) = p;
    end
end

% --- Build full population tensor T ---
tensor_dims = repmat(2, 1, J);
T = zeros(tensor_dims);

% Iterate over all 2^J response patterns
for r_idx = 0:(2^J - 1)
    r_pattern = bitget(r_idx, J:-1:1);  % 1xJ binary
    
    % Compute T[r] = sum_alpha p_alpha(alpha) * prod_j P(R_j=r_j|alpha)
    prob = 0;
    for a = 1:(2^K)
        % For each item j: P(R_j = r_j | alpha) =
        %   p_correct(j,a) if r_j=1, else 1 - p_correct(j,a)
        pc = p_correct(:, a);                 % J x 1
        per_item = (r_pattern(:) == 1) .* pc + ...
                   (r_pattern(:) == 0) .* (1 - pc);
        prob = prob + p_alpha(a) * prod(per_item);
    end
    
    % Store in tensor (MATLAB uses 1-based indexing, +1 to each)
    idx_cell = num2cell(r_pattern + 1);
    T(idx_cell{:}) = prob;
end

% Sanity check: tensor should sum to 1
total = sum(T(:));
assert(abs(total - 1) < 1e-10, ...
    'Population tensor does not sum to 1 (got %.6e)', total);

% --- Build the oracle: closure over p_correct, p_alpha, Q ---
% This is the KEY object for TT-cross: a function that returns
% T[r_pattern] WITHOUT having T in memory. In our didactic setting
% we DO have T, but TT-cross will only ever call this function.
% We count calls to compare against 2^J.
%
% Special calls:
%   oracle('reset')  -> reset counter to 0
%   oracle('count')  -> return current count
%   oracle(r_pattern) -> return P(R = r_pattern), increment counter

persistent_count = 0;

    function out = oracle_fn(arg)
        if ischar(arg)
            switch arg
                case 'reset'
                    persistent_count = 0;
                    out = 0;
                case 'count'
                    out = persistent_count;
                otherwise
                    error('Unknown oracle command: %s', arg);
            end
            return;
        end
        % arg is a 1xJ pattern
        r_pattern_local = arg(:)';
        persistent_count = persistent_count + 1;
        prob_local = 0;
        for aa = 1:(2^K)
            pc_local = p_correct(:, aa);
            per_item_local = (r_pattern_local(:) == 1) .* pc_local + ...
                             (r_pattern_local(:) == 0) .* (1 - pc_local);
            prob_local = prob_local + p_alpha(aa) * prod(per_item_local);
        end
        out = prob_local;
    end

oracle = @oracle_fn;

% Diagnostics
info.J = J;
info.K = K;
info.tensor_sum = total;
info.tensor_min = min(T(:));
info.tensor_max = max(T(:));
info.full_storage = 2^J;
info.entropy = -sum(T(:) .* log2(max(T(:), 1e-300)));

end
