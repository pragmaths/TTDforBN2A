function responses = generate_bn2a_responses(Q, params, alpha_matrix)
% GENERATE_BN2A_RESPONSES - Generate responses from BN2A model (RRUM)
%
% Inputs:
%   Q           - Q-matrix (J x K): binary incidence matrix
%   params      - struct with fields:
%                 .leak (J x 1): q_{j,0} leak probabilities
%                 .penalty (J x K): q_{j,k} penalty parameters
%   alpha_matrix - (n x K): attribute patterns for n students
%
% Output:
%   responses   - (n x J): binary response matrix
%
% Model (from IJAR manuscript, Eq. 12):
%   P(Y_j = 1 | X) = q_{j,0} * prod_{k in pa(Y_j)} q_{j,k}^{(1-x_k)}
%
% Interpretation:
%   - q_{j,0}: leak probability (P(correct) when all skills present)
%   - q_{j,k}: penalty for lacking skill k on item j (0 < q_{j,k} <= 1)
%   - If x_k = 1 (skill present): factor = 1 (no penalty)
%   - If x_k = 0 (skill absent): factor = q_{j,k} < 1 (penalty applied)

[n, K] = size(alpha_matrix);
[J, ~] = size(Q);

% Validate parameters
if length(params.leak) ~= J
    error('params.leak must have J elements');
end
if ~all(size(params.penalty) == [J, K])
    error('params.penalty must be J x K matrix');
end

% Validate parameter ranges
if any(params.leak <= 0) || any(params.leak > 1)
    error('leak parameters must be in (0, 1]');
end
if any(params.penalty(:) <= 0) || any(params.penalty(:) > 1)
    error('penalty parameters must be in (0, 1]');
end

% Generate responses
responses = zeros(n, J);

for i = 1:n
    alpha_i = alpha_matrix(i, :);  % Student i's attribute pattern
    
    for j = 1:J
        % Start with leak probability
        prob_correct = params.leak(j);
        
        % Apply penalty for each missing skill
        for k = 1:K
            if Q(j, k) == 1  % If item j requires skill k
                if alpha_i(k) == 0  % If student lacks skill k
                    prob_correct = prob_correct * params.penalty(j, k);
                end
                % If alpha_i(k) == 1, no penalty (factor = 1)
            end
        end
        
        % Generate binary response
        responses(i, j) = rand() < prob_correct;
    end
end

end
