%% VERIFICATION: Reproduce Gu's toy example (J=5, K=2)
%%
%% Gu's paper describes a toy example with a specific Q-matrix and predicts
%% which unfoldings should have rank <= 2 vs > 2. This script:
%%
%%   1. Uses Gu's exact Q-matrix
%%   2. Builds the population tensor T (no sampling noise)
%%   3. Computes ALL pair unfoldings and reports their numerical ranks
%%   4. Verifies that the predicted pattern holds:
%%      - unfold({1,2},...): rank <= 2  (items 1,2 pure for A1)
%%      - unfold({3,4},...): rank <= 2  (items 3,4 pure for A2)
%%      - unfold({1,3},...): rank > 2   (items 1,3 different attributes)
%%      - any unfolding with item 5:    rank > 2 (item 5 is multi-attribute)
%%   5. Runs our Gu v2 pipeline on TT cores and checks recovery

clear; clc; rng(2026);

J = 5;
K = 2;

% --- Gu's exact Q-matrix ---
Q_true = [1 0;
          1 0;
          0 1;
          0 1;
          1 1];

fprintf('============================================================\n');
fprintf('GU TOY EXAMPLE VERIFICATION (J=5, K=2)\n');
fprintf('============================================================\n');
fprintf('\nGu''s Q-matrix:\n');
disp(Q_true);

% --- Parameters (Gu does not specify; we use reasonable defaults) ---
params.leak    = [0.9; 0.9; 0.9; 0.9; 0.9];   % all leaks = 0.9
params.penalty = ones(J, K);
% Item 1 (pure A1): only penalty for A1
params.penalty(1, 1) = 0.2;
% Item 2 (pure A1): only penalty for A1
params.penalty(2, 1) = 0.3;
% Item 3 (pure A2): only penalty for A2
params.penalty(3, 2) = 0.2;
% Item 4 (pure A2): only penalty for A2
params.penalty(4, 2) = 0.3;
% Item 5 (multi A1+A2): penalty for both
params.penalty(5, 1) = 0.2;
params.penalty(5, 2) = 0.3;

fprintf('\nLeak parameters:\n');
disp(params.leak');
fprintf('Penalty parameters (only non-trivial entries):\n');
disp(params.penalty);

% --- Build population tensor T directly via marginalization ---
fprintf('\nBuilding population tensor T (size 2^%d = %d entries)...\n', J, 2^J);

T_pop = zeros(repmat(2, 1, J));
% Uniform prior over attribute patterns
prior = 1 / (2^K);

for r1 = 0:1
    for r2 = 0:1
        for r3 = 0:1
            for r4 = 0:1
                for r5 = 0:1
                    response = [r1 r2 r3 r4 r5];
                    p_total = 0;
                    for a1 = 0:1
                        for a2 = 0:1
                            attrs = [a1 a2];
                            % BN2A response probability per item
                            p_response = 1;
                            for j = 1:J
                                p_correct = params.leak(j);
                                for k = 1:K
                                    if Q_true(j, k) == 1
                                        p_correct = p_correct * params.penalty(j, k)^(1 - attrs(k));
                                    end
                                end
                                if response(j) == 1
                                    p_response = p_response * p_correct;
                                else
                                    p_response = p_response * (1 - p_correct);
                                end
                            end
                            p_total = p_total + prior * p_response;
                        end
                    end
                    T_pop(r1+1, r2+1, r3+1, r4+1, r5+1) = p_total;
                end
            end
        end
    end
end

fprintf('Sum of T entries: %.6f (should be 1.0)\n', sum(T_pop(:)));

% --- Compute all pair unfoldings and their numerical ranks ---
fprintf('\n============================================================\n');
fprintf('PAIR UNFOLDINGS AND THEIR NUMERICAL RANKS\n');
fprintf('============================================================\n');
fprintf('Predicted by Gu:\n');
fprintf('  - rank <= 2  when both items pure for same attribute\n');
fprintf('  - rank  > 2  otherwise\n\n');

all_items = 1:J;
pair_list = nchoosek(all_items, 2);

fprintf('  pair  | sigma_1   sigma_2   sigma_3   sigma_4 | numerical rank | Gu prediction\n');
fprintf('  ------+--------------------------------------------+----------------+--------------\n');

for p = 1:size(pair_list, 1)
    S = pair_list(p, :);
    Sbar = setdiff(all_items, S);
    
    % Build the unfolding manually
    perm_order = [S, Sbar];
    T_perm = permute(T_pop, perm_order);
    rows = 2^length(S);
    cols = 2^length(Sbar);
    U = reshape(T_perm, [rows, cols]);
    
    sigma = svd(U);
    while length(sigma) < 4
        sigma(end+1) = 0;
    end
    
    % Determine numerical rank (tolerance: 1e-8 relative to largest)
    num_rank = sum(sigma > 1e-8 * sigma(1));
    
    % Gu's prediction: rank <= 2 if both items are pure for same attribute
    q1 = Q_true(S(1), :);
    q2 = Q_true(S(2), :);
    same_pure = (sum(q1) == 1) && (sum(q2) == 1) && isequal(q1, q2);
    pred = '> 2';
    if same_pure
        pred = '<= 2';
    end
    
    % Check if observation matches prediction
    obs_low = num_rank <= 2;
    pred_low = same_pure;
    if obs_low == pred_low
        verdict = 'MATCH';
    else
        verdict = 'MISMATCH';
    end
    
    fprintf('  (%d,%d) | %.5f  %.5f  %.5f  %.5f  |       %d        | %4s  [%s]\n', ...
        S(1), S(2), sigma(1), sigma(2), sigma(3), sigma(4), num_rank, pred, verdict);
end

% --- Now run our pipeline ---
fprintf('\n============================================================\n');
fprintf('RUNNING OUR PIPELINE (TT-SVD + Gu v2) ON POPULATION TENSOR\n');
fprintf('============================================================\n');

max_rank = 2^K;
[cores, ranks, ~] = tt_svd_full(T_pop, max_rank, 1e-12);
fprintf('TT-SVD ranks: [%s] (max bound = 2^K = %d)\n', num2str(ranks'), max_rank);

opts_gu.threshold_prop1 = 0.15;
opts_gu.threshold_prop2 = 0.03;
opts_gu.verbose = true;

[Q_est, info_gu] = qmatrix_identification_full_gu_v2(cores, K, opts_gu);

fprintf('\n--- RESULT ---\n');
if isfield(info_gu, 'step1_failed') && info_gu.step1_failed
    fprintf('STEP 1 FAILED.\n');
else
    % Try both column permutations (only 2 for K=2)
    perms_to_try = perms(1:K);
    best_correct = 0;
    best_Q = Q_est;
    for p = 1:size(perms_to_try, 1)
        Q_perm = Q_est(:, perms_to_try(p, :));
        c = sum(Q_perm(:) == Q_true(:));
        if c > best_correct
            best_correct = c;
            best_Q = Q_perm;
        end
    end
    
    fprintf('Recovered Q (best column permutation):\n');
    disp(best_Q);
    fprintf('True Q:\n');
    disp(Q_true);
    
    entry = sum(best_Q(:) == Q_true(:));
    row = sum(all(best_Q == Q_true, 2));
    fprintf('Entries correct: %d/%d\n', entry, J*K);
    fprintf('Rows correct:    %d/%d\n', row, J);
    
    if entry == J*K
        fprintf('\n>>> PIPELINE RECOVERS GU''S TOY EXAMPLE EXACTLY.\n');
    else
        fprintf('\n>>> PIPELINE DOES NOT FULLY RECOVER. Investigate.\n');
    end
end

fprintf('\n============================================================\n');
fprintf('VERIFICATION SUMMARY\n');
fprintf('============================================================\n');
fprintf('1. If all "MATCH" labels appear above: rank predictions confirmed.\n');
fprintf('2. If pipeline returns Q_true: full implementation confirmed.\n');
fprintf('Both together verify implementation fidelity to Gu.\n');