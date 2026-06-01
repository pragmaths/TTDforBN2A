%% REGRESSION TEST: rerun the successful J=15, K=3 experiment
%%
%% Same script structure as extension_J18.m (which gave 9/18) but with
%% J=15. Goal: confirm whether the current script base still recovers
%% Q correctly at J=15 (as full_q_comparison.m did earlier).
%%
%% If this script gives 45/45, the issue with J=18 is the extension itself.
%% If it gives less, the script base has drifted from the original.

clear; clc; rng(2026);

J = 15;
K = 3;
n = 100000;
alpha_smooth = 0.5;
max_rank = 2^K;

% --- Q-matrix: exactly as in full_q_comparison.m ---
Q_true = zeros(J, K);
Q_true(1, 1) = 1;  Q_true(2, 1) = 1;
Q_true(3, 2) = 1;  Q_true(4, 2) = 1;
Q_true(5, 3) = 1;  Q_true(6, 3) = 1;
multi_patterns = [1 1 0; 1 0 1; 0 1 1; 1 1 1];
for j = 7:J
    Q_true(j, :) = multi_patterns(mod(j-7, 4)+1, :);
end

fprintf('Q-matrix (J=%d, K=%d):\n', J, K);
disp(Q_true);

% Count multi-pattern duplicates for reference
multi_rows = Q_true(sum(Q_true,2) > 1, :);
fprintf('\nMulti-pattern counts:\n');
unique_patterns = unique(multi_rows, 'rows');
for p = 1:size(unique_patterns, 1)
    pat = unique_patterns(p, :);
    count = sum(all(multi_rows == pat, 2));
    fprintf('  [%d %d %d] appears %d times\n', pat(1), pat(2), pat(3), count);
end

% --- Item parameters (same generation logic) ---
leak_set    = [0.70 0.75 0.80 0.85 0.90 0.95];
penalty_set = [0.10 0.20 0.30 0.40];
params.leak    = leak_set(randi(length(leak_set), J, 1))';
params.penalty = ones(J, K);
for j = 1:J
    for k = 1:K
        if Q_true(j, k) == 1
            params.penalty(j, k) = penalty_set(randi(length(penalty_set)));
        end
    end
end

% --- Canonical thresholds ---
opts_gu.threshold_prop1 = 0.15;
opts_gu.threshold_prop2 = 0.03;
opts_gu.verbose = false;

tensor_dims = repmat(2, 1, J);
tensor_size = 2^J;

alpha_patterns_all = zeros(2^K, K);
for a = 0:(2^K - 1)
    bits = bitget(a, K:-1:1);
    alpha_patterns_all(a+1, :) = bits;
end

fprintf('\n============================================================\n');
fprintf('REGRESSION TEST: J=%d, K=%d\n', J, K);
fprintf('Total tensor size: 2^%d = %d entries\n', J, tensor_size);
fprintf('Sample size: %d (n/2^J = %.1f%%)\n', n, 100*n/tensor_size);
fprintf('Expected result (from earlier experiment): 45/45 entries, 15/15 rows\n');
fprintf('============================================================\n\n');

% --- Sample ---
fprintf('Sampling responses... ');
t0 = tic;
alpha_indices = randi(2^K, n, 1);
alpha_matrix = alpha_patterns_all(alpha_indices, :);
responses = generate_bn2a_responses(Q_true, params, alpha_matrix);
fprintf('%.1fs\n', toc(t0));

% --- Build T_hat ---
fprintf('Building T_hat... ');
t0 = tic;
T_hat = zeros(tensor_dims);
for i = 1:n
    ix = num2cell(responses(i, :) + 1);
    T_hat(ix{:}) = T_hat(ix{:}) + 1;
end
T_hat = (T_hat + alpha_smooth) / (n + alpha_smooth * tensor_size);
fprintf('%.1fs\n', toc(t0));

% --- TT-SVD ---
fprintf('Running TT-SVD (cap=%d)... ', max_rank);
t0 = tic;
[cores, ranks, info_svd] = tt_svd_full(T_hat, max_rank, 1e-12);
fprintf('%.1fs (ranks=[%s])\n', toc(t0), num2str(ranks'));

% --- Gu v2 ---
fprintf('Running Gu v2... ');
t0 = tic;
[Q_est, info_gu] = qmatrix_identification_full_gu_v2(cores, K, opts_gu);
fprintf('%.1fs\n\n', toc(t0));

step1_failed = isfield(info_gu, 'step1_failed') && info_gu.step1_failed;

if step1_failed
    fprintf('RESULT: STEP 1 FAILED\n');
    fprintf('  K_estimated = %d (expected %d)\n', info_gu.K_estimated, K);
else
    best_correct = 0;
    best_perm = 1:K;
    all_perms = perms(1:K);
    for p = 1:size(all_perms,1)
        Q_perm = Q_est(:, all_perms(p,:));
        c = sum(Q_perm(:) == Q_true(:));
        if c > best_correct
            best_correct = c;
            best_perm = all_perms(p,:);
        end
    end
    Q_aligned = Q_est(:, best_perm);
    entry = sum(Q_aligned(:) == Q_true(:));
    row = sum(all(Q_aligned == Q_true, 2));
    
    fprintf('============================================================\n');
    fprintf('RESULT: %d/%d entries, %d/%d rows\n', entry, J*K, row, J);
    if entry == J*K
        fprintf('  >>> PERFECT recovery (matches earlier successful experiment).\n');
    else
        fprintf('  >>> DEGRADED recovery (script base has drifted from original).\n');
    end
    fprintf('============================================================\n');
    
    fprintf('\n   j  | true   | est    | type   | match\n');
    fprintf('   ---+--------+--------+--------+------\n');
    for j = 1:J
        t = sprintf('%d %d %d', Q_true(j,1), Q_true(j,2), Q_true(j,3));
        e = sprintf('%d %d %d', Q_aligned(j,1), Q_aligned(j,2), Q_aligned(j,3));
        s = sum(Q_true(j,:));
        if s == 1, type_str = 'pure  ';
        elseif s == 2, type_str = 'pair  ';
        else, type_str = 'triple';
        end
        if all(Q_aligned(j,:) == Q_true(j,:))
            mark = 'OK ';
        else
            mark = 'X  ';
        end
        fprintf('   %2d | %s  | %s  | %s | %s\n', j, t, e, type_str, mark);
    end
end

save('regression_J15.mat', 'Q_true', 'Q_est', 'info_gu', 'ranks', 'params', 'n', 'J', 'K');
fprintf('\nResults saved to regression_J15.mat\n');