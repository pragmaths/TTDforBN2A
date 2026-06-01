%% FULL Q-IDENTIFICATION COMPARISON
%% Three routes to TT cores, then Gu v2 on each:
%%   Route A: TT-SVD on T_pop  (ideal, noise-free)
%%   Route B: TT-SVD on T_hat  (current pipeline; full tensor materialized)
%%   Route C: TT-cross on T_hat (new pipeline; tensor never fully materialized)

clear; clc; rng(2026);

J = 15;
K = 3;
n = 100000;
alpha_smooth = 0.5;

% --- True Q-matrix and parameters ---
Q_true = zeros(J, K);
Q_true(1, 1) = 1;  Q_true(2, 1) = 1;
Q_true(3, 2) = 1;  Q_true(4, 2) = 1;
Q_true(5, 3) = 1;  Q_true(6, 3) = 1;
multi_patterns = [1 1 0; 1 0 1; 0 1 1; 1 1 1];
for j = 7:J
    Q_true(j, :) = multi_patterns(mod(j-7, 4)+1, :);
end

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

fprintf('============================================================\n');
fprintf('FULL Q-IDENTIFICATION COMPARISON (J=%d, K=%d, n=%d)\n', J, K, n);
fprintf('============================================================\n\n');

fprintf('True Q-matrix:\n');
disp(Q_true);

% --- Build T_pop (ground truth) ---
fprintf('Building T_pop... ');
tic;
[T_pop, ~, ~] = bn2a_population_tensor(Q_true, params);
fprintf('%.1f s\n', toc);

% --- Generate samples and build T_hat ---
fprintf('Sampling n=%d responses and building T_hat... ', n);
tic;
alpha_patterns_all = zeros(2^K, K);
for a = 0:(2^K - 1)
    bits = bitget(a, K:-1:1);
    alpha_patterns_all(a+1, :) = bits;
end
alpha_indices = randi(2^K, n, 1);
alpha_matrix = alpha_patterns_all(alpha_indices, :);
responses = generate_bn2a_responses(Q_true, params, alpha_matrix);

tensor_dims = repmat(2, 1, J);
tensor_size = 2^J;
T_hat = zeros(tensor_dims);
for i = 1:n
    idx = num2cell(responses(i, :) + 1);
    T_hat(idx{:}) = T_hat(idx{:}) + 1;
end
T_hat = (T_hat + alpha_smooth) / (n + alpha_smooth * tensor_size);
fprintf('%.1f s\n\n', toc);

opts_gu.threshold_prop1 = 0.15;
opts_gu.threshold_prop2 = 0.03;
opts_gu.verbose = false;

opts_cross.n_sweeps = 8;
opts_cross.tol = 1e-10;
opts_cross.verbose = false;

% =================================================================
% ROUTE A: TT-SVD on T_pop  (noise-free baseline)
% =================================================================
fprintf('=== ROUTE A: TT-SVD on T_pop (ideal) ===\n');
tic;
[cores_A, ranks_A, ~] = tt_svd_full(T_pop, 2^K, 1e-12);
t_A_tt = toc;
fprintf('  TT-SVD: %.1fs, ranks=[%s]\n', t_A_tt, num2str(ranks_A'));

tic;
[Q_A, info_A] = qmatrix_identification_full_gu_v2(cores_A, K, opts_gu);
t_A_gu = toc;
fprintf('  Gu v2: %.1fs\n', t_A_gu);

% =================================================================
% ROUTE B: TT-SVD on T_hat  (current pipeline)
% =================================================================
fprintf('\n=== ROUTE B: TT-SVD on T_hat (baseline) ===\n');
tic;
[cores_B, ranks_B, ~] = tt_svd_full(T_hat, 2^K, 1e-12);
t_B_tt = toc;
fprintf('  TT-SVD: %.1fs, ranks=[%s]\n', t_B_tt, num2str(ranks_B'));

tic;
[Q_B, info_B] = qmatrix_identification_full_gu_v2(cores_B, K, opts_gu);
t_B_gu = toc;
fprintf('  Gu v2: %.1fs\n', t_B_gu);

% =================================================================
% ROUTE C: TT-cross on T_hat (new pipeline)
% =================================================================
fprintf('\n=== ROUTE C: TT-cross on T_hat (new) ===\n');
oracle_emp = make_empirical_oracle(T_hat);

tic;
[cores_C, info_cross_C] = tt_cross_basic(oracle_emp, J, 2^K, opts_cross);
t_C_tt = toc;
fprintf('  TT-cross: %.1fs, ranks=[%s], queries=%d/%d (%.1f%%)\n', ...
    t_C_tt, num2str(info_cross_C.ranks'), ...
    info_cross_C.n_oracle_calls, 2^J, ...
    100*double(info_cross_C.n_oracle_calls)/double(2^J));

tic;
[Q_C, info_C] = qmatrix_identification_full_gu_v2(cores_C, K, opts_gu);
t_C_gu = toc;
fprintf('  Gu v2: %.1fs\n', t_C_gu);

% =================================================================
% RESULTS COMPARISON
% =================================================================

fprintf('\n============================================================\n');
fprintf('Q-MATRIX RECOVERY COMPARISON\n');
fprintf('============================================================\n');

routes = {'A (TT-SVD on T_pop)', 'B (TT-SVD on T_hat)', 'C (TT-cross on T_hat)'};
Q_ests = {Q_A, Q_B, Q_C};

for r = 1:3
    Q_est = Q_ests{r};
    
    % Best permutation alignment
    best_perm = [];
    best_correct = 0;
    all_perms = perms(1:K);
    for p = 1:size(all_perms, 1)
        Q_perm = Q_est(:, all_perms(p, :));
        c = sum(Q_perm(:) == Q_true(:));
        if c > best_correct
            best_correct = c;
            best_perm = all_perms(p, :);
        end
    end
    Q_aligned = Q_est(:, best_perm);
    
    correct_entries = sum(Q_aligned(:) == Q_true(:));
    correct_rows = sum(all(Q_aligned == Q_true, 2));
    n_entries = J * K;
    
    fprintf('\nRoute %s:\n', routes{r});
    fprintf('  Entry-level: %d/%d (%.1f%%)\n', ...
        correct_entries, n_entries, 100*correct_entries/n_entries);
    fprintf('  Row-level:   %d/%d (%.1f%%)\n', ...
        correct_rows, J, 100*correct_rows/J);
    fprintf('  Estimated Q (aligned, perm=%s):\n', num2str(best_perm));
    
    % Show side-by-side
    fprintf('    j  | true   | est    | match\n');
    fprintf('    ---+--------+--------+------\n');
    for j = 1:J
        tstr = sprintf('%d %d %d', Q_true(j,1), Q_true(j,2), Q_true(j,3));
        estr = sprintf('%d %d %d', Q_aligned(j,1), Q_aligned(j,2), Q_aligned(j,3));
        ok = all(Q_aligned(j,:) == Q_true(j,:));
        if ok, mark = 'OK '; else, mark = 'X  '; end
        fprintf('    %2d | %s  | %s  | %s\n', j, tstr, estr, mark);
    end
end

fprintf('\n============================================================\n');
fprintf('TIMING SUMMARY\n');
fprintf('============================================================\n');
fprintf('  Route A: TT %.2fs + Gu %.2fs = %.2fs (touched %d tensor entries)\n', ...
    t_A_tt, t_A_gu, t_A_tt + t_A_gu, 2^J);
fprintf('  Route B: TT %.2fs + Gu %.2fs = %.2fs (touched %d tensor entries)\n', ...
    t_B_tt, t_B_gu, t_B_tt + t_B_gu, 2^J);
fprintf('  Route C: TT %.2fs + Gu %.2fs = %.2fs (touched %d tensor entries)\n', ...
    t_C_tt, t_C_gu, t_C_tt + t_C_gu, info_cross_C.n_oracle_calls);

save('q_identification_comparison.mat', 'Q_true', 'Q_A', 'Q_B', 'Q_C', ...
    'info_A', 'info_B', 'info_C', 'cores_A', 'cores_B', 'cores_C', ...
    'ranks_A', 'ranks_B', 'info_cross_C', 'params', 'n', 'J', 'K');

fprintf('\nSaved to q_identification_comparison.mat\n');
