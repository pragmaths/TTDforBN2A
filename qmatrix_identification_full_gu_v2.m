function [Q_estimated, info] = qmatrix_identification_full_gu_v2(cores, K, options)
% QMATRIX_IDENTIFICATION_FULL_GU_V2 - Fixed version with proper Prop 2 threshold
%
% Key fix: Use ADAPTIVE threshold for Proposition 2.
% The σ_(2^(K-1)+1) / σ_1 ratio scales with K.
% Empirical observation:
%   K=2: ratios ~0.05 vs ~0.95 (large gap)
%   K=3: ratios ~0.005 vs ~0.10 (smaller gap, ~10× difference)
%
% Solution: Use threshold that detects ORDER OF MAGNITUDE jumps.

if nargin < 3
    options = struct();
end
if ~isfield(options, 'threshold_prop1')
    options.threshold_prop1 = 0.15;  % For Proposition 1
end
if ~isfield(options, 'threshold_prop2')
    options.threshold_prop2 = 0.03;  % For Proposition 2 (adaptive default)
end
if ~isfield(options, 'verbose')
    options.verbose = true;
end
if ~isfield(options, 'min_pure_size')
    options.min_pure_size = 2;
end

J = length(cores);

if options.verbose
    fprintf('============================================================\n');
    fprintf('GU (2026) FULL Q-IDENTIFICATION VIA TT (v2)\n');
    fprintf('Implementing Proposition 1 + Proposition 2 (FIXED)\n');
    fprintf('============================================================\n\n');
    fprintf('Configuration: J=%d items, K=%d attributes\n', J, K);
    fprintf('Threshold Prop 1 (σ_3/σ_1): %.3f\n', options.threshold_prop1);
    fprintf('Threshold Prop 2 (σ_(2^(K-1)+1)/σ_1): %.3f\n\n', options.threshold_prop2);
end

%% PRE-COMPUTE TT CONTRACTIONS
E = cell(J, 1);
for j = 1:J
    G = cores{j};
    [r_prev, ~, r_next] = size(G);
    E_j = zeros(r_prev * r_prev, r_next * r_next);
    for i = 1:2
        G_i = squeeze(G(:, i, :));
        if r_prev == 1
            G_i = reshape(G_i, 1, r_next);
        elseif r_next == 1
            G_i = reshape(G_i, r_prev, 1);
        end
        E_j = E_j + kron(G_i, G_i);
    end
    E{j} = E_j;
end

L = cell(J+1, 1);
L{1} = 1;
for j = 1:J
    L{j+1} = L{j} * E{j};
end

R = cell(J+1, 1);
R{J+1} = 1;
for j = J:-1:1
    R{j} = E{j} * R{j+1};
end

%% STEP 1: PROPOSITION 1 (PURE ITEMS)

if options.verbose
    fprintf('STEP 1: Identifying pure items (Proposition 1)...\n');
end

n_pairs = J*(J-1)/2;
singular_ratios = zeros(n_pairs, 1);
pair_list = zeros(n_pairs, 2);
rank2_indicator = false(n_pairs, 1);

pair_idx = 0;
tic;
for j1 = 1:J-1
    for j2 = j1+1:J
        pair_idx = pair_idx + 1;
        pair_list(pair_idx, :) = [j1, j2];
        
        MMt = compute_MMt_pair(cores, E, L, R, j1, j2);
        
        eigs_MMt = eig(MMt);
        eigs_MMt = sort(eigs_MMt(:), 'descend');
        eigs_MMt = max(eigs_MMt, 0);
        sigma = sqrt(eigs_MMt);
        
        if length(sigma) >= 3 && sigma(1) > 1e-15
            ratio = sigma(3) / sigma(1);
        else
            ratio = 0;
        end
        singular_ratios(pair_idx) = ratio;
        rank2_indicator(pair_idx) = (ratio < options.threshold_prop1);
    end
end
prop1_time = toc;

rank2_pairs = pair_list(rank2_indicator, :);

adj = false(J, J);
for i = 1:size(rank2_pairs, 1)
    adj(rank2_pairs(i,1), rank2_pairs(i,2)) = true;
    adj(rank2_pairs(i,2), rank2_pairs(i,1)) = true;
end

clusters = find_connected_components(adj);
cluster_sizes = cellfun(@length, clusters);

if options.verbose
    fprintf('  Found %d rank-2 pairs, %d clusters in %.4f s\n', ...
        sum(rank2_indicator), length(clusters), prop1_time);
end

valid_idx = find(cluster_sizes >= options.min_pure_size);
valid_clusters = clusters(valid_idx);
valid_sizes = cluster_sizes(valid_idx);

if length(valid_clusters) < K
    if options.verbose
        fprintf('  ⚠ Not enough valid clusters\n');
    end
    valid_clusters = clusters;
    valid_sizes = cluster_sizes;
end

[~, sort_idx] = sort(valid_sizes);
sorted_clusters = valid_clusters(sort_idx);
n_select = min(K, length(sorted_clusters));
pure_clusters = sorted_clusters(1:n_select);

% Defensive check: if Step 1 did not find K usable clusters, abort cleanly.
% Each selected cluster must have at least 2 items (need two representatives).
sizes_selected = cellfun(@length, pure_clusters);
if n_select < K || any(sizes_selected < 2)
    if n_select < K
        fail_reason = sprintf('only %d cluster(s) of size >= %d found (need %d)', ...
            n_select, options.min_pure_size, K);
    else
        fail_reason = sprintf('selected clusters too small (sizes: %s); need >=2 each', ...
            num2str(sizes_selected(:)'));
    end
    fprintf('\n  STEP 1 FAILED: %s.\n', fail_reason);
    fprintf('  Returning partial result with Step 2 skipped.\n');
    Q_estimated = zeros(J, K);
    for k = 1:length(pure_clusters)
        if length(pure_clusters{k}) >= 1
            for j = pure_clusters{k}
                Q_estimated(j, k) = 1;
            end
        end
    end
    info.rank2_pairs = rank2_pairs;
    info.clusters = clusters;
    info.pure_clusters = pure_clusters;
    info.multi_items = [];
    info.K_estimated = length(clusters);
    info.singular_ratios = singular_ratios;
    info.pair_list = pair_list;
    info.threshold_prop1 = options.threshold_prop1;
    info.threshold_prop2 = options.threshold_prop2;
    info.prop1_time = prop1_time;
    info.prop2_time = 0;
    info.total_time = prop1_time;
    info.step1_failed = true;
    info.fail_reason = fail_reason;
    return;
end

pure_reps_1 = zeros(K, 1);
pure_reps_2 = zeros(K, 1);
for k = 1:K
    cluster_k = pure_clusters{k};
    pure_reps_1(k) = cluster_k(1);
    pure_reps_2(k) = cluster_k(2);
end

items_in_pure = unique(cell2mat(reshape(pure_clusters, 1, [])));
multi_items_all = setdiff(1:J, items_in_pure);

if options.verbose
    fprintf('\n  Pure clusters identified:\n');
    for k = 1:K
        fprintf('    Attribute %d: items {%s}\n', k, num2str(pure_clusters{k}));
    end
    fprintf('  Multi items: {%s}\n\n', num2str(multi_items_all));
end

%% STEP 2: PROPOSITION 2 (MULTI ITEMS) - WITH FIXED THRESHOLD

if options.verbose
    fprintf('STEP 2: Identifying multi-item Q-vectors (Proposition 2)...\n');
end

Q_estimated = zeros(J, K);
for k = 1:K
    for j = pure_clusters{k}
        Q_estimated(j, k) = 1;
    end
end

rank_threshold_position = 2^(K-1) + 1;
prop2_time = 0;
tic;

% Store all ratios for analysis
all_prop2_ratios = zeros(length(multi_items_all), K);

for mi = 1:length(multi_items_all)
    j_multi = multi_items_all(mi);
    
    if options.verbose
        fprintf('    Item %d: ', j_multi);
    end
    
    q_vector = zeros(1, K);
    
    for k = 1:K
        S1 = pure_reps_1;
        S1(k) = j_multi;
        S1 = sort(S1);
        S2 = sort(pure_reps_2);
        
        if any(ismember(S1, S2))
            if options.verbose
                fprintf('[overlap] ');
            end
            continue;
        end
        
        M_unfold = compute_general_unfolding(cores, S1, S2);
        sigma = svd(M_unfold);
        
        if length(sigma) >= rank_threshold_position && sigma(1) > 1e-15
            ratio = sigma(rank_threshold_position) / sigma(1);
            all_prop2_ratios(mi, k) = ratio;
            
            % FIXED: Use proper threshold for Proposition 2
            if ratio > options.threshold_prop2
                q_vector(k) = 1;
            end
        end
        
        if options.verbose
            if q_vector(k) == 1
                fprintf('A%d=1(%.4f) ', k, all_prop2_ratios(mi, k));
            else
                fprintf('A%d=0(%.4f) ', k, all_prop2_ratios(mi, k));
            end
        end
    end
    
    Q_estimated(j_multi, :) = q_vector;
    
    if options.verbose
        fprintf('→ q_%d = [%s]\n', j_multi, num2str(q_vector));
    end
end

prop2_time = toc;

if options.verbose
    fprintf('\n  Proposition 2 completed in %.4f seconds\n', prop2_time);
end

%% PACKAGE INFO
info.rank2_pairs = rank2_pairs;
info.clusters = clusters;
info.pure_clusters = pure_clusters;
info.pure_reps_1 = pure_reps_1;
info.pure_reps_2 = pure_reps_2;
info.multi_items = multi_items_all;
info.K_estimated = length(clusters);
info.singular_ratios = singular_ratios;
info.pair_list = pair_list;
info.prop2_ratios = all_prop2_ratios;
info.threshold_prop1 = options.threshold_prop1;
info.threshold_prop2 = options.threshold_prop2;
info.prop1_time = prop1_time;
info.prop2_time = prop2_time;
info.total_time = prop1_time + prop2_time;
info.step1_failed = false;

if options.verbose
    fprintf('\n');
    fprintf('============================================================\n');
    fprintf('Q-IDENTIFICATION COMPLETE (v2 - Fixed)\n');
    fprintf('  Total time: %.4f s\n', info.total_time);
    fprintf('============================================================\n');
end

end


%% ============================================================================
%% HELPER FUNCTIONS
%% ============================================================================

function MMt = compute_MMt_pair(cores, E, L, R, j1, j2)
J = length(cores);
MMt = zeros(4, 4);

G_j1 = cores{j1};
G_j2 = cores{j2};
[r_j1_prev, ~, r_j1_next] = size(G_j1);
[r_j2_prev, ~, r_j2_next] = size(G_j2);

if j1 + 1 <= j2 - 1
    E_between = E{j1+1};
    for j = j1+2:j2-1
        E_between = E_between * E{j};
    end
else
    E_between = 1;
end

L_left = L{j1};
R_right = R{j2+1};

for v1 = 0:1
    for v2 = 0:1
        for v1p = 0:1
            for v2p = 0:1
                G_j1_v1 = squeeze(G_j1(:, v1+1, :));
                G_j1_v1p = squeeze(G_j1(:, v1p+1, :));
                G_j2_v2 = squeeze(G_j2(:, v2+1, :));
                G_j2_v2p = squeeze(G_j2(:, v2p+1, :));
                
                if r_j1_prev == 1
                    G_j1_v1 = reshape(G_j1_v1, 1, r_j1_next);
                    G_j1_v1p = reshape(G_j1_v1p, 1, r_j1_next);
                elseif r_j1_next == 1
                    G_j1_v1 = reshape(G_j1_v1, r_j1_prev, 1);
                    G_j1_v1p = reshape(G_j1_v1p, r_j1_prev, 1);
                end
                
                if r_j2_prev == 1
                    G_j2_v2 = reshape(G_j2_v2, 1, r_j2_next);
                    G_j2_v2p = reshape(G_j2_v2p, 1, r_j2_next);
                elseif r_j2_next == 1
                    G_j2_v2 = reshape(G_j2_v2, r_j2_prev, 1);
                    G_j2_v2p = reshape(G_j2_v2p, r_j2_prev, 1);
                end
                
                K_j1 = kron(G_j1_v1, G_j1_v1p);
                K_j2 = kron(G_j2_v2, G_j2_v2p);
                
                val = L_left * K_j1 * E_between * K_j2 * R_right;
                
                row_idx = v1 * 2 + v2 + 1;
                col_idx = v1p * 2 + v2p + 1;
                MMt(row_idx, col_idx) = val;
            end
        end
    end
end

MMt = (MMt + MMt') / 2;
end


function M = compute_general_unfolding(cores, S1, S2)
J = length(cores);
n_S1 = length(S1);
n_S2 = length(S2);

M = zeros(2^n_S1, 2^n_S2);

S1_sorted = sort(S1(:))';
S2_sorted = sort(S2(:))';

for s1_idx = 0:(2^n_S1 - 1)
    val_S1_pattern = de2bi(s1_idx, n_S1, 'left-msb');
    
    for s2_idx = 0:(2^n_S2 - 1)
        val_S2_pattern = de2bi(s2_idx, n_S2, 'left-msb');
        
        result = 1;
        
        for j = 1:J
            is_in_S1 = ismember(j, S1_sorted);
            is_in_S2 = ismember(j, S2_sorted);
            
            if is_in_S1
                pos = find(S1_sorted == j);
                val = val_S1_pattern(pos);
                G = cores{j};
                [r_prev, ~, r_next] = size(G);
                G_slice = squeeze(G(:, val+1, :));
                if r_prev == 1
                    G_slice = reshape(G_slice, 1, r_next);
                elseif r_next == 1
                    G_slice = reshape(G_slice, r_prev, 1);
                end
                if isscalar(result) && result == 1
                    result = G_slice;
                else
                    result = result * G_slice;
                end
            elseif is_in_S2
                pos = find(S2_sorted == j);
                val = val_S2_pattern(pos);
                G = cores{j};
                [r_prev, ~, r_next] = size(G);
                G_slice = squeeze(G(:, val+1, :));
                if r_prev == 1
                    G_slice = reshape(G_slice, 1, r_next);
                elseif r_next == 1
                    G_slice = reshape(G_slice, r_prev, 1);
                end
                if isscalar(result) && result == 1
                    result = G_slice;
                else
                    result = result * G_slice;
                end
            else
                G = cores{j};
                [r_prev, ~, r_next] = size(G);
                G_summed = squeeze(sum(G, 2));
                if r_prev == 1
                    G_summed = reshape(G_summed, 1, r_next);
                elseif r_next == 1
                    G_summed = reshape(G_summed, r_prev, 1);
                end
                if isscalar(result) && result == 1
                    result = G_summed;
                else
                    result = result * G_summed;
                end
            end
        end
        
        M(s1_idx + 1, s2_idx + 1) = result(1);
    end
end

end


function clusters = find_connected_components(adj)
J = size(adj, 1);
visited = false(J, 1);
clusters = {};

for start = 1:J
    if ~visited(start)
        component = [];
        queue = start;
        
        while ~isempty(queue)
            node = queue(1);
            queue(1) = [];
            
            if ~visited(node)
                visited(node) = true;
                component = [component, node];
                
                neighbors = find(adj(node, :));
                for n = neighbors
                    if ~visited(n)
                        queue = [queue, n];
                    end
                end
            end
        end
        
        clusters{end+1} = sort(component);
    end
end

end
