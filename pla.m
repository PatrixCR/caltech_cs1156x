function pla()
    iteration_counts = zeros(1000, 1);
    f_not_equal_g_freqs = zeros(1000, 1);
    
    for i = 1:1000
        [iteration_counts(i), f_not_equal_g_freqs(i)] = pla_run(100, 1000);
    end
    
    disp(['avg # of iteration: ', num2str(mean(iteration_counts)), '; avg freq f ~= g: ', num2str(mean(f_not_equal_g_freqs))]);
end

function [iteration_count, f_not_equal_g_freq] = pla_run(trainN, testN)
    f = target_function();
    
    [iteration_count, g] = train(trainN, f);
    
    f_not_equal_g_freq = test(testN, f, g);
end

function f = target_function()
    point1 = gen_data_in_domain(1, 2);
    point2 = gen_data_in_domain(1, 2);
    
    w2 = point2(2) - point1(2);
    w1 = -(point2(1) - point1(1));
    w0 = -(point2(2) - point1(2)) * point1(1) + (point2(1) - point1(1)) * point1(2);
    w = [w0; w1; w2];
    
    f = @(X) sign(X * w);
end

function non_match_freq = test(N, f, g)
    X = [ones(N,1) gen_data_in_domain(N, 2)];
    
    non_matches = f(X) ~= g(X);
    non_match_freq = length(find(non_matches)) / N;
end

function [iteration_count, g] = train(N, f)
    X = [ones(N,1) gen_data_in_domain(N, 2)];
    y = f(X);
    
    w = zeros(3, 1);
    h = sign(X * w);
    misclassified_idxs = find(h ~= y);
    iteration_count = 0;
    
    while ~isempty(misclassified_idxs)
        misclassified_idx = misclassified_idxs(randi(length(misclassified_idxs)));
        
        w = w + (y(misclassified_idx) * X(misclassified_idx, :))';
        h = sign(X * w);
        misclassified_idxs = find(h ~= y);
        
        iteration_count = iteration_count + 1;
    end
    
    g = @(gX) sign(gX * w);
end


function p = gen_data_in_domain(m, n)
    p = -1 + 2 * rand(m, n);
end
