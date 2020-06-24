module Bandits

using Random
using StatsPlots


struct Bandit
    k::Int
    means::Vector{Real}
    stds::Vector{Real}
    function Bandit(k::Int)
        means = 3*randn(k)
        stds = 3*randexp(k)
        new(k, means, stds)
    end
end


function pull(bandit::Bandit, i::Int)
    @assert i ≤ bandit.k "There aren't enough arms on this bandit, try a smaller number."
    return bandit.stds[i] * randn() + bandit.means[i]
end


function plot_bandit(bandit::Bandit)
    qs = [[pull(bandit, i) for _ = 1:1000] for i = 1:bandit.k]
    violin(qs)
end


struct ϵ_Greedy
    ϵ::Real
    Q::Vector{Real}
    N::Vector{Int}
    rewards::Vector{Real}
    bandit::Bandit
    function ϵ_Greedy(ϵ::Real, k::Int)
        bandit = Bandit(k)
        Q = zeros(Real, k)
        N = zeros(Int, k)
        rewards = Vector{Real}[]
        new(ϵ, Q, N, rewards, bandit)
    end
    function ϵ_Greedy(ϵ::Real, bandit::Bandit)
        Q = zeros(Real, bandit.k)
        N = zeros(Int, bandit.k)
        rewards = Vector{Real}[]
        new(ϵ, Q, N, rewards, bandit)
    end
end


function update_policy(policy::ϵ_Greedy, a::Int, reward::Real)
    append!(policy.rewards, reward)
    policy.N[a] += 1
    policy.Q[a] += 1 / policy.N[a] * (reward - policy.Q[a])
end


function act(policy::ϵ_Greedy)
    p = rand()
    k = policy.bandit.k
    if p < policy.ϵ
        a = rand(Vector(1:k))
    else
        a = argmax(policy.Q)
    end
    return a
end


function episode(policy)
    action = act(policy)
    reward = pull(policy.bandit, action)
    update_policy(policy, action, reward)
end


function compare_policies(bandit::Bandit, policy_type, parameters::Vector;
    ensemble_size=50, episode_count=5000)
    results = []
    for p in parameters
        result = []
        rewards = []
        for e = 1:ensemble_size
            policy = policy_type(p, bandit)
            for _ = 1:episode_count
                episode(policy)
            end
            push!(rewards, policy.rewards)
        end
        for ep = 1:episode_count
            total = 0
            for x = 1:ensemble_size
                total += rewards[x][ep]
            end
            push!(result, total/ensemble_size)
        end
        push!(results, result)
    end
    return results
end



export Bandit, pull, plot_bandit, episode, act, update_policy, ϵ_Greedy, compare_policies

end