module Bandits

using Random
using StatsPlots


abstract type AbstractBandit end


struct Bandit <: AbstractBandit
    k::Int
    means::Vector{Real}
    stds::Vector{Real}
    function Bandit(k::Int)
        means = 3*randn(k)
        stds = 3*randexp(k)
        new(k, means, stds)
    end
end


mutable struct NonStationaryBandit <: AbstractBandit
    k::Int
    means::Vector{Real}
    stds::Vector{Real}
    function NonStationaryBandit(k::Int)
        means = 3*randn(k)
        stds = 3*randexp(k)
        new(k, means, stds)
    end
end


function pull(bandit::Bandit, i::Int)
    @assert i ≤ bandit.k "There aren't enough arms on this bandit, try a smaller number."
    return bandit.stds[i] * randn() + bandit.means[i]
end


function pull(bandit::NonStationaryBandit, i::Int)
    @assert i ≤ bandit.k "There aren't enough arms on this bandit, try a smaller number."
    bandit.means[i] += 0.01 * randn()
    bandit.stds[i] += 0.01 * randn()
    return bandit.stds[i] * randn() + bandit.means[i]
end

function plot_bandit(bandit::AbstractBandit)
    qs = [[pull(bandit, i) for _ = 1:1000] for i = 1:bandit.k]
    violin(qs)
end


abstract type policy end
abstract type ϵ_Greedy <: policy end

struct ϵ_Greedy_sample_average <: ϵ_Greedy
    ϵ::Real
    Q::Vector{Real}
    N::Vector{Int}
    rewards::Vector{Real}
    bandit::AbstractBandit
    function ϵ_Greedy_sample_average(ϵ::Real, k::Int)
        bandit = Bandit(k)
        Q = zeros(Real, k)
        N = zeros(Int, k)
        rewards = Vector{Real}[]
        new(ϵ, Q, N, rewards, bandit)
    end
    function ϵ_Greedy_sample_average(ϵ::Real, bandit::AbstractBandit)
        Q = zeros(Real, bandit.k)
        N = zeros(Int, bandit.k)
        rewards = Vector{Real}[]
        new(ϵ, Q, N, rewards, bandit)
    end
end


struct ϵ_Greedy_with_constant_step_size <: ϵ_Greedy
    ϵ::Real
    Q::Vector{Real}
    N::Vector{Int}
    step_size::Real
    rewards::Vector{Real}
    bandit::AbstractBandit
    function ϵ_Greedy_with_constant_step_size(ϵ::Real, k::Int, step_size::Real)
        bandit = Bandit(k)
        Q = zeros(Real, k)
        N = zeros(Int, k)
        rewards = Vector{Real}[]
        new(ϵ, Q, N, step_size, rewards, bandit)
    end
    function ϵ_Greedy_with_constant_step_size(ϵ::Real, bandit::AbstractBandit, step_size::Real)
        Q = zeros(Real, bandit.k)
        N = zeros(Int, bandit.k)
        rewards = Vector{Real}[]
        new(ϵ, Q, N, step_size, rewards, bandit)
    end
end


struct ϵ_Greedy_with_unbiased_step_size <:  ϵ_Greedy
    ϵ::Real
    Q::Vector{Real}
    N::Vector{Int}
    step_size::Real
    o_n::Ref{Real}
    rewards::Vector{Real}
    bandit::AbstractBandit
    function ϵ_Greedy_with_unbiased_step_size(ϵ::Real, k::Int, step_size::Real)
        bandit = Bandit(k)
        Q = zeros(Real, k)
        N = zeros(Int, k)
        rewards = Vector{Real}[]
        o_n = 0.0
        new(ϵ, Q, N, step_size, o_n, rewards, bandit)
    end
    function ϵ_Greedy_with_unbiased_step_size(ϵ::Real, bandit::AbstractBandit, step_size::Real)
        Q = zeros(Real, bandit.k)
        N = zeros(Int, bandit.k)
        o_n = 0.0
        rewards = Vector{Real}[]
        new(ϵ, Q, N, step_size, o_n, rewards, bandit)
    end
end


function update_policy(policy::ϵ_Greedy_with_unbiased_step_size, a::Int, reward::Real)
    append!(policy.rewards, reward)
    policy.o_n[] = policy.o_n[] + policy.step_size * (1 - policy.o_n[])
    policy.N[a] += 1
    policy.Q[a] += (policy.step_size / policy.o_n[]) * (reward - policy.Q[a])
end


function update_policy(policy::ϵ_Greedy_with_constant_step_size, a::Int, reward::Real)
    append!(policy.rewards, reward)
    policy.N[a] += 1
    policy.Q[a] += policy.step_size * (reward - policy.Q[a])
end


function update_policy(policy::ϵ_Greedy_sample_average, a::Int, reward::Real)
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


function episode(policy::policy)
    action = act(policy)
    reward = pull(policy.bandit, action)
    update_policy(policy, action, reward)
end


function compare_policies(bandit::AbstractBandit, policy_type, parameters::Vector;
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


function compare_policies(policies::Vector;
    ensemble_size=50, episode_count=5000)
    results = []
    for p in policies
        result = []
        rewards = []
        for e = 1:ensemble_size
            policy = deepcopy(p)
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
