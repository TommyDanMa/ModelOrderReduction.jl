# Reference experiment for balanced POD. Not part of the package API.
#
# This file does not modify ModelOrderReduction sources. It builds a 4-state,
# 2-input, 2-output continuous-time system, forms balanced POD from exact
# impulse-response samples, and compares the result with `baltrunc`.
#
# Algorithm, following Rowley, Int. J. Bifurcation Chaos 15 (2005), eqs. (19)–(24):
#
#   x_j(t) = exp(A t) b_j,          z_i(t) = exp(A' t) c_i,
#   X = [ ... sqrt(δ_k) x_j(t_k) ... ],
#   Y = [ ... sqrt(δ_k) z_i(t_k) ... ],
#   Y' X = U Σ V',
#   T_r = X V_r Σ_r^{-1/2},         S = Σ_r^{-1/2} U_r' Y',
#   A_r = S A T_r,  B_r = S B,  C_r = C T_r,  D_r = D.
#
# The weights δ_k are the composite trapezoidal weights on the nodes t_k.
# Columns are ordered as in Rowley's (20): every time node of input 1, then
# input 2; likewise for the adjoint columns and the output channels.
#
# Assumptions, stated so the numbers below have a precise meaning:
#
# 1. The open-loop matrix is Hurwitz, so the infinite-horizon Gramians exist and
#    `baltrunc` is the right classical target. Plain truncation is compared, not
#    residualization. D is feedthrough and is copied unchanged.
# 2. Snapshots are samples of the matrix exponential, not of an ODE solver.
#    Discrepancies are therefore horizon truncation and trapezoidal quadrature,
#    not local integration tolerance. An ODE-based implementation would hand the
#    same weighted samples to the SVD.
# 3. The state inner product is Euclidean, so the adjoint generator is A' and
#    the adjoint impulse initial conditions are the columns of C'. There is no
#    output projection: both outputs are simulated.
# 4. Willcox and Peraire (AIAA J. 40, 2002) compress each Gramian by its own POD
#    before balancing. That intermediate truncation is not used. Tu and Rowley
#    (J. Comput. Phys. 231, 2012) append analytic tail snapshots. That correction
#    is not used; the horizon study measures the tail it would remove.
# 5. Biorthogonality S T_r = I is an algebraic identity of the SVD whenever the
#    retained singular values are positive. It does not test the quadrature.
#    A square full-order factor is a similarity, so it reproduces the transfer
#    function even when the Hankel values are still wrong. Those two traps are
#    checked on a deliberately coarse grid.

using LinearAlgebra
using Printf
using ModelOrderReduction: baltrunc

const A_PLANT = [
    -1.0 20.0 0.0 0.0
    0.0 -2.0 15.0 0.0
    0.0 0.0 -3.0 10.0
    0.0 0.0 0.0 -6.0
]
const B_PLANT = [
    1.0 0.0
    0.4 1.0
    0.0 0.5
    0.2 -0.3
]
const C_PLANT = [
    1.0 0.2 0.0 0.4
    0.0 1.0 0.3 -0.2
]
const D_PLANT = [
    0.15 -0.02
    0.01 0.08
]

struct BPODModel{T <: AbstractFloat}
    A::Matrix{T}
    B::Matrix{T}
    C::Matrix{T}
    D::Matrix{T}
    hsv::Vector{T}
    Tr::Matrix{T}
    S::Matrix{T}
    Pc::Matrix{T}
    Po::Matrix{T}
end

struct Check
    name::String
    ok::Bool
    detail::String
end

function uniform_nodes(T::Real, dt::Real)
    nsteps = round(Int, T / dt)
    abs(nsteps * dt - T) <= 100 * eps(float(T)) * max(T, one(float(T))) ||
        throw(ArgumentError("dt = $dt does not divide the horizon T = $T"))
    return collect(range(float(T) * 0, float(T); length = nsteps + 1))
end

"""
    trapezoidal_weights(t)

Composite trapezoidal weights for a strictly increasing node set. The rule is
exact for affine integrands, and `sum(weights) == t[end] - t[1]` by telescoping.
"""
function trapezoidal_weights(t::AbstractVector{<:Real})
    nt = length(t)
    nt >= 2 || throw(ArgumentError("trapezoidal rule needs at least two nodes"))
    all(>(0), diff(t)) || throw(ArgumentError("time nodes must be strictly increasing"))
    w = similar(float.(t))
    w[1] = (t[2] - t[1]) / 2
    for k in 2:(nt - 1)
        w[k] = (t[k + 1] - t[k - 1]) / 2
    end
    w[nt] = (t[nt] - t[nt - 1]) / 2
    return w
end

"""
    impulse_factor(generator, columns, t, w)

Snapshot factor whose outer product is the trapezoidal Gramian.

The column for channel `j` at node `k` is `sqrt(w[k]) * exp(generator * t[k]) * columns[:, j]`.
For the primal, `generator = A` and `columns = B`. For the adjoint, `generator = A'`
and `columns = C'`.
"""
function impulse_factor(generator::AbstractMatrix, columns::AbstractMatrix, t, w)
    n, m = size(columns)
    nt = length(t)
    length(w) == nt || throw(ArgumentError("weights and nodes must have the same length"))
    X = zeros(eltype(generator), n, m * nt)
    for k in eachindex(t)
        response = exp(generator * t[k])
        scale = sqrt(w[k])
        for j in 1:m
            X[:, (j - 1) * nt + k] = scale * (response * columns[:, j])
        end
    end
    return X
end

"""
    gramian_from_impulses(generator, columns, t, w)

Same quadrature as [`impulse_factor`](@ref), accumulated as an explicit sum of
outer products. Used to check that the square-root weights are applied once.
"""
function gramian_from_impulses(generator::AbstractMatrix, columns::AbstractMatrix, t, w)
    n = size(generator, 1)
    G = zeros(eltype(generator), n, n)
    for j in 1:size(columns, 2)
        column = columns[:, j]
        for k in eachindex(t)
            x = exp(generator * t[k]) * column
            G .+= w[k] .* (x * x')
        end
    end
    return G
end

"""
Square-root factor `L` with `W ≈ L L'`. Eigenvalues below `eps * λmax` are dropped.
This mirrors `ModelOrderReduction._psd_sqrt` so the independent reference is the
same square-root balancing method, not a second algorithm.
"""
function psd_sqrt(W::AbstractMatrix{T}) where {T <: AbstractFloat}
    F = eigen(Symmetric(Matrix{T}(W)))
    λ = F.values
    λmax = maximum(abs, λ)
    thresh = eps(T) * λmax
    kept = findall(>(thresh), λ)
    isempty(kept) && return zeros(T, size(W, 1), 0)
    return F.vectors[:, kept] * Diagonal(sqrt.(λ[kept]))
end

function sqrt_balance(A, B, C, D, Pc, Po, r)
    Lc = psd_sqrt(Pc)
    Lo = psd_sqrt(Po)
    F = svd(Lo' * Lc)
    length(F.S) >= r || throw(ArgumentError("requested order $r exceeds the numerical rank"))
    Sinv = Diagonal(inv.(sqrt.(F.S[1:r])))
    Tr = Lc * F.V[:, 1:r] * Sinv
    Sl = Sinv * F.U[:, 1:r]' * Lo'
    return BPODModel(
        Sl * A * Tr, Sl * B, C * Tr, Matrix(D), F.S[1:r], Tr, Sl, Matrix(Pc), Matrix(Po)
    )
end

function balanced_pod(A, B, C, D, t; r)
    w = trapezoidal_weights(t)
    X = impulse_factor(A, B, t, w)
    Y = impulse_factor(A', C', t, w)
    Pc = X * X'
    Po = Y * Y'
    F = svd(Y' * X)
    r <= count(>(0), F.S) || throw(ArgumentError("requested order $r exceeds the snapshot rank"))
    F.S[r] > 0 || throw(ArgumentError("retained singular value $r is not positive"))
    Sinv = Diagonal(inv.(sqrt.(F.S[1:r])))
    Tr = X * F.V[:, 1:r] * Sinv
    Sl = Sinv * F.U[:, 1:r]' * Y'
    hsv = F.S[1:min(size(A, 1), length(F.S))]
    return BPODModel(Sl * A * Tr, Sl * B, C * Tr, Matrix(D), hsv, Tr, Sl, Pc, Po)
end

function finite_horizon_gramians(A, Pc, Po, T)
    transition = exp(A * T)
    Pc_horizon = Pc - transition * Pc * transition'
    Po_horizon = Po - transition' * Po * transition
    return Symmetric(Pc_horizon), Symmetric(Po_horizon)
end

function relative_error(got, reference)
    denom = norm(reference)
    denom > 0 || throw(ArgumentError("relative error requires a nonzero reference"))
    return norm(got - reference) / denom
end

function component_relative_error(got, reference)
    length(got) >= length(reference) ||
        throw(ArgumentError("approximation has fewer values than the reference"))
    return [abs(got[i] - reference[i]) / reference[i] for i in eachindex(reference)]
end

function principal_angles(U::AbstractMatrix, V::AbstractMatrix)
    size(U, 2) == size(V, 2) || throw(ArgumentError("bases must have the same width"))
    r = size(U, 2)
    Qu = Matrix(qr(U).Q)[:, 1:r]
    Qv = Matrix(qr(V).Q)[:, 1:r]
    singular = svdvals(Qu' * Qv)
    return acos.(clamp.(singular, 0, 1))
end

function mode_cosines(left::AbstractMatrix, right::AbstractMatrix)
    r = size(left, 2)
    size(right, 2) == r || throw(ArgumentError("mode sets must have the same width"))
    C = zeros(r, r)
    for j in 1:r, i in 1:r
        C[i, j] = dot(left[:, i], right[:, j]) / (norm(left[:, i]) * norm(right[:, j]))
    end
    return C
end

"""
Flip columns of the BPOD basis so each one has positive overlap with the
corresponding `baltrunc` mode. Valid only when the modes are already paired in
Hankel-value order, which requires separated singular values.
"""
function align_signature(reference_modes::AbstractMatrix, model::BPODModel)
    r = size(model.Tr, 2)
    signs = ones(r)
    for j in 1:r
        overlap = dot(reference_modes[:, j], model.Tr[:, j])
        if overlap < 0
            signs[j] = -1
        end
    end
    E = Diagonal(signs)
    return BPODModel(
        E * model.A * E,
        E * model.B,
        model.C * E,
        model.D,
        model.hsv,
        model.Tr * E,
        E * model.S,
        model.Pc,
        model.Po,
    ), signs
end

function pole_set_distance(λ, μ)
    length(λ) == length(μ) || throw(ArgumentError("pole sets must have the same length"))
    best = Inf
    used = falses(length(μ))
    function search(i, acc)
        acc >= best && return
        if i > length(λ)
            best = acc
            return
        end
        for j in eachindex(μ)
            used[j] && continue
            used[j] = true
            search(i + 1, max(acc, abs(λ[i] - μ[j])))
            used[j] = false
        end
    end
    search(1, 0.0)
    return best
end

function h2_difference(A, B, C, Ar, Br, Cr)
    maximum(real, eigvals(A)) < 0 || throw(ArgumentError("full-order A is not Hurwitz"))
    maximum(real, eigvals(Ar)) < 0 ||
        throw(ArgumentError("reduced A is not Hurwitz; the H2 error is infinite"))
    n = size(A, 1)
    r = size(Ar, 1)
    Ae = [A zeros(n, r); zeros(r, n) Ar]
    Be = [B; Br]
    Ce = [C -Cr]
    P = lyap(Ae, Be * Be')
    return sqrt(abs(tr(Ce * P * Ce')))
end

function h2_norm(A, B, C, Pc)
    return sqrt(abs(tr(C * Pc * C')))
end

function transfer(A, B, C, D, ω)
    n = size(A, 1)
    return C * (((im * ω) * I(n) - A) \ B) + D
end

function frequency_samples()
    ω = 10 .^ range(-3, 2; length = 161)
    resonant = range(0.2, 2.0; length = 81)
    return sort(vcat(ω, resonant))
end

function peak_gain_error(A, B, C, D, Ar, Br, Cr, Dr, ω)
    peaks = [opnorm(transfer(A, B, C, D, ωk) - transfer(Ar, Br, Cr, Dr, ωk)) for ωk in ω]
    return maximum(peaks)
end

function step_output(A, B, C, D, u, times)
    Bu = B * u
    Du = D * u
    Y = zeros(size(C, 1), length(times))
    for (k, tk) in enumerate(times)
        x = tk == 0 ? zeros(size(A, 1)) : A \ (exp(A * tk) * Bu - Bu)
        Y[:, k] = C * x + Du
    end
    return Y
end

function zoh_output(A, B, C, D, U, dt)
    n = size(A, 1)
    Ad = exp(A * dt)
    Bd = A \ ((Ad - I(n)) * B)
    x = zeros(eltype(A), n)
    Y = zeros(eltype(A), size(C, 1), size(U, 2))
    for k in axes(U, 2)
        uk = U[:, k]
        Y[:, k] = C * x + D * uk
        x = Ad * x + Bd * uk
    end
    return Y
end

function modal_galerkin(A, B, C, D, r)
    F = eigen(A)
    order = sortperm(real.(F.values); rev = true)
    keep = order[1:r]
    maximum(abs, imag.(F.values[keep])) < 1.0e-10 ||
        throw(ArgumentError("selected modal subspace is not a real invariant subspace"))
    basis = real.(F.vectors[:, keep])
    Q = Matrix(qr(basis).Q)[:, 1:r]
    return BPODModel(
        Q' * A * Q, Q' * B, C * Q, Matrix(D), Float64[], Q, Matrix(Q'), zeros(0, 0), zeros(0, 0)
    )
end

function biorthogonal_modal(A, B, C, D, r)
    right = eigen(A)
    left = eigen(A')
    order = sortperm(real.(right.values); rev = true)
    keep = order[1:r]
    Vr = real.(right.vectors[:, keep])
    # Pair each retained right eigenvalue with the left eigenvector of A' for
    # the conjugate eigenvalue, which is the left eigenvector of A.
    Wl = zeros(size(A, 1), r)
    for (j, idx) in enumerate(keep)
        target = conj(right.values[idx])
        partner = argmin(abs.(left.values .- target))
        Wl[:, j] = real.(left.vectors[:, partner])
    end
    mass = Wl' * Vr
    return BPODModel(
        Matrix(mass \ (Wl' * A * Vr)),
        Matrix(mass \ (Wl' * B)),
        C * Vr,
        Matrix(D),
        Float64[],
        Vr,
        Matrix(inv(mass) * Wl'),
        zeros(0, 0),
        zeros(0, 0),
    )
end

function observed_orders(steps, errors)
    return [log(errors[i] / errors[i + 1]) / log(steps[i] / steps[i + 1]) for i in 1:(length(steps) - 1)]
end

"""
Leading principal subsystem of a BPOD model built at a higher order.

Column `j` of `T_r` and row `j` of `S` are scaled by `σ_j^{-1/2}` independently of
how many later modes are retained, so the order-`r` realization is exactly the
leading `r × r` block.
"""
function leading_reduction(model::BPODModel, r::Integer)
    1 <= r <= size(model.A, 1) || throw(ArgumentError("order $r is outside 1:$(size(model.A, 1))"))
    return BPODModel(
        model.A[1:r, 1:r],
        model.B[1:r, :],
        model.C[:, 1:r],
        model.D,
        model.hsv,
        model.Tr[:, 1:r],
        model.S[1:r, :],
        model.Pc,
        model.Po,
    )
end

function record!(checks, name, ok, detail = "")
    push!(checks, Check(string(name), ok, string(detail)))
    return ok
end

function print_vector(label, values)
    print(label)
    for value in values
        @printf("  %.6e", value)
    end
    println()
end

function main()
    A, B, C, D = A_PLANT, B_PLANT, C_PLANT, D_PLANT
    checks = Check[]
    println("Balanced POD reference experiment")
    println("System: 4 states, 2 inputs, 2 outputs, continuous time")
    println("A =")
    show(stdout, "text/plain", A)
    println("\nB =")
    show(stdout, "text/plain", B)
    println("\nC =")
    show(stdout, "text/plain", C)
    println("\nD =")
    show(stdout, "text/plain", D)
    println()

    spectrum = eigvals(A)
    spectral_abscissa = maximum(real, spectrum)
    numerical_abscissa = maximum(real, eigvals(Symmetric((A + A') / 2)))
    nonnormality = opnorm(A * A' - A' * A)
    eigenvector_condition = cond(eigen(A).vectors)
    growth_times = collect(0:0.01:10)
    growth = map(t -> opnorm(exp(A * t)), growth_times)
    transient_gain, transient_index = findmax(growth)
    println("Open-loop spectrum: ", spectrum)
    @printf("Spectral abscissa:       %.6e\n", spectral_abscissa)
    @printf("Numerical abscissa:      %.6e\n", numerical_abscissa)
    @printf("||A A' - A' A||_2:       %.6e\n", nonnormality)
    @printf("Eigenvector condition:   %.6e\n", eigenvector_condition)
    @printf("max ||exp(A t)||_2:      %.6e at t = %.2f\n", transient_gain, growth_times[transient_index])
    record!(checks, "A is Hurwitz", spectral_abscissa < 0, @sprintf("%.3e", spectral_abscissa))
    record!(checks, "A is non-normal", nonnormality > 1, @sprintf("%.3e", nonnormality))
    record!(
        checks,
        "transient growth exceeds the spectral envelope",
        numerical_abscissa > 0 && transient_gain > 10,
        @sprintf("abscissa %.3e, gain %.3e", numerical_abscissa, transient_gain),
    )

    Pc = Matrix(Symmetric(lyap(A, B * B')))
    Po = Matrix(Symmetric(lyap(A', C' * C)))
    controllability_residual = opnorm(A * Pc + Pc * A' + B * B') / opnorm(B * B')
    observability_residual = opnorm(A' * Po + Po * A + C' * C) / opnorm(C' * C)
    @printf("Lyapunov residual, controllability: %.3e\n", controllability_residual)
    @printf("Lyapunov residual, observability:   %.3e\n", observability_residual)
    record!(checks, "Lyapunov residuals are roundoff", max(controllability_residual, observability_residual) < 1.0e-10)

    reference = sqrt_balance(A, B, C, D, Pc, Po, 4)
    classical = baltrunc(A, B, C, D; n = 4, atol = 0.0, rtol = 0.0)
    classical_aligned_source, _ = align_signature(classical.Tr, reference)
    hsv_reference_error = maximum(component_relative_error(reference.hsv, classical.hsv))
    sqrt_matrix_error = relative_error(classical_aligned_source.A, classical.A)
    println("Hankel singular values from baltrunc: ", classical.hsv)
    @printf("Independent square-root HSV discrepancy: %.3e\n", hsv_reference_error)
    @printf("Independent square-root A_r discrepancy: %.3e\n", sqrt_matrix_error)
    @printf("baltrunc biorthogonality ||S T - I||:    %.3e\n", opnorm(classical.S * classical.Tr - I(4)))
    record!(checks, "independent square-root HSVs match baltrunc", hsv_reference_error < 1.0e-9)
    record!(checks, "independent square-root realization matches baltrunc", sqrt_matrix_error < 1.0e-8)
    full_h2 = h2_norm(A, B, C, Pc)
    @printf("Full-order H2 norm: %.6e\n", full_h2)

    println("\nTrapezoidal weight checks")
    irregular = [0.0, 0.2, 0.5, 0.9, 1.0]
    irregular_weights = trapezoidal_weights(irregular)
    linear_integrand = 3 .+ 2 .* irregular
    linear_error = abs(sum(irregular_weights .* linear_integrand) - 4.0)
    uniform = uniform_nodes(16.0, 0.0625)
    uniform_weights = trapezoidal_weights(uniform)
    exponential_exact = 1 - exp(-16.0)
    exponential_error = sum(uniform_weights .* exp.(-uniform)) - exponential_exact
    halved = uniform_nodes(16.0, 0.03125)
    halved_error = sum(trapezoidal_weights(halved) .* exp.(-halved)) - exponential_exact
    exponential_order = log(abs(exponential_error / halved_error)) / log(2)
    euler_maclaurin = (0.0625^2 / 12) * (1 - exp(-16.0))
    telescope_error = abs(sum(uniform_weights) - (uniform[end] - uniform[1]))
    @printf("Affine integrand on an irregular mesh, absolute error: %.3e\n", linear_error)
    @printf("Weight telescoping error on [0, 16]:                  %.3e\n", telescope_error)
    @printf("Trapezoidal error for exp(-t) at dt = 1/16:           %.3e\n", exponential_error)
    @printf("Euler–Maclaurin leading term h^2/12 (f'(T)-f'(0)):    %.3e\n", euler_maclaurin)
    @printf("Observed order on halving that step:                  %.6f\n", exponential_order)
    record!(checks, "trapezoidal rule integrates affine functions exactly", linear_error < 1.0e-12)
    record!(checks, "trapezoidal weights telescope to the interval length", telescope_error < 1.0e-12)
    record!(
        checks,
        "scalar exponential quadrature is second-order accurate",
        1.99 <= exponential_order <= 2.01 && abs(exponential_error - euler_maclaurin) / euler_maclaurin < 1.0e-3,
        @sprintf("order %.6f, error %.3e, predicted %.3e", exponential_order, exponential_error, euler_maclaurin),
    )

    small_nodes = uniform_nodes(2.0, 0.5)
    small_weights = trapezoidal_weights(small_nodes)
    primal_factor = impulse_factor(A, B, small_nodes, small_weights)
    adjoint_factor = impulse_factor(A', C', small_nodes, small_weights)
    primal_sum = gramian_from_impulses(A, B, small_nodes, small_weights)
    adjoint_sum = gramian_from_impulses(A', C', small_nodes, small_weights)
    primal_factor_error = relative_error(primal_factor * primal_factor', primal_sum)
    adjoint_factor_error = relative_error(adjoint_factor * adjoint_factor', adjoint_sum)
    @printf("Primal factor versus explicit impulse sum:  %.3e\n", primal_factor_error)
    @printf("Adjoint factor versus explicit impulse sum: %.3e\n", adjoint_factor_error)
    record!(checks, "snapshot factors reproduce the weighted impulse sums", max(primal_factor_error, adjoint_factor_error) < 1.0e-10)

    println("\nCoarse grid: biorthogonality and full-order input-output match are not accuracy")
    coarse = balanced_pod(A, B, C, D, uniform_nodes(2.0, 1.0); r = 4)
    coarse_r2 = balanced_pod(A, B, C, D, uniform_nodes(2.0, 1.0); r = 2)
    classical_r2 = baltrunc(A, B, C, D; n = 2, atol = 0.0, rtol = 0.0)
    coarse_hsv_error = maximum(component_relative_error(coarse.hsv, classical.hsv))
    coarse_h2 = h2_difference(A, B, C, coarse.A, coarse.B, coarse.C)
    coarse_r2_h2 = h2_difference(A, B, C, coarse_r2.A, coarse_r2.B, coarse_r2.C)
    classical_r2_h2 = h2_difference(A, B, C, classical_r2.A, classical_r2.B, classical_r2.C)
    coarse_projector = relative_error(coarse_r2.Tr * coarse_r2.S, classical_r2.Tr * classical_r2.S)
    coarse_aligned, _ = align_signature(classical.Tr, coarse)
    coarse_Ar = relative_error(coarse_aligned.A, classical.A)
    @printf("||S T - I|| at r = 4:                    %.3e\n", opnorm(coarse.S * coarse.Tr - I(4)))
    @printf("||S T - I|| at r = 2:                    %.3e\n", opnorm(coarse_r2.S * coarse_r2.Tr - I(2)))
    @printf("max relative HSV error:                  %.3e\n", coarse_hsv_error)
    print_vector("coarse HSVs: ", coarse.hsv)
    print_vector("exact  HSVs: ", classical.hsv)
    @printf("r = 4 H2 distance to the plant:          %.3e\n", coarse_h2)
    @printf("r = 4 pole distance to the plant:        %.3e\n", pole_set_distance(eigvals(coarse.A), spectrum))
    @printf("r = 4 relative A_r error versus baltrunc: %.3e\n", coarse_Ar)
    @printf("r = 2 H2 error, BPOD / baltrunc:         %.6e / %.6e\n", coarse_r2_h2, classical_r2_h2)
    @printf("r = 2 oblique-projector error:           %.3e\n", coarse_projector)
    record!(checks, "coarse biorthogonality holds for a bad quadrature", opnorm(coarse.S * coarse.Tr - I(4)) < 1.0e-8)
    record!(checks, "coarse Hankel values are nevertheless inaccurate", coarse_hsv_error > 0.2, @sprintf("%.3e", coarse_hsv_error))
    record!(checks, "full-order coarse model still reproduces the plant poles", pole_set_distance(eigvals(coarse.A), spectrum) < 1.0e-8)
    record!(checks, "full-order coarse model still reproduces the plant H2 behavior", coarse_h2 < 1.0e-5)
    record!(checks, "coarse balancing coordinates differ from baltrunc", coarse_Ar > 0.5, @sprintf("%.3e", coarse_Ar))
    record!(
        checks,
        "coarse order-2 model misses the balanced input-output error",
        abs(coarse_r2_h2 - classical_r2_h2) / classical_r2_h2 > 0.2,
        @sprintf("%.3e versus %.3e", coarse_r2_h2, classical_r2_h2),
    )

    println("\nQuadrature refinement at fixed horizon T = 16")
    println("The tail ||exp(AT) Pc exp(A'T)|| / ||Pc|| at T = 16 is near roundoff,")
    println("so this table isolates the trapezoidal contribution.")
    println("dt         nt    Pc rel     Po rel    ordPc   σ1 rel     σ2 rel     σ3 rel     σ4 rel")
    quadrature_steps = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125]
    quadrature = NamedTuple[]
    for dt in quadrature_steps
        model = balanced_pod(A, B, C, D, uniform_nodes(16.0, dt); r = 4)
        Pc_error = relative_error(model.Pc, Pc)
        Po_error = relative_error(model.Po, Po)
        hsv_error = component_relative_error(model.hsv, classical.hsv)
        push!(quadrature, (; dt, nt = length(uniform_nodes(16.0, dt)), Pc_error, Po_error, hsv_error, model))
    end
    pc_orders = observed_orders(quadrature_steps, [row.Pc_error for row in quadrature])
    hsv_orders = [observed_orders(quadrature_steps, [row.hsv_error[mode] for row in quadrature]) for mode in 1:4]
    for (index, row) in enumerate(quadrature)
        order = index == 1 ? NaN : pc_orders[index - 1]
        @printf(
            "%9.6f  %5d  %.3e  %.3e  %6.3f  %.3e  %.3e  %.3e  %.3e\n",
            row.dt, row.nt, row.Pc_error, row.Po_error, order,
            row.hsv_error[1], row.hsv_error[2], row.hsv_error[3], row.hsv_error[4],
        )
    end
    println("Observed HSV orders on successive halvings, modes 1 to 4:")
    for mode in 1:4
        print_vector("  mode $mode:", hsv_orders[mode])
    end
    if hsv_orders[4][1] < 1
        println("The smallest Hankel value is still pre-asymptotic on the coarsest meshes.")
        println("Its relative error need not decrease at dt = 1/2, and the first observed order can be negative.")
        println("The order-two checks below use only the refinements that start at dt = 1/16.")
    end
    fine_orders = 4:length(quadrature_steps)-1  # refinements beginning at dt = 0.0625
    pc_asymptotic = pc_orders[fine_orders]
    hsv_asymptotic = reduce(vcat, [orders[fine_orders] for orders in hsv_orders])
    record!(
        checks,
        "controllability Gramian converges at order two",
        all(order -> 1.8 <= order <= 2.2, pc_asymptotic),
        string(pc_asymptotic),
    )
    record!(
        checks,
        "all four Hankel values converge at order two once the mesh is fine",
        all(order -> 1.8 <= order <= 2.2, hsv_asymptotic),
        string(hsv_asymptotic),
    )
    monotone_hsv = all(mode -> issorted([row.hsv_error[mode] for row in quadrature[3:end]]; rev = true), 1:4)
    record!(checks, "per-mode Hankel errors decrease monotonically for dt <= 1/8", monotone_hsv)

    route_model = quadrature[end - 1].model
    route_hsvs = svdvals(psd_sqrt(route_model.Po)' * psd_sqrt(route_model.Pc))
    route_difference = maximum(component_relative_error(route_hsvs, route_model.hsv))
    @printf("Interaction-SVD versus empirical-Gramian square-root, max relative HSV gap: %.3e\n", route_difference)
    record!(
        checks,
        "snapshot interaction SVD and empirical square-root method agree",
        route_difference < 1.0e-8,
        @sprintf("%.3e", route_difference),
    )

    println("\nHorizon refinement at fixed dt = 1/32")
    println("Finite-horizon HSVs come from the exact truncated Gramians")
    println("Pc(T) = Pc - exp(AT) Pc exp(A'T) and Po(T) = Po - exp(A'T) Po exp(AT).")
    println("T      tail Pc     vs infinite   vs horizon    σ1 vs inf")
    horizon_steps = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0]
    horizon_rows = NamedTuple[]
    for T in horizon_steps
        model = balanced_pod(A, B, C, D, uniform_nodes(T, 0.03125); r = 4)
        Pc_T, Po_T = finite_horizon_gramians(A, Pc, Po, T)
        finite_hsv = svdvals(psd_sqrt(Po_T)' * psd_sqrt(Pc_T))
        tail = norm(Pc - Pc_T) / norm(Pc)
        versus_infinite = maximum(component_relative_error(model.hsv, classical.hsv))
        versus_horizon = maximum(component_relative_error(model.hsv, finite_hsv[1:4]))
        leading = component_relative_error(model.hsv, classical.hsv)[1]
        push!(horizon_rows, (; T, tail, versus_infinite, versus_horizon, leading))
        @printf("%5.1f  %.3e  %.3e     %.3e    %.3e\n", T, tail, versus_infinite, versus_horizon, leading)
    end
    tails = [row.tail for row in horizon_rows]
    record!(checks, "Gramian tail decreases as the horizon grows", issorted(tails; rev = true) && tails[end] < 1.0e-10)
    record!(
        checks,
        "short horizon is tail-dominated rather than quadrature-dominated",
        horizon_rows[1].versus_infinite > 5 * horizon_rows[1].versus_horizon,
        @sprintf("infinite %.3e, horizon %.3e", horizon_rows[1].versus_infinite, horizon_rows[1].versus_horizon),
    )
    record!(
        checks,
        "long horizon is quadrature-dominated",
        abs(horizon_rows[end].versus_infinite - horizon_rows[end].versus_horizon) / horizon_rows[end].versus_horizon < 1.0e-2,
    )
    record!(
        checks,
        "fixed-step finite-horizon Hankel error stays at the quadrature floor",
        all(row -> 1.0e-2 < row.versus_horizon < 1.0e-1, horizon_rows),
    )

    println("\nResolved comparison against baltrunc")
    hero = quadrature[end]
    println("Resolution: T = 16, dt = $(hero.dt), nodes = $(hero.nt), order r = 2 and r = 3 and r = 4")
    println("At this mesh the relative Hankel errors are")
    print_vector(" ", hero.hsv_error)
    for r in (2, 3, 4)
        bpod = leading_reduction(hero.model, r)
        bt = r == 4 ? classical : baltrunc(A, B, C, D; n = r, atol = 0.0, rtol = 0.0)
        cosines = mode_cosines(bt.Tr, bpod.Tr)
        paired = all(j -> argmax(abs.(cosines[j, :])) == j && abs(cosines[j, j]) > 0.5, 1:r)
        aligned, signs = align_signature(bt.Tr, bpod)
        angles_T = principal_angles(bpod.Tr, bt.Tr)
        angles_S = principal_angles(bpod.S', bt.S')
        projector_error = relative_error(bpod.Tr * bpod.S, bt.Tr * bt.S)
        h2_bpod = h2_difference(A, B, C, bpod.A, bpod.B, bpod.C)
        h2_bt = h2_difference(A, B, C, bt.A, bt.B, bt.C)
        poles_bpod = eigvals(bpod.A)
        poles_bt = eigvals(bt.A)
        println("\n--- reduced order r = $r")
        if r == size(A, 1)
            println("Principal angles at full order are zero for any invertible basis of the state space.")
            println("The coordinate and Hankel-value comparisons are the informative ones.")
        end
        @printf("biorthogonality:             %.3e\n", opnorm(bpod.S * bpod.Tr - I(r)))
        @printf("mode pairing accepted:       %s\n", paired)
        @printf("signature applied to BPOD:   %s\n", signs)
        print_vector("right principal angles: ", angles_T)
        print_vector("left principal angles:  ", angles_S)
        @printf("oblique projector error:     %.3e\n", projector_error)
        @printf("relative A_r / B_r / C_r:    %.3e / %.3e / %.3e\n",
            relative_error(aligned.A, bt.A), relative_error(aligned.B, bt.B), relative_error(aligned.C, bt.C))
        @printf("feedthrough discrepancy:     %.3e\n", maximum(abs, bpod.D - bt.D))
        println("BPOD poles: ", poles_bpod)
        println("baltrunc poles: ", poles_bt)
        @printf("pole-set distance:           %.3e\n", pole_set_distance(poles_bpod, poles_bt))
        @printf("H2 error BPOD / baltrunc:    %.6e / %.6e\n", h2_bpod, h2_bt)
        @printf("spectral abscissa of A_r:    %.3e\n", maximum(real, poles_bpod))
        if r == 2
            record!(checks, "leading modes are paired with baltrunc", paired)
            record!(checks, "order-2 biorthogonality", opnorm(bpod.S * bpod.Tr - I(2)) < 1.0e-8)
            record!(checks, "leading two Hankel values are resolved", maximum(hero.hsv_error[1:2]) < 1.0e-4)
            record!(checks, "order-2 principal angles agree", maximum(vcat(angles_T, angles_S)) < 5.0e-4)
            record!(
                checks,
                "order-2 reduced matrices agree after signature alignment",
                max(
                    relative_error(aligned.A, bt.A),
                    relative_error(aligned.B, bt.B),
                    relative_error(aligned.C, bt.C),
                ) < 5.0e-4,
            )
            record!(checks, "feedthrough is copied unchanged", bpod.D == D && bt.D == D)
            record!(checks, "order-2 BPOD model is Hurwitz", maximum(real, poles_bpod) < 0)
            record!(
                checks,
                "order-2 H2 error agrees with baltrunc",
                abs(h2_bpod - h2_bt) / h2_bt < 1.0e-3,
                @sprintf("%.3e versus %.3e", h2_bpod, h2_bt),
            )
            record!(
                checks,
                "order-2 poles agree with baltrunc and are not the slow plant poles",
                pole_set_distance(poles_bpod, poles_bt) < 1.0e-3 &&
                    pole_set_distance(poles_bt, ComplexF64[-1.0, -2.0]) > 0.2,
            )
            ω = frequency_samples()
            plant_peak = maximum(opnorm.(transfer.([A], [B], [C], [D], ω)))
            peak_bpod = peak_gain_error(A, B, C, D, bpod.A, bpod.B, bpod.C, bpod.D, ω)
            peak_bt = peak_gain_error(A, B, C, D, bt.A, bt.B, bt.C, bt.D, ω)
            peak_between = peak_gain_error(bt.A, bt.B, bt.C, bt.D, bpod.A, bpod.B, bpod.C, bpod.D, ω)
            bound = 2 * sum(classical.hsv[3:4])
            step_times = collect(0:0.05:30)
            step_errors = Float64[]
            for u in ([1.0, 0.0], [0.0, 1.0], [0.7, -0.4])
                full_step = step_output(A, B, C, D, u, step_times)
                bpod_step = step_output(bpod.A, bpod.B, bpod.C, bpod.D, u, step_times)
                bt_step = step_output(bt.A, bt.B, bt.C, bt.D, u, step_times)
                push!(step_errors, relative_error(bpod_step, bt_step))
                @printf(
                    "step u = %-16s  BPOD vs baltrunc %.3e, BPOD vs plant %.3e, baltrunc vs plant %.3e\n",
                    string(u), relative_error(bpod_step, bt_step), relative_error(bpod_step, full_step), relative_error(bt_step, full_step),
                )
            end
            sim_dt = 0.01
            sim_times = collect(0:sim_dt:30)
            U = hcat(map(t -> [sin(0.7 * t), cos(1.3 * t)], sim_times[1:(end - 1)])...)
            full_forced = zoh_output(A, B, C, D, U, sim_dt)
            bpod_forced = zoh_output(bpod.A, bpod.B, bpod.C, bpod.D, U, sim_dt)
            bt_forced = zoh_output(bt.A, bt.B, bt.C, bt.D, U, sim_dt)
            @printf("sampled peak gain, BPOD vs plant:    %.6e\n", peak_bpod)
            @printf("sampled peak gain, baltrunc vs plant: %.6e\n", peak_bt)
            @printf("sampled peak gain, BPOD vs baltrunc: %.6e\n", peak_between)
            @printf("plant peak gain on the same grid:    %.6e\n", plant_peak)
            @printf("Enns bound 2(σ3+σ4) for exact truncation: %.6e\n", bound)
            @printf("two-tone ZOH, BPOD vs baltrunc:      %.3e\n", relative_error(bpod_forced, bt_forced))
            @printf("two-tone ZOH, BPOD vs plant:         %.3e\n", relative_error(bpod_forced, full_forced))
            @printf("two-tone ZOH, baltrunc vs plant:     %.3e\n", relative_error(bt_forced, full_forced))
            dc_plant = -C * (A \ B) + D
            dc_bpod = -bpod.C * (bpod.A \ bpod.B) + bpod.D
            dc_bt = -bt.C * (bt.A \ bt.B) + bt.D
            @printf("DC gain, BPOD vs baltrunc:           %.3e\n", relative_error(dc_bpod, dc_bt))
            @printf("DC gain, baltrunc vs plant:          %.3e\n", relative_error(dc_bt, dc_plant))
            record!(checks, "sampled frequency responses of the order-2 models agree", peak_between / plant_peak < 1.0e-3)
            record!(checks, "step responses of the order-2 models agree", maximum(step_errors) < 1.0e-3)
            record!(checks, "forced two-tone responses of the order-2 models agree", relative_error(bpod_forced, bt_forced) < 1.0e-3)
            record!(
                checks,
                "exact baltrunc respects the sampled Enns bound on this frequency grid",
                peak_bt <= bound * (1 + 1.0e-6),
                @sprintf("peak %.3e, bound %.3e", peak_bt, bound),
            )
        elseif r == 4
            record!(
                checks,
                "resolved full-order model is a similarity of the plant",
                pole_set_distance(poles_bpod, spectrum) < 1.0e-8 && h2_bpod < 1.0e-5,
            )
            record!(
                checks,
                "resolved full-order coordinates improved over the coarse grid",
                relative_error(aligned.A, bt.A) < 0.02 && relative_error(aligned.A, bt.A) < coarse_Ar / 10,
                @sprintf("%.3e", relative_error(aligned.A, bt.A)),
            )
        end
    end

    println("\nModal truncation of the same plant, kept poles nearest the imaginary axis")
    println("Balanced poles are not required to be plant eigenvalues. This block records that distinction.")
    orthogonal_modal = modal_galerkin(A, B, C, D, 2)
    biorthogonal = biorthogonal_modal(A, B, C, D, 2)
    println("Orthogonal Galerkin poles: ", eigvals(orthogonal_modal.A))
    println("Biorthogonal modal poles:  ", eigvals(biorthogonal.A))
    modal_h2 = h2_difference(A, B, C, orthogonal_modal.A, orthogonal_modal.B, orthogonal_modal.C)
    biorthogonal_h2 = h2_difference(A, B, C, biorthogonal.A, biorthogonal.B, biorthogonal.C)
    @printf("H2 error, orthogonal modal:    %.6e\n", modal_h2)
    @printf("H2 error, biorthogonal modal:  %.6e\n", biorthogonal_h2)
    @printf("H2 error, baltrunc order 2:    %.6e\n", classical_r2_h2)
    record!(
        checks,
        "biorthogonal modal truncation retains the slow plant poles",
        pole_set_distance(eigvals(biorthogonal.A), ComplexF64[-1.0, -2.0]) < 1.0e-8,
    )
    record!(
        checks,
        "balanced truncation has smaller H2 error than slow modal truncation",
        modal_h2 > classical_r2_h2 && biorthogonal_h2 > classical_r2_h2,
        @sprintf("modal %.3e, spectral %.3e, balanced %.3e", modal_h2, biorthogonal_h2, classical_r2_h2),
    )

    println("\nChecks")
    for item in checks
        status = item.ok ? "PASS" : "FAIL"
        suffix = isempty(item.detail) ? "" : "  [" * item.detail * "]"
        println(status, "  ", item.name, suffix)
    end
    failed = filter(item -> !item.ok, checks)
    if !isempty(failed)
        message = join((item.name for item in failed), "; ")
        error("BPOD reference checks failed: " * message)
    end
    println("All $(length(checks)) checks passed.")
    return nothing
end

main()
