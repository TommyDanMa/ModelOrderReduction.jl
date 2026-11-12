# Gate 3: balanced POD reference experiment

These notes describe `scratch/bpod_reference.jl`. That file is a standalone numerical experiment. It is not part of the package API, and it does not change anything under `src/`. The numbers below come from a completed run of that script: the process exited 0 and printed `All 41 checks passed.`

Read the notes together with the script. The script is the definition of the checks; these notes record what that run measured and how to read it.

## What the experiment implements, and why

Issue [#11](https://github.com/SciML/ModelOrderReduction.jl/issues/11) asks for balanced proper orthogonal decomposition (balanced POD). The classical algorithm already in the package is `baltrunc` in `src/balanced_truncation.jl`. It balances a continuous-time linear system by solving two Lyapunov equations and then truncating small Hankel singular values. Balanced POD approximates that same calculation from impulse-response snapshots, which is the form that can later be used when the state dimension is too large for the Lyapunov solves.

The experiment implements the snapshot method of Rowley, *International Journal of Bifurcation and Chaos* 15 (2005), equations (19)–(24):

```text
x_j(t) = exp(A t) b_j
z_i(t) = exp(A' t) c_i
X columns: sqrt(δ_k) * x_j(t_k)
Y columns: sqrt(δ_k) * z_i(t_k)
Y' X = U Σ V'
T_r = X V_r Σ_r^{-1/2}
S   = Σ_r^{-1/2} U_r' Y'
A_r = S A T_r,  B_r = S B,  C_r = C T_r,  D_r = D
```

The weights `δ_k` are the composite trapezoidal weights on the time nodes. Columns follow Rowley's equation (20): all times for input channel 1, then all times for input channel 2, and the same ordering for the two output channels of the adjoint.

The comparison target is plain `baltrunc` (`residual = false`). Feedthrough `D` is copied unchanged. The experiment uses this method because it is the algorithm a first implementation would ship. Two neighboring papers are deliberately left out, and the run is designed so their absence is visible in the numbers:

- Willcox and Peraire, *AIAA Journal* 40 (2002), first compress each Gramian with its own POD and only then balance. This experiment keeps every snapshot until the single interaction SVD.
- Tu and Rowley, *Journal of Computational Physics* 231 (2012), append analytic tail snapshots after a short simulation. This experiment stops the integral at a finite horizon `T`. The horizon sweep measures the tail that paper would restore.

Snapshots are samples of the matrix exponential, computed by `LinearAlgebra.exp`. They are not the output of an ODE solver. A later data-based implementation would hand the same weighted matrices `X` and `Y` to the same SVD. Using the exponential isolates horizon error and quadrature error from integrator tolerance.

The state inner product is the ordinary dot product, so the adjoint generator is `A'` and each adjoint impulse starts at a column of `C'`. Both outputs are simulated. There is no output projection.

## The plant

The system is hardcoded as `A_PLANT`, `B_PLANT`, `C_PLANT`, and `D_PLANT`.

```text
A = [-1  20   0   0
      0  -2  15   0
      0   0  -3  10
      0   0   0  -6]

B = [1.0   0.0
     0.4   1.0
     0.0   0.5
     0.2  -0.3]

C = [1.0  0.2  0.0   0.4
     0.0  1.0  0.3  -0.2]

D = [0.15  -0.02
     0.01   0.08]
```

It has 4 states, 2 inputs, and 2 outputs. The eigenvalues are `-6, -3, -2, -1`, so `A` is Hurwitz and the infinite-horizon Gramians exist. `A` is far from normal: `||A A' - A' A||_2 = 400.695`, the numerical abscissa is `+10.915`, and `||exp(A t)||_2` reaches `42.483` at `t = 1.24`. The eigenvector condition number is `2032`. Those facts matter because balanced truncation and modal truncation then answer different questions. The Hankel singular values are separated,

```text
15.057393552644522
 3.949552648039867
 0.9642919891801569
 0.056473131603322645
```

so each balanced mode is a simple direction and can be paired with the corresponding `baltrunc` mode up to a sign.

`D` does not enter the Gramians. It is there so the experiment can check that the reduced feedthrough is a copy, not a projection.

## The main operations, in order

### 1. Impulse responses

The continuous-time system is

```text
x'(t) = A x(t) + B u(t)
y(t)  = C x(t) + D u(t)
```

A unit impulse in input `j` is equivalent to starting at `x(0) = b_j`, the `j`th column of `B`, and then letting the state decay with no further input. The decay is exactly `exp(A t) b_j`.

The adjoint system runs the same idea backward through the outputs. An impulse in output `i` starts at the `i`th column of `C'` and decays under `A'`. Its state `z_i(t) = exp(A' t) c_i` records how strongly a state direction is seen by that output.

### 2. Gramians

The controllability Gramian adds up every primal impulse, over all inputs and all positive times:

```text
P_c = ∫_0^∞ (exp(A t) B) (exp(A t) B)' dt
```

The observability Gramian does the same for the adjoint impulses. Large eigenvalues of `P_c` are directions the inputs can easily reach. Large eigenvalues of `P_o` are directions the outputs can easily see. `baltrunc` obtains both matrices from the Lyapunov equations

```text
A P_c + P_c A' + B B' = 0
A' P_o + P_o A + C' C = 0
```

using `LinearAlgebra.lyap`. On this plant those residuals, scaled by `||B B'||` and `||C' C||`, were `2.211e-15` and `2.101e-12`.

### 3. Hankel singular values

The Hankel singular values are the singular values of `L_o' L_c`, where `P_c = L_c L_c'` and `P_o = L_o L_o'`. They rank directions that are simultaneously reachable and observable. Exact balanced truncation drops the small ones. For plain truncation, the `H∞` error is at most twice the sum of the discarded values (the Enns bound). On this plant, keeping two modes gives the bound `2(σ_3 + σ_4) = 2.041530`.

### 4. Trapezoidal weights

The integrals are replaced by a finite sum on nodes `0 = t_1 < ... < t_N = T`. The composite trapezoidal weight is half the neighboring gap at each end and the average of the neighboring gaps in the interior. On a uniform step `Δt` that is `Δt/2` at the endpoints and `Δt` in the interior. The weights are positive on a strictly increasing grid, they sum exactly to `T - t_1` by telescoping, and they integrate any straight line exactly.

Each snapshot is stored already multiplied by `sqrt(δ_k)`. The sum of outer products of those columns is then the quadrature of the Gramian. The square root is applied once, inside the snapshot. Applying the weight again when forming `X X'` would square it.

The scalar check integrates `exp(-t)` on `[0, 16]`. At `Δt = 1/16` the trapezoidal error was `3.255e-4`, which matched the Euler–Maclaurin term `h^2/12 (f'(T) - f'(0))` to a relative difference of about `6.5e-5`. Halving the step produced observed order `1.999930`.

### 5. The interaction SVD

Forming `P_c ≈ X X'` and `P_o ≈ Y Y'` is valid, but those products square the snapshot information. Rowley's method avoids that. The nonzero singular values of the smaller interaction matrix `Y' X` are exactly the Hankel singular values of the quadrature Gramians. The same SVD supplies the projectors:

```text
T_r rebuilds a full state from r coefficients:   x ≈ T_r x_r
S   compresses a full state to r coefficients:   x_r = S x
```

`S T_r = I` means that compressing a reconstructed state returns the coefficients. For any positive retained singular values this identity is a property of the SVD, including on a bad time grid. It checks linear algebra, not the quality of the integral.

The reduced model is the original system seen through those two maps. Because each mode is scaled by its own `σ_j^{-1/2}`, the order-`r` model is the leading `r × r` block of a model built at a higher order from the same SVD. `leading_reduction` uses that fact.

One sign per mode is arbitrary: flipping column `j` of `T_r` and row `j` of `S` together leaves `S T_r`, the transfer function, and the Hankel values unchanged, while it changes the sign pattern of `A_r`, `B_r`, and `C_r`. `align_signature` fixes those signs before matrix entries are compared. Another computer can print a different sign pattern and still pass.

### 6. What a full-order factor means

If all four positive modes are kept, `T_r` is square and `S` is its inverse. Then `A_r = S A T_r` is a similarity of `A`. The transfer function matches the plant even when the time grid is too coarse for the Hankel values to be right. Full-order input-output agreement is therefore another check that can pass for a bad quadrature. The coarse-grid section exists to show both traps on one mesh.

### 7. Two different errors

The exact tail of the controllability integral has the closed form `exp(A T) P_c exp(A' T)`. The exact integral from `0` to `T` is `P_c` minus that tail, and likewise for `P_o` with `A` and `A'` swapped. Comparing balanced POD with those finite-horizon Gramians measures quadrature error. Comparing it with the infinite-horizon Hankel values measures quadrature error and the omitted tail together.

## Functions in `scratch/bpod_reference.jl`

The file calls `main()` on load. Helpers are local to that script.

| Name | Purpose |
|---|---|
| `A_PLANT`, `B_PLANT`, `C_PLANT`, `D_PLANT` | The fixed 4-state plant. |
| `BPODModel` | Reduced `A, B, C, D`, Hankel values, right factor `Tr`, left factor `S`, and the quadrature Gramians `Pc`, `Po`. |
| `Check` | One named pass/fail record collected by `main`. |
| `uniform_nodes` | Nodes from `0` to `T` that hit both endpoints. Throws if `Δt` does not divide `T`. |
| `trapezoidal_weights` | Composite trapezoidal weights for a strictly increasing node set. |
| `impulse_factor` | Weighted snapshot matrix. Primal call uses `(A, B)`; adjoint call uses `(A', C')`. |
| `gramian_from_impulses` | The same quadrature written as an explicit sum of outer products, to confirm the square-root weight is applied once. |
| `psd_sqrt` | Factor `L` with `W ≈ L L'`. Drops eigenvalues at or below `eps * λmax`. Mirrors the clipping idea in `ModelOrderReduction._psd_sqrt`. |
| `sqrt_balance` | Square-root balanced truncation from given Gramians. Independent of `baltrunc`, used to confirm the package result. |
| `balanced_pod` | The Gate 3 algorithm: weights, both impulse factors, SVD of `Y' X`, projectors, reduced matrices. |
| `finite_horizon_gramians` | Exact Gramians of the integral stopped at `T`, using the tail formula above. |
| `relative_error` | Frobenius relative error `||got - reference|| / ||reference||`. `norm` on a matrix is Frobenius. |
| `component_relative_error` | Relative error of each Hankel value against the reference list. |
| `principal_angles` | Angles between the subspaces spanned by two mode sets, after orthonormalizing with `qr`. |
| `mode_cosines` | Cosine of the angle between every pair of individual modes. Used to test that Hankel-value order already pairs them. |
| `align_signature` | Multiplies mode `j` by `-1` when its overlap with the `baltrunc` mode is negative, and updates `A`, `B`, `C`, `Tr`, and `S` consistently. |
| `pole_set_distance` | Smallest maximum modulus mismatch over pairings of two eigenvalue lists. Order does not matter. |
| `h2_difference` | `H2` norm of the parallel difference system. Requires both generators to be Hurwitz, because an unstable difference has infinite `H2` norm. Feedthrough cancels because both models use the same `D`. |
| `h2_norm` | Plant `H2` norm from the controllability Gramian, `sqrt(tr(C P_c C'))`. |
| `transfer` | Frequency response `C (iω I - A)^{-1} B + D`. |
| `frequency_samples` | 161 log-spaced frequencies from `10^{-3}` to `10^2`, plus 81 linear samples from `0.2` to `2`. |
| `peak_gain_error` | Largest spectral norm of the difference of two frequency responses on that grid. This is a sampled lower estimate of the `H∞` distance. |
| `step_output` | Exact step response from rest, `x(t) = A^{-1}(exp(A t) - I) B u`. |
| `zoh_output` | Exact response to a piecewise-constant input, sampled at the left endpoint of each step. |
| `modal_galerkin` | Orthogonal Galerkin projection onto the `r` right eigenvectors closest to the imaginary axis. |
| `biorthogonal_modal` | Petrov–Galerkin projection onto those same eigenvalues using paired left and right eigenvectors. |
| `observed_orders` | Convergence order between successive errors when a step is refined, `log(e_i / e_{i+1}) / log(Δt_i / Δt_{i+1})`. |
| `leading_reduction` | Leading principal subsystem of a model already built at a higher order from one SVD. |
| `record!` | Append a `Check`. The pass/fail list is printed only after the whole run. |
| `print_vector` | Print a label and values in scientific notation. |
| `main` | Build the reference, run the coarse grid, the `Δt` sweep, the horizon sweep, the resolved comparison, and the modal comparison, then throw if any check failed. |

`main` uses `ModelOrderReduction.baltrunc` only as the classical reference. The balanced POD path does not call it.

## How to run it

From the repository root, with the package environment available:

```bash
julia --project=. scratch/bpod_reference.jl
```

On a fresh clone, instantiate that environment first so `ModelOrderReduction` and its Project dependencies can load. The script itself adds no packages. It uses `LinearAlgebra`, `Printf`, and `baltrunc`.

A successful run ends with

```text
All 41 checks passed.
```

and exit code 0. A failed check throws `BPOD reference checks failed: ...` after printing every check as `PASS` or `FAIL`. The run that produced these notes took about 70 seconds. Almost all of that time is the interaction SVD. At `Δt = 1/128` and `T = 16` there are 2049 nodes and two channels, so `Y' X` is `4098 × 4098`.

Expect the same pass/fail outcome on another machine. Printed mode signs can differ by the SVD convention, and residuals near roundoff can differ in the last digits. The checks compare inequalities with margins, then `align_signature` removes the sign freedom before matrix entries are judged.

## Numerical results from the passing run

### Reference balanced truncation

| Quantity | Value |
|---|---|
| Spectral abscissa | `-1` |
| Numerical abscissa | `+10.91527` |
| `||A A' - A' A||_2` | `400.6953` |
| Eigenvector condition | `2032.204` |
| Peak of `||exp(A t)||_2` on `0:0.01:10` | `42.48344` at `t = 1.24` |
| Controllability Lyapunov residual | `2.211e-15` |
| Observability Lyapunov residual | `2.101e-12` |
| Plant `H2` norm | `13.69082` |
| `baltrunc` biorthogonality `||S T - I||` | `2.709e-14` |
| Independent square-root HSV discrepancy | `0` at the printed precision |
| Independent square-root `A_r` discrepancy | `0` at the printed precision |

The Hankel singular values are the four numbers listed in the plant section. `baltrunc` was called with `n` set explicitly and with `atol = 0`, `rtol = 0`, so the package's default relative cutoff was not allowed to drop modes.

### Weight and factor checks

| Check | Measured value |
|---|---|
| Integral of `3 + 2t` on the irregular nodes `[0, 0.2, 0.5, 0.9, 1]` | absolute error `0` (exact value `4`) |
| Sum of weights on `[0, 16]` | error `0` |
| `∫_0^16 exp(-t) dt` at `Δt = 1/16` | error `3.255e-4`, Euler–Maclaurin prediction `3.255e-4`, order on halving `1.999930` |
| `X X'` against the explicit primal sum, `T = 2`, `Δt = 1/2` | relative error `1.879e-16` |
| `Y Y'` against the explicit adjoint sum | relative error `1.496e-16` |

### Coarse grid: `T = 2`, `Δt = 1`

| Quantity | Value |
|---|---|
| `||S T_r - I||` at order 4 | `1.615e-14` |
| `||S T_r - I||` at order 2 | `9.601e-16` |
| Largest relative Hankel error | `0.3429` |
| Coarse Hankel values | `12.88130`, `5.303657`, `1.168920`, `0.05616027` |
| Order-4 `H2` distance to the plant | `2.645e-7` |
| Order-4 pole distance to the plant | `5.014e-13` |
| Order-4 relative `A_r` error against `baltrunc` | `1.005` |
| Order-2 `H2` error, balanced POD / `baltrunc` | `4.339650 / 2.548426` |
| Order-2 oblique-projector error | `0.9076` |

Biorthogonality and the full-order transfer function are satisfied. The Hankel values and the order-2 input-output error are not.

### Quadrature at fixed `T = 16`

The controllability tail at this horizon is `1.655e-13` relative to `||P_c||`, so the table is a quadrature study. `Pc rel` and `Po rel` are Frobenius relative errors of the quadrature Gramians. `ordPc` is the observed order of `Pc rel` from the previous row. The last four columns are relative Hankel errors.

| `Δt` | nodes | `Pc` rel | `Po` rel | order of `Pc` | `σ1` | `σ2` | `σ3` | `σ4` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 33 | `7.312e-3` | `1.277e-3` |  | `1.184e-2` | `1.253e-1` | `7.092e-1` | `3.825e-1` |
| 0.25 | 65 | `1.295e-3` | `5.788e-5` | 2.498 | `2.990e-3` | `2.972e-2` | `2.220e-1` | `9.539e-1` |
| 0.125 | 129 | `2.947e-4` | `1.671e-5` | 2.135 | `7.506e-4` | `7.304e-3` | `6.091e-2` | `4.520e-1` |
| 0.0625 | 257 | `7.276e-5` | `4.418e-6` | 2.018 | `1.879e-4` | `1.817e-3` | `1.574e-2` | `1.334e-1` |
| 0.03125 | 513 | `1.815e-5` | `1.120e-6` | 2.003 | `4.698e-5` | `4.538e-4` | `3.971e-3` | `3.478e-2` |
| 0.015625 | 1025 | `4.535e-6` | `2.809e-7` | 2.001 | `1.175e-5` | `1.134e-4` | `9.951e-4` | `8.786e-3` |
| 0.0078125 | 2049 | `1.134e-6` | `7.029e-8` | 2.000 | `2.937e-6` | `2.835e-5` | `2.489e-4` | `2.202e-3` |

Observed Hankel orders on the six successive halvings:

| Mode | orders |
|---|---|
| 1 | 1.986, 1.994, 1.998, 2.000, 2.000, 2.000 |
| 2 | 2.076, 2.025, 2.007, 2.002, 2.000, 2.000 |
| 3 | 1.676, 1.866, 1.952, 1.987, 1.997, 1.999 |
| 4 | -1.318, 1.078, 1.761, 1.939, 1.985, 1.996 |

The order-two checks use only the last three halvings, those that start at `Δt = 1/16`. All twelve of those orders lie in `[1.939, 2.002]`. The controllability orders on those same three halvings are `2.003`, `2.001`, and `2.000`. From `Δt = 1/8` downward, every mode's relative error decreases. The fourth mode is not in that regime on the coarsest step: its relative error rises from `0.383` at `Δt = 1/2` to `0.954` at `Δt = 1/4`.

At `Δt = 1/64` (`quadrature[end - 1]`), the Hankel values from `svd(Y' X)` and from the square-root method on the empirical Gramians `X X'` and `Y Y'` differ by a maximum relative amount `1.714e-12`.

### Horizon at fixed `Δt = 1/32`

| `T` | relative `P_c` tail | max relative Hankel error vs infinite horizon | max relative Hankel error vs exact horizon | relative error of `σ1` vs infinite horizon |
|---:|---:|---:|---:|---:|
| 1 | `6.109e-1` | `5.330e-1` | `4.915e-2` | `3.628e-1` |
| 2 | `1.604e-1` | `1.410e-1` | `3.726e-2` | `6.672e-2` |
| 4 | `4.153e-3` | `2.990e-2` | `3.485e-2` | `1.487e-3` |
| 8 | `1.469e-6` | `3.477e-2` | `3.478e-2` | `4.748e-5` |
| 12 | `4.934e-10` | `3.478e-2` | `3.478e-2` | `4.698e-5` |
| 16 | `1.655e-13` | `3.478e-2` | `3.478e-2` | `4.698e-5` |

At `T = 1` the infinite-horizon error is about ten times the finite-horizon error, so the tail dominates. At `T = 16` the two errors agree. Across every horizon, the finite-horizon Hankel error stays between `0.01` and `0.1`. That floor matches the `Δt = 1/32` row of the quadrature table, whose fourth-mode relative error is `0.03478`.

### Resolved grid: `T = 16`, `Δt = 1/128`, 2049 nodes

Relative Hankel errors: `2.937e-6`, `2.835e-5`, `2.489e-4`, `2.202e-3`.

Order 2, after the sign pattern `[-1, -1]`:

| Quantity | Value |
|---|---|
| Biorthogonality | `1.050e-15` |
| Right principal angles | `4.47e-8`, `2.47e-5` |
| Left principal angles | `1.12e-7`, `3.30e-5` |
| Oblique projector `T_r S` | relative error `5.847e-5` |
| Relative `A_r`, `B_r`, `C_r` | `1.649e-5`, `2.452e-5`, `8.398e-6` |
| Feedthrough discrepancy | `0` |
| BPOD poles | `-0.72462625 ± 0.66872503 i` |
| `baltrunc` poles | `-0.72464132 ± 0.66871841 i` |
| Pole-set distance | `1.646e-5` |
| `H2` error, BPOD / `baltrunc` | `2.548461 / 2.548426` |
| Spectral abscissa of `A_r` | `-0.7246` |

The distance from the `baltrunc` poles to the slow plant poles `{-1, -2}` is greater than `0.2`. The reduced poles are the balanced poles.

Input-output checks at order 2:

| Test | BPOD vs `baltrunc` | BPOD vs plant | `baltrunc` vs plant |
|---|---:|---:|---:|
| Step `[1, 0]` on `0:0.05:30` | `2.678e-6` | `7.541e-2` | `7.541e-2` |
| Step `[0, 1]` | `7.789e-6` | `4.906e-2` | `4.905e-2` |
| Step `[0.7, -0.4]` | `3.099e-6` | `1.174e-1` | `1.174e-1` |
| Sampled peak gain | `3.029e-4` | `1.837857` | `1.837619` |
| Two-tone zero-order hold | `1.537e-5` | `0.1115` | `0.1115` |
| DC gain | `3.892e-6` |  | `7.552e-2` |

The plant peak gain on the same frequency grid is `24.25494`. The sampled `baltrunc` peak error `1.837619` lies under the Enns bound `2.041530`. The two-tone input is `[sin(0.7 t), cos(1.3 t)]`, held constant on steps of `0.01` over `[0, 30]`.

Order 3, sign pattern `[-1, -1, 1]`:

| Quantity | Value |
|---|---|
| Third-mode principal angles, right / left | `5.679e-4` / `3.393e-4` |
| Oblique projector | relative error `8.506e-4` |
| Relative `A_r`, `B_r`, `C_r` | `2.218e-4`, `2.852e-5`, `9.644e-5` |
| Pole-set distance | `5.776e-4` |
| `H2` error, BPOD / `baltrunc` | `0.2254221 / 0.2253855` |
| BPOD poles | `-1.82559944 ± 1.10529698 i`, `-0.87151293` |
| `baltrunc` poles | `-1.82600425 ± 1.104884996 i`, `-0.87164241` |

Order 4, sign pattern `[-1, -1, 1, -1]`:

| Quantity | Value |
|---|---|
| Relative `A_r`, `B_r`, `C_r` | `2.045e-4`, `9.350e-5`, `1.751e-4` |
| Pole-set distance to `baltrunc`, and to the plant | `3.402e-13` |
| `H2` distance to the plant, BPOD / `baltrunc` | `2.147e-7` / `2.899e-7` |

At full order the principal angles are essentially zero because both bases span all of `R^4`. The informative full-order comparisons are the Hankel values, the coordinate matrices, and the similarity residuals.

### Modal truncation of the slow subspace

Both the orthogonal Galerkin model and the biorthogonal modal model retain the poles `-2` and `-1`. Their `H2` errors against the plant are `9.382556` and `45.88107`. The order-2 `baltrunc` error is `2.548426`.

## The 41 passing checks

The script records these in order. Each one passed.

1. `A` is Hurwitz. Spectral abscissa `-1.000e+00`.
2. `A` is non-normal. `||A A' - A' A||_2 = 4.007e+02`.
3. Transient growth exceeds the spectral envelope. Numerical abscissa `1.092e+01`, peak gain `4.248e+01`.
4. Lyapunov residuals are roundoff, both below `1e-10` after scaling.
5. Independent square-root Hankel values match `baltrunc` within `1e-9`.
6. Independent square-root realization matches `baltrunc` within `1e-8` in `A_r` after signature alignment.
7. Trapezoidal rule integrates the affine test exactly, absolute error below `1e-12`.
8. Trapezoidal weights telescope to the interval length, error below `1e-12`.
9. Scalar exponential quadrature has observed order in `[1.99, 2.01]` and matches the Euler–Maclaurin term within `0.1%`. This run: order `1.999930`, error `3.255e-4`, prediction `3.255e-4`.
10. Snapshot factors reproduce the weighted impulse sums within relative error `1e-10`.
11. Coarse-grid biorthogonality at order 4 is below `1e-8`.
12. Coarse-grid Hankel values are still inaccurate: largest relative error above `0.2`. This run: `0.343`.
13. Full-order coarse model reproduces the plant poles within `1e-8`.
14. Full-order coarse model has `H2` distance to the plant below `1e-5`.
15. Coarse balancing coordinates differ from `baltrunc`: relative `A_r` error above `0.5`. This run: `1.005`.
16. Coarse order-2 `H2` error differs from `baltrunc` by more than 20%. This run: `4.340` against `2.548`.
17. Controllability Gramian orders on the three finest halvings lie in `[1.8, 2.2]`.
18. All four Hankel values have orders in `[1.8, 2.2]` on those same three halvings.
19. Per-mode Hankel errors decrease for every `Δt <= 1/8`.
20. Interaction SVD and empirical square-root Hankel values agree within `1e-8`. This run: `1.714e-12`.
21. The controllability tail decreases with `T` and is below `1e-10` at `T = 16`.
22. At `T = 1`, the infinite-horizon Hankel error is more than five times the finite-horizon error.
23. At `T = 16`, infinite-horizon and finite-horizon Hankel errors agree within 1%.
24. Every finite-horizon Hankel error at `Δt = 1/32` lies strictly between `0.01` and `0.1`.
25. The two leading resolved modes pair with `baltrunc`: each reference mode's largest absolute cosine is its partner, and that cosine exceeds `0.5`.
26. Order-2 biorthogonality on the resolved grid is below `1e-8`.
27. The two leading relative Hankel errors are below `1e-4`.
28. Order-2 principal angles are below `5e-4` radians.
29. Order-2 `A_r`, `B_r`, and `C_r` each agree with `baltrunc` within relative error `5e-4` after signature alignment.
30. Reduced and reference feedthrough equal `D` exactly.
31. Order-2 balanced POD is Hurwitz.
32. Order-2 `H2` errors agree within a relative difference of `0.1%`.
33. Order-2 poles match `baltrunc` within `1e-3` and remain more than `0.2` away from `{-1, -2}`.
34. Sampled peak gain between the order-2 models is below `0.1%` of the plant peak.
35. All three step responses agree within relative Frobenius error `1e-3`.
36. The two-tone forced responses agree within relative error `1e-3`.
37. The sampled `baltrunc` peak error does not exceed the Enns bound, up to a relative slack of `1e-6`. This run: peak `1.838`, bound `2.042`.
38. Resolved full-order poles match the plant within `1e-8` and the `H2` distance is below `1e-5`.
39. Resolved full-order `A_r` is within `0.02` of `baltrunc` and at least ten times closer than the coarse `A_r`. This run: `2.045e-4`.
40. Biorthogonal modal truncation retains `{-1, -2}` within `1e-8`.
41. Both slow modal models have larger plant `H2` error than order-2 `baltrunc`. This run: modal `9.383`, spectral `45.88`, balanced `2.548`.

## What this run establishes

On this plant, with exact exponential samples and the Euclidean inner product, the Rowley projectors built from trapezoidal weights converge to plain balanced truncation.

- The weight code is the composite trapezoidal rule: exact on affine functions, telescoping, and second order on `exp(-t)`, with the predicted `h^2/12` constant.
- The snapshot matrix is the factor of that quadrature. Its outer product matches the explicit weighted sum of impulses.
- The interaction SVD and square-root balancing of the resulting empirical Gramians produce the same Hankel values, here to `1.7e-12`.
- As `Δt` is halved at a horizon long enough that the tail is roundoff, the Gramian error and all four Hankel errors settle at order two. The fourth value needs the finer meshes before that rate appears.
- As `T` grows at fixed `Δt`, the remaining Hankel error changes from a tail error into the quadrature error already measured at that `Δt`.
- Once the first two Hankel values are resolved, the order-2 subspaces, reduced matrices, poles, `H2` error, steps, forced response, DC gain, and sampled frequency response agree with `baltrunc`. The agreement with the plant is the agreement that exact order-2 truncation itself has. Both models sit several percent away from the plant, and they sit there together.
- The order-2 balanced poles are a complex pair near `-0.725 ± 0.669i`. Slow modal truncation keeps `-1` and `-2` and has a larger `H2` error, `9.38` or `45.9` against `2.55`.
- `baltrunc` on this example matches an independent implementation of the same square-root method. The package reference used for the comparison is consistent.

The coarse grid is part of the result. A method that only checked `S T_r ≈ I`, matrix sizes, or full-order input-output error would accept a quadrature whose Hankel values are still wrong by tens of percent.

## What this run leaves open

The checks are for one 4-state plant, one family of uniform grids, plain truncation, and the Euclidean inner product.

- They do not exercise an ODE integrator, a mass-matrix inner product, output projection, residualization, or the Tu–Rowley tail correction.
- They do not exercise Willcox–Peraire's intermediate POD compression of each Gramian.
- The Enns comparison is a sampled peak of exact `baltrunc` on one frequency grid. The script does not compute a certified `H∞` norm, and the quadrature model is not claimed to satisfy the bound at every mesh.
- Order two is demonstrated for this impulse response. The negative order of `σ_4` between `Δt = 1/2` and `Δt = 1/4` shows that a coarse mesh can leave the asymptotic regime.
- At the finest mesh the fourth relative Hankel error is still `2.2e-3`. The corresponding full-order `A_r` still differs from `baltrunc` by `2.0e-4`. Further refinement would be expected to continue at order two; this run stops at `Δt = 1/128`.
- Full-order principal angles are uninformative because the whole state space is being compared.
- Modal `H2` numbers show that, on this plant, balanced truncation tracks the input-output map more closely than truncation to the slow invariant subspace. That is a comparison for these projectors, not a general theorem.
- Passing checks use fixed thresholds in `main`. A future edit to the plant or the grids can fail a threshold that was appropriate here. The printed tables are the evidence; the booleans are the gates.

## Limits, and the recommended next implementation step

The experiment is ready to be used as the numerical specification of a first API. It is not itself that API. `scratch/bpod_reference.jl` is outside the test suite, depends on a state dimension of 4 for the exact Lyapunov reference, and spends its time on a dense interaction SVD that would be the wrong dominant cost for a large state.

The next implementation step is a single function for an explicit real continuous-time `(A, B, C, D)`:

- Take the time nodes, or a horizon and a dividing step, as an explicit argument.
- Build primal and adjoint impulse factors with `trapezoidal_weights`, compute `svd(Y' X)`, and form `T_r`, `S`, `A_r`, `B_r`, `C_r`, and `D_r = D` exactly as `balanced_pod` does.
- Return plain truncation in the existing `BalancedTruncation` layout (`A`, `B`, `C`, `D`, `hsv`, `Tr`, `S`), for a requested order.
- Port two tests from this script into the package tests: the coarse `T = 2`, `Δt = 1` case, where biorthogonality holds and the Hankel values must still be far from `baltrunc`, and the resolved `T = 16`, `Δt = 1/128` order-2 case, where the reduced matrices, poles, and `H2` error must agree with `baltrunc` at the tolerances already used here.

Leave analytic tails, output projection, residualization, ODE-generated snapshots, and a ModelingToolkit method for a later change. Those are different algorithms. This experiment does not validate them.
