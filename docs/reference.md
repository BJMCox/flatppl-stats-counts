# Count statistics reference

FlatPPL compatibility and library version: **0.1**. All arguments are scalar.
Means and `signal` are expected counts, not count rates per unit time.
Observed counts are nonnegative integers. Every argument is explicit; the library
introduces no inferred parameter or prior on the caller's behalf.

| Function | Result |
|---|---|
| [`onoff(signal, background, off_on_ratio)`](#onoff) | Joint ON/OFF Poisson law |
| [`onoff_with_leakage(signal, control_signal, background, off_on_ratio)`](#onoff_with_leakage) | Joint law with an explicit OFF signal mean |
| [`onoff_background_mle(on_counts, off_counts, signal, off_on_ratio)`](#onoff_background_mle) | Conditional ON-background maximum likelihood estimate |
| [`onoff_profile_loglikelihood(on_counts, off_counts, signal, off_on_ratio)`](#onoff_profile_loglikelihood) | Full profile log likelihood |
| [`onoff_gamma_logdensity(on_counts, off_counts, signal, off_on_ratio, shape, rate)`](#onoff_gamma_logdensity) | Exact normalized joint-count log PMF after Gamma integration |

<a id="onoff"></a>
## `onoff(signal, background, off_on_ratio)`

**Arguments.** `signal` is the expected ON source count, `background` the expected
ON background count, and `off_on_ratio` the OFF/ON background exposure ratio.

**Definition.** The returned law is over a record with fields `on` and `off`:

```math
n\sim\mathrm{Poisson}(s+b),\qquad m\sim\mathrm{Poisson}(\rho b),\qquad n\perp m\mid(s,b,\rho).
```

**Returns.** A normalized joint count law with no background prior. At fixed
parameters its log density is

```math
\ell(n,m;s,b,\rho)=n\log(s+b)-(s+b)+m\log(\rho b)-\rho b-\log(n!)-\log(m!).
```

**Domain.** `signal >= 0`, `background >= 0`, `off_on_ratio > 0`.
The event samples must be disjoint and conditionally independent.
The convention is $0\log0=0$: a zero mean gives probability one for a zero count
and zero for a positive count. Negative means are outside the model.
The commonly used ON/OFF exposure parameter is $\alpha=1/\rho$.

<a id="onoff_with_leakage"></a>
## `onoff_with_leakage(signal, control_signal, background, off_on_ratio)`

**Arguments.** As for [`onoff`](#onoff), with `control_signal` the separately
predicted expected source count in the OFF region.

**Definition.**

```math
n\sim\mathrm{Poisson}(s+b),\qquad m\sim\mathrm{Poisson}(s_{\rm off}+\rho b),\qquad n\perp m\mid(s,s_{\rm off},b,\rho).
```

**Returns.** A normalized joint law over record fields `on` and `off`. No prior
or leakage fraction is inferred or introduced.

**Domain.** `signal`, `control_signal`, and `background` are nonnegative;
`off_on_ratio > 0`. The independence and zero-mean conventions of `onoff` apply.

<a id="onoff_background_mle"></a>
## `onoff_background_mle(on_counts, off_counts, signal, off_on_ratio)`

**Arguments.** `on_counts` and `off_counts` are the observed ON and OFF counts
$n,m$; `signal` is a fixed candidate ON source mean $s$ and `off_on_ratio` is $\rho$.

**Definition.** Maximize $\ell(n,m;s,b,\rho)$ over $b\geq0$. Put
$c=1+\rho$, $A=n+m-cs$, and $D=\sqrt{A^2+4cms}$. The implementation evaluates the
positive root without subtracting nearly equal numbers:

```math
\widehat b=\begin{cases}
\max(n/c-s,0),&m=0,\\
(A+D)/(2c),&m>0,\ A\geq0,\\
2ms/(D-A),&m>0,\ A<0.
\end{cases}
```

In particular, $s=0$ gives $\widehat b=(n+m)/c$, and $n=m=0$ gives zero.

**Returns.** The conditional maximum likelihood estimate of the **ON** background
mean. It is a number, not a law or posterior mean.

**Domain.** Nonnegative integer counts, `signal >= 0`, `off_on_ratio > 0`.
This helper assumes a signal-free OFF region and independent per-bin backgrounds;
it does not profile a leakage or shared-background model. The estimate can have a
kink when its constrained optimum reaches zero. At equality the implementation
selects the zero-background branch. It does not clip an invalid physical mean.

<a id="onoff_profile_loglikelihood"></a>
## `onoff_profile_loglikelihood(on_counts, off_counts, signal, off_on_ratio)`

**Arguments.** As for [`onoff_background_mle`](#onoff_background_mle).

**Definition.**

```math
\ell_{\rm prof}(n,m;s,\rho)=\ell(n,m;s,\widehat b(s,\rho),\rho).
```

**Returns.** The full numeric log likelihood, including both $-\log(n!)$ and
$-\log(m!)$ constants. It is neither a normalized law over counts nor a Bayesian
background integral. A statistic defined as $-2\ell$ requires an explicit
factor of $-2$; comparisons with WStat also require matching data-only constants.

**Domain.** The same counts, source, ratio, and observation assumptions as
`onoff_background_mle`. Zero-count logarithms are evaluated through their Poisson
limits, including $\ell_{\rm prof}(0,0;s,\rho)=-s$ and its right derivative $-1$
at $s=0$.

<a id="onoff_gamma_logdensity"></a>
## `onoff_gamma_logdensity(on_counts, off_counts, signal, off_on_ratio, shape, rate)`

**Arguments.** Counts, source and exposure ratio as above. `shape` is $a>0$ and
`rate` is $r>0$ in the explicitly supplied background prior
$b\sim\mathrm{Gamma}(a,r)$, with mean $a/r$.

**Definition.** Integrate the joint ON/OFF Poisson likelihood over that background:

```math
p(n,m\mid s,\rho,a,r)=
\frac{e^{-s}\rho^m r^a}{n!m!\Gamma(a)}
\sum_{k=0}^{n}\binom{n}{k}s^{n-k}
\frac{\Gamma(m+a+k)}{(1+\rho+r)^{m+a+k}}.
```

At $s=0$ only $k=n$ contributes. Its value and first signal derivative are

```math
\log p(n,m\mid0)=m\log\rho+a\log r+\log\Gamma(n+m+a)
-\log(n!)-\log(m!)-\log\Gamma(a)-(n+m+a)\log(1+\rho+r),
```

```math
\left.\frac{\partial\log p}{\partial s}\right|_{s=0^+}
=\begin{cases}-1,&n=0,\\
n(1+\rho+r)/(n+m+a-1)-1,&n>0.
\end{cases}
```

Put $t=1+\rho+r$ and $u=st$. Factoring $t^{-(n+m+a)}$ out of the sum
gives a polynomial in $u$ whose coefficients contain no powers of $t$. Adjacent
Gamma/factorial ratios are accumulated as log products to avoid subtracting large
log-Gamma values when forming these coefficients. The finite sum is evaluated as this exact scaled positive polynomial for $0\leq u\leq1$,
and in log space for $u>1$. Both expressions represent the same function; the
switch at one has no mathematical discontinuity. The polynomial retains signal
boundary derivatives, rather than replacing the zero-signal value with a constant
or a linear approximation. Unused logarithms, powers, square roots and divisions
are guarded to keep inactive branches finite.

**Returns.** The normalized **joint-count** log PMF, retaining OFF evidence and
all constants. It is a named closed form for this Gamma prior, not a general
marginalization operation. Supplying an OFF-conditioned Gamma posterior and then
counting the OFF evidence again represents a different, duplicated likelihood.

**Domain.** Nonnegative integer counts, `signal >= 0`, `off_on_ratio > 0`,
`shape > 0`, `rate > 0`. `on_counts` determines the $n+1$ sum terms and must be
fixed data for compilation. Signal, ratio, shape, and rate remain differentiable.
The method costs linear work in `on_counts`; it is intended for scalar count bins.
Floating-point coefficient underflow or overflow can limit extreme counts and
parameter scales. Invalid physical inputs are outside the domain, with no clipping
or implicit regularization.
