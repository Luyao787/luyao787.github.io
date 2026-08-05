---
layout: post
title: Score Ascent View of MPPI
date: 2026-06-28
description: Deriving the MPPI update as score ascent on a noised target distribution
tags: sampling optimization score control
categories:
  - Sampling-based Optimization
related_posts: false
---

In the previous post, we derived the exponential weighted sampling update from a KL-regularized distribution update. The main result was the weighted mean update

$$
\bar{x}^{k+1}
=
\bar{x}^k
+
\sum_{i=1}^{N} w_i \epsilon_i,
$$

where

$$
x_i = \bar{x}^k + \epsilon_i,
\quad
\epsilon_i \sim \mathcal{N}(0,\Sigma^k),
$$

and

$$
w_i
=
\frac{
\exp\left(-\frac{1}{\lambda} f_i\right)
}{
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda} f_j\right)
},
\quad
f_i = f(x_i).
$$

This update is the standard sampling update used in MPPI.

There is another way to arrive at the same update. Instead of starting from a KL-regularized distribution optimization problem, we can define a smooth objective over the sampling mean $\bar{x}$ and view MPPI as a **natural-gradient ascent** step on that objective. The objective is

$$
L_{\Sigma}(\bar{x})
:=
\log
\mathbb{E}_{x\sim\mathcal{N}(\bar{x},\Sigma)}
\left[
\exp\left(-\frac{1}{\lambda}f(x)\right)
\right],
$$

which evaluates the cost not at the single point $\bar{x}$, but over the Gaussian neighborhood.

The reason this is called a *score*-ascent view is that $\nabla_{\bar{x}} L_{\Sigma}(\bar{x})$ turns out to be the score (the gradient of the log density) of a *noised target distribution* over the center $\bar{x}$. Specifically, $\exp(L_{\Sigma}(\bar{x}))$ is an unnormalized density, and normalizing it does not change the gradient, so the gradient of the objective and the score of the noised target are the same vector.

The goal of this post is to derive the same MPPI update as natural-gradient ascent on $L_{\Sigma}(\bar{x})$, while keeping the notation aligned with the previous post.

## Low-Cost Target Distribution

We start from the deterministic optimization problem

$$
\begin{equation}
\label{eq:score_original_problem}
    x^\star
    \in
    \underset{x \in \mathbb{R}^n}{\operatorname{argmin}}\;
    f(x).
\end{equation}
$$

As before, $x$ is the decision variable and $f(x)$ is the cost we want to minimize. A natural way to turn this optimization problem into a distributional object is to define an unnormalized low-cost density

$$
\phi(x)
:=
\exp\left(-\frac{1}{\lambda}f(x)\right),
$$

where $\lambda > 0$ is the temperature parameter. If the normalizing constant

$$
Z
=
\int
\exp\left(-\frac{1}{\lambda}f(x)\right)dx
$$

is finite, then this defines the normalized target distribution

$$
\begin{equation}
\label{eq:score_raw_target}
p_{\mathrm{tar}}(x)
=
\frac{1}{Z}
\exp\left(-\frac{1}{\lambda}f(x)\right).
\end{equation}
$$

This distribution assigns larger probability to lower-cost points. Directly maximizing $p_{\mathrm{tar}}(x)$ is equivalent to minimizing $f(x)$, because the exponential map is monotone:

$$
\underset{x}{\operatorname{argmax}}\;
p_{\mathrm{tar}}(x)
=
\underset{x}{\operatorname{argmin}}\;
f(x).
$$

The score of this raw target distribution is simple. Since

$$
\log p_{\mathrm{tar}}(x)
=
-\frac{1}{\lambda}f(x)
-
\log Z,
$$

and $Z$ does not depend on $x$, we have

$$
\begin{equation}
\label{eq:raw_target_score}
\nabla_x \log p_{\mathrm{tar}}(x)
=
-
\frac{1}{\lambda}
\nabla_x f(x).
\end{equation}
$$

Following this score with a gradient-ascent step gives

$$
\begin{equation}
\label{eq:raw_target_ascent}
x^{k+1}
=
x^k
+
\alpha\,\nabla_x \log p_{\mathrm{tar}}(x^k)
=
x^k
-
\frac{\alpha}{\lambda}
\nabla_x f(x^k).
\end{equation}
$$

Ascending the log density of the target is nothing but (temperature-scaled) gradient descent on the cost $f$. This is the most direct sense in which "follow the score" and "minimize the cost" coincide.

However, this update relies on the cost gradient $\nabla_x f(x)$, which is exactly what MPPI tries to avoid: in practice $f$ may be non-differentiable or expensive to differentiate. Rather than computing $\nabla_x f$, MPPI samples perturbations around the current mean, evaluates the cost of those samples, and takes a weighted average of the perturbations. To see that this sampling update is also a score-ascent step, just on a different target, we now introduce a noised version of the target distribution.

## Noised Target Distribution

The raw target $p_{\mathrm{tar}}$ from the previous section is the object we would like to ascend, but its score \eqref{eq:raw_target_score} requires the cost gradient $\nabla_x f(x)$. To avoid that, we replace the pointwise unnormalized density $\phi(x)=\exp\left(-\frac{1}{\lambda}f(x)\right)$ by its local average over a Gaussian neighborhood.

At iteration $k$, suppose the current sampling distribution is Gaussian:

$$
p_{\theta^k}(x)
=
\mathcal{N}(\bar{x}^k,\Sigma^k).
$$

With the covariance fixed, the parameter of the sampling distribution is just its mean, $\theta = \bar{x}$. Given a covariance matrix $\Sigma$, we define the Gaussian-averaged version of $\phi$,

$$
\begin{equation}
h_{\Sigma}(\bar{x})
:=
\mathbb{E}_{x \sim \mathcal{N}(\bar{x},\Sigma)}
\left[
\phi(x)
\right]
=
\int
\mathcal{N}(x;\bar{x},\Sigma)
\,\phi(x)\,
dx
=
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
dx.
\label{eq:h_sigma_expectation}
\end{equation}
$$

Comparing with the raw target $p_{\mathrm{tar}}(x)=\phi(x)/Z$ in \eqref{eq:score_raw_target}, the only change is that $\phi$ is now smoothed by the sampling Gaussian. As $\Sigma\to 0$ the smoothing disappears and $h_{\Sigma}(\bar{x})\to\phi(\bar{x})$, so we recover the raw target up to the constant $Z$; for finite $\Sigma$ we obtain a smoothed surrogate.

The smoothing also gives $h_{\Sigma}$ a natural interpretation. Its value is large when Gaussian samples

$$
x = \bar{x} + \epsilon,
\quad
\epsilon \sim \mathcal{N}(0,\Sigma),
$$

drawn around $\bar{x}$ tend to have low cost. So $h_{\Sigma}(\bar{x})$ does not measure whether the center $\bar{x}$ itself is good, but whether its neighborhood contains good samples. The optimization problem in this view is

$$
\begin{equation}
\label{eq:log_h_objective}
\underset{\bar{x}}{\operatorname{maximize}}\;
L_{\Sigma}(\bar{x})
:=
\log h_{\Sigma}(\bar{x}).
\end{equation}
$$

Maximizing $L_{\Sigma}(\bar{x})=\log h_{\Sigma}(\bar{x})$ has the same maximizers as maximizing $h_{\Sigma}(\bar{x})$, because the logarithm is monotone. Working in log space is also convenient: its gradient is the normalized ratio $\nabla_{\bar{x}}h_{\Sigma}/h_{\Sigma}$, and this normalization is exactly what turns raw exponential weights into the normalized MPPI weights. We will see this concretely when we compute the score below.

So far, $h_{\Sigma}$ is just an unnormalized objective, and $\nabla_{\bar{x}}L_{\Sigma}$ is just its gradient. To justify calling that gradient a *score*, we need an actual probability distribution behind it. The next step does exactly that: we normalize $h_{\Sigma}$ into a distribution $p_{\Sigma}$, so that $\nabla_{\bar{x}}L_{\Sigma}$ becomes literally the score $\nabla_{\bar{x}}\log p_{\Sigma}$.

To this end, define the normalized noised target distribution

$$
\begin{equation}
\label{eq:noised_target}
p_{\Sigma}(\bar{x})
=
\frac{1}{Z_{\Sigma}}
h_{\Sigma}(\bar{x}),
\end{equation}
$$

where $Z_{\Sigma}$ is the normalizing constant over $\bar{x}$. Since $Z_{\Sigma}$ does not depend on $\bar{x}$,

$$
\nabla_{\bar{x}} \log p_{\Sigma}(\bar{x})
=
\nabla_{\bar{x}} \log h_{\Sigma}(\bar{x})
=
\nabla_{\bar{x}} L_{\Sigma}(\bar{x}).
$$

Therefore, computing the score of $p_{\Sigma}$ is the same as computing the gradient of the optimization objective $L_{\Sigma}$.

## Score of the Noised Target

We now compute the gradient of the objective. Differentiating the logarithm gives

$$
\nabla_{\bar{x}} L_{\Sigma}(\bar{x})
=
\nabla_{\bar{x}} \log h_{\Sigma}(\bar{x})
=
\frac{
\nabla_{\bar{x}} h_{\Sigma}(\bar{x})
}{
h_{\Sigma}(\bar{x})
}.
$$

The remaining work is to compute $\nabla_{\bar{x}} h_{\Sigma}(\bar{x})$. The key is to differentiate the Gaussian density with respect to its mean rather than the cost: this moves the derivative onto a known distribution and yields an estimator that uses only sampled costs, never the cost gradient $\nabla_x f$. Starting from the integral form \eqref{eq:h_sigma_expectation} and differentiating under the integral sign with respect to $\bar{x}$,

$$
\begin{aligned}
\nabla_{\bar{x}} h_{\Sigma}(\bar{x})
&=
\nabla_{\bar{x}}
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
dx \\
&=
\int
\nabla_{\bar{x}}
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
dx.
\end{aligned}
$$

Using

$$
\nabla_{\bar{x}}
\mathcal{N}(x;\bar{x},\Sigma)
=
\mathcal{N}(x;\bar{x},\Sigma)
\nabla_{\bar{x}}
\log
\mathcal{N}(x;\bar{x},\Sigma),
$$

we obtain

$$
\nabla_{\bar{x}} h_{\Sigma}(\bar{x})
=
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
\nabla_{\bar{x}}
\log
\mathcal{N}(x;\bar{x},\Sigma)
dx.
$$

For a Gaussian density with fixed covariance,

$$
\log
\mathcal{N}(x;\bar{x},\Sigma)
=
-
\frac{1}{2}
(x-\bar{x})^\top
\Sigma^{-1}
(x-\bar{x})
+
\mathrm{const},
$$

so

$$
\nabla_{\bar{x}}
\log
\mathcal{N}(x;\bar{x},\Sigma)
=
\Sigma^{-1}
(x-\bar{x}).
$$

Substituting this into the previous expression gives

$$
\nabla_{\bar{x}} h_{\Sigma}(\bar{x})
=
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
\Sigma^{-1}(x-\bar{x})
dx.
$$

Therefore,

$$
\begin{aligned}
\nabla_{\bar{x}} \log p_{\Sigma}(\bar{x})
&=
\nabla_{\bar{x}} \log h_{\Sigma}(\bar{x}) \\
&=
\frac{
\nabla_{\bar{x}} h_{\Sigma}(\bar{x})
}{
h_{\Sigma}(\bar{x})
} \\
&=
\frac{
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
\Sigma^{-1}(x-\bar{x})
dx
}{
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
dx
}.
\end{aligned}
$$

Since $\Sigma$ does not depend on $x$, we can pull $\Sigma^{-1}$ outside the integral:

$$
\begin{equation}
\label{eq:noised_score_integral}
\nabla_{\bar{x}} \log p_{\Sigma}(\bar{x})
=
\Sigma^{-1}
\frac{
\int
(x-\bar{x})
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
dx
}{
\int
\mathcal{N}(x;\bar{x},\Sigma)
\exp\left(-\frac{1}{\lambda}f(x)\right)
dx
}.
\end{equation}
$$

The score of the noised target distribution is a weighted average of the sampled perturbations. Next, we talk about the natural gradient. For the Gaussian mean parameter $\theta=\bar{x}$ with fixed covariance, the Fisher information of the sampling distribution is defined as the covariance of the score of $\mathcal{N}(\bar{x},\Sigma)$,

$$
F(\bar{x})
:=
\mathbb{E}_{x\sim\mathcal{N}(\bar{x},\Sigma)}
\left[
\nabla_{\bar{x}}\log\mathcal{N}(x;\bar{x},\Sigma)\,
\nabla_{\bar{x}}\log\mathcal{N}(x;\bar{x},\Sigma)^\top
\right].
$$

We already computed the score of this Gaussian in the mean,

$$
\nabla_{\bar{x}}\log\mathcal{N}(x;\bar{x},\Sigma)
=
\Sigma^{-1}(x-\bar{x}).
$$

Substituting and using $\mathbb{E}\left[(x-\bar{x})(x-\bar{x})^\top\right]=\Sigma$ gives

$$
\begin{aligned}
F(\bar{x})
&=
\mathbb{E}
\left[
\Sigma^{-1}(x-\bar{x})\,(x-\bar{x})^\top\Sigma^{-1}
\right] \\
&=
\Sigma^{-1}\,
\mathbb{E}\left[(x-\bar{x})(x-\bar{x})^\top\right]\,
\Sigma^{-1} \\
&=
\Sigma^{-1}\,\Sigma\,\Sigma^{-1} \\
&=
\Sigma^{-1}.
\end{aligned}
$$

Therefore, the natural gradient of the noised log density is

$$
\begin{aligned}
\widetilde{\nabla}_{\bar{x}}
\log p_{\Sigma}(\bar{x})
&=
F(\bar{x})^{-1}
\nabla_{\bar{x}}
\log p_{\Sigma}(\bar{x}) \\
&=
\Sigma
\nabla_{\bar{x}}
\log p_{\Sigma}(\bar{x}).
\end{aligned}
$$

The preconditioner $\Sigma$ exactly cancels the explicit $\Sigma^{-1}$ in the score \eqref{eq:noised_score_integral}, leaving the bare cost-weighted average of the perturbations $x-\bar{x}$. This is the same geometric cancellation that appears in the natural-gradient view of Gaussian mean updates. We estimate this quantity from samples in the next section.

## Monte Carlo Approximation

Equation \eqref{eq:noised_score_integral} is written as a ratio of two integrals under the Gaussian sampling distribution

$$
x \sim \mathcal{N}(\bar{x},\Sigma).
$$

Therefore, it can be approximated using samples from that Gaussian. Draw

$$
\epsilon_i \sim \mathcal{N}(0,\Sigma),
\quad
x_i = \bar{x} + \epsilon_i,
\quad
i=1,\ldots,N.
$$

Evaluate

$$
f_i = f(x_i).
$$

The denominator in \eqref{eq:noised_score_integral} is approximated by

$$
\frac{1}{N}
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda}f_j\right),
$$

and the numerator is approximated by

$$
\frac{1}{N}
\sum_{i=1}^{N}
\epsilon_i
\exp\left(-\frac{1}{\lambda}f_i\right).
$$

The factors $\frac{1}{N}$ cancel, giving

$$
\nabla_{\bar{x}} \log p_{\Sigma}(\bar{x})
\approx
\Sigma^{-1}
\frac{
\sum_{i=1}^{N}
\epsilon_i
\exp\left(-\frac{1}{\lambda}f_i\right)
}{
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda}f_j\right)
}.
$$

Using the same normalized weights as in the previous post,

$$
w_i
:=
\frac{
\exp\left(-\frac{1}{\lambda}f_i\right)
}{
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda}f_j\right)
},
$$

we obtain the Monte Carlo estimate of the objective gradient, equivalently the score of the noised target:

$$
\begin{equation}
\label{eq:monte_carlo_noised_score}
\nabla_{\bar{x}} \log p_{\Sigma}(\bar{x})
\approx
\Sigma^{-1}
\sum_{i=1}^{N}
w_i \epsilon_i.
\end{equation}
$$

This formula is important because it expresses the gradient of $L_{\Sigma}(\bar{x})$ using only sampled costs. We do not need to compute $\nabla_x f(x)$.

## MPPI as Score Ascent

We have now defined the objective

$$
L_{\Sigma}(\bar{x})
=
\log h_{\Sigma}(\bar{x})
=
\log
\mathbb{E}_{x\sim\mathcal{N}(\bar{x},\Sigma)}
\left[
\exp\left(-\frac{1}{\lambda}f(x)\right)
\right].
$$

Since $\nabla_{\bar{x}}L_{\Sigma}(\bar{x})=\nabla_{\bar{x}}\log p_{\Sigma}(\bar{x})$, ordinary gradient ascent on this objective would be

$$
\bar{x}^{k+1}
=
\bar{x}^k
+
\alpha
\nabla_{\bar{x}}
\log
p_{\Sigma^k}(\bar{x}^k),
$$

where $\alpha > 0$ is a step size. However, the score in \eqref{eq:monte_carlo_noised_score} contains the factor $(\Sigma^k)^{-1}$. MPPI instead ascends the natural gradient derived above:

$$
\begin{equation}
\label{eq:preconditioned_score_ascent}
\bar{x}^{k+1}
=
\bar{x}^k
+
\alpha
\widetilde{\nabla}_{\bar{x}}
\log
p_{\Sigma^k}(\bar{x}^k)
=
\bar{x}^k
+
\alpha
\Sigma^k
\nabla_{\bar{x}}
\log
p_{\Sigma^k}(\bar{x}^k).
\end{equation}
$$

Substituting \eqref{eq:monte_carlo_noised_score} into \eqref{eq:preconditioned_score_ascent} gives

$$
\bar{x}^{k+1}
\approx
\bar{x}^k
+
\alpha
\Sigma^k
\left(
(\Sigma^k)^{-1}
\sum_{i=1}^{N}
w_i \epsilon_i
\right).
$$

Therefore,

$$
\bar{x}^{k+1}
\approx
\bar{x}^k
+
\alpha
\sum_{i=1}^{N}
w_i \epsilon_i.
$$

With $\alpha=1$, this is exactly the exponential weighted sampling update:

$$
\begin{equation}
\label{eq:score_mppi_update}
\bar{x}^{k+1}
\approx
\bar{x}^k
+
\sum_{i=1}^{N}
w_i \epsilon_i.
\end{equation}
$$

Thus, the standard MPPI update can be interpreted as one natural-gradient ascent step on the noised target distribution $p_{\Sigma^k}(\bar{x})$, smoothed by the Gaussian distribution.

## Closing Remarks

We took the same weighted update from the previous post and gave it a second derivation. Rather than projecting a reweighted distribution back onto the Gaussian family, we built a smooth objective over the sampling mean, recognized its gradient as a *score*, and showed that a single covariance-preconditioned score-ascent step reproduces the MPPI update

$$
\bar{x}^{k+1}
=
\bar{x}^k
+
\sum_{i=1}^{N} w_i \epsilon_i
$$

exactly. The chain of reasoning was:

- smooth the low-cost target over a Gaussian neighborhood to obtain $h_{\Sigma}(\bar{x})$;
- take the objective $L_{\Sigma}=\log h_{\Sigma}$, whose gradient is the score of the normalized noised target $p_{\Sigma}$;
- differentiate the Gaussian density rather than the cost, writing the score as a cost-weighted average of perturbations;
- estimate that score by Monte Carlo and precondition by the Gaussian-mean Fisher metric $\Sigma$, which cancels the $\Sigma^{-1}$ and leaves precisely the MPPI weights.

This tutorial was written with the assistance of ChatGPT-5.


<!-- ## References

{% bibliography --cited %} -->
