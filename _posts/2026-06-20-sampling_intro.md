---
layout: post
title: Sampling-Based Optimization
date: 2026-06-20
description: A short introduction to exponential weighted sampling updates
tags: sampling optimization control
categories:
related_posts: false
---

In previous posts, we mainly focused on optimization methods that exploit first-order (gradient) or second-order (Hessian) information.
These methods are powerful when the objective and constraints are smooth enough and when useful local approximations are available. In addition, they are often local methods: each iteration only uses information around the current solution. For nonconvex problems, the final solution can strongly depend on the initialization, and the algorithm may converge to a poor local minimum.

Sampling-based optimization provides a different perspective. Instead of directly following a gradient direction, we maintain a sampling distribution, generate candidate solutions from it, evaluate their costs, and then update the distribution toward lower-cost regions. 
This makes the method relatively easy to combine with complex simulators: as long as we can evaluate the cost of a candidate solution, we can use it for the update.

## Problem Formulation

Let us consider the deterministic optimization problem

$$
\begin{equation}
\label{eq:original_problem}
    x^\star
    \in
    \underset{x \in \mathbb{R}^n}{\operatorname{argmin}}\;
    f(x),
\end{equation}
$$

where $x \in \mathbb{R}^{n}$ denotes the decision variable and $f(x)$ is the cost function we want to minimize. In a standard optimization method, we directly update a candidate point $x^k$. In a sampling-based method, we instead update a probability distribution over possible candidates.

Let $\mathcal{P}$ denote the set of probability distributions over $\mathbb{R}^n$. We can reformulate \eqref{eq:original_problem} as an optimization problem over distributions:

$$
\begin{equation}
\label{eq:distribution_problem}
    p_{\mathrm{opt}}
    \in
    \underset{p \in \mathcal{P}}{\operatorname{argmin}}\;
    \mathbb{E}_{p}\left[f(x)\right].
\end{equation}
$$

Since an expectation is an average, it cannot be smaller than the minimum value of $f(x)$:

$$
\mathbb{E}_{p}\left[f(x)\right]
\ge
\mathbb{E}_{p}\left[\min_x f(x)\right]
=
\min_x f(x).
$$

Equality is achieved by placing all probability mass on a global minimizer $x^\star$, that is,

$$
p_{\mathrm{opt}}(x)
=
\delta(x-x^\star),
$$

where $\delta(\cdot)$ denotes the Dirac distribution.
Therefore, the distributional problem has the same optimal value as the original problem, and it is another way of viewing the same optimization problem.

However, solving \eqref{eq:distribution_problem} directly is not computationally practical, as the search space $\mathcal{P}$ is the space of all probability distributions over $\mathbb{R}^n$, which is infinite-dimensional. 
Therefore, in practice, we maintain a tractable sampling distribution from a parameterized family

$$
\mathcal{P}_{\Theta}
=
\left\{
p_{\theta}(x)
\;\middle|\;
\theta \in \Theta
\right\}.
$$

The corresponding parameterized problem is

$$
\begin{equation}
\label{eq:parametric_distribution_problem}
    \underset{\theta \in \Theta}{\operatorname{minimize}}\;
    J(\theta)
    :=
    \mathbb{E}_{p_{\theta}}\left[f(x)\right].
\end{equation}
$$

The parameter $\theta$ determines how samples are generated. For example, if the sampling distribution is Gaussian, then the parameters are the mean and covariance,

$$
\theta = \left(\bar{x}, \Sigma\right),
$$

and we can write

$$
p_{\theta}(x) = \mathcal{N}(\bar{x}, \Sigma).
$$

At iteration $k$, suppose the current sampling distribution is

$$
p_{\theta^k}(x),
$$

where $\theta^k$ denotes the current distribution parameters. In the Gaussian case,

$$
\theta^k = \left(\bar{x}^k, \Sigma^k\right),
$$

and we can write

$$
p_{\theta^k}(x) = \mathcal{N}(\bar{x}^k, \Sigma^k).
$$

Here, $\bar{x}^k$ is the current center of the search distribution, and $\Sigma^k$ describes the exploration scale and direction.

The next question is how to update $p_{\theta^k}(x)$ so that the new distribution places more probability mass on lower-cost regions. 
The update should reduce the expected cost, but it should not move arbitrarily far away from $p_{\theta^k}(x)$ in one iteration. Since we only observe the cost through finitely many samples drawn from $p_{\theta^k}(x)$, regions far away from the current distribution are poorly estimated. A large distribution shift can therefore make the update unstable or overly sensitive to a few lucky samples.

## KL-Regularized Distribution Update

A natural way to formulate this idea is to first define an ideal target distribution $p^\star(x)$. This target distribution should have low expected cost, while staying close to the current sampling distribution:

$$
\begin{equation}
\label{eq:kl_sampling_update}
    p^\star(x)
    =
    \underset{q}{\operatorname{argmin}}\;
    \mathbb{E}_{q}\left[f(x)\right]
    +
    \lambda D_{\mathrm{KL}}\left(q(x)\,\Vert\,p_{\theta^k}(x)\right),
\end{equation}
$$

where the optimization is over all probability distributions $q(x)$. The parameter $\lambda > 0$ controls how strongly we penalize the change of distribution. The KL term acts as a regularization in distribution space. If $\lambda$ is large, the target distribution is forced to stay close to the current one. If $\lambda$ is small, the target distribution can move more aggressively toward low-cost regions.

Interestingly, this is closely related to proximal methods in gradient-based optimization. A classical proximal step has the form

$$
x^{k+1}
=
\underset{x}{\operatorname{argmin}}\;
f(x)
+
\frac{1}{2\alpha}
\left\Vert x - x^k \right\Vert^2,
$$

where the quadratic term prevents the next iterate from moving too far away from the current one. In \eqref{eq:kl_sampling_update}, we apply the same idea to probability distributions: the variable being updated is $q(x)$ instead of a point $x$, and the distance to the previous iterate is measured by the KL divergence instead of the Euclidean distance. In this sense, the update can be viewed as a proximal step in distribution space.

The optimization problem in \eqref{eq:kl_sampling_update} has a closed-form solution. By expanding the KL divergence, we obtain

$$
\begin{aligned}
    p^\star
    &=
    \underset{q}{\operatorname{argmin}}\;
    \int q(x) f(x) dx
    +
    \lambda
    \int q(x)
    \log \frac{q(x)}{p_{\theta^k}(x)} dx.
\end{aligned}
$$

Solving this variational problem gives the exponentially reweighted distribution

$$
p^\star(x)
=
\frac{1}{\eta}
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda} f(x)\right),
$$

where $\lambda > 0$ is the temperature parameter, and $\eta$ is the normalizing constant

$$
\eta =
\int
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda} f(x)\right)
dx.
$$

This expression says that samples with lower cost should receive larger probability under the ideal target distribution. The parameter $\lambda$ plays the role of a temperature. When $\lambda$ is small, the probability mass concentrates around the best samples. When $\lambda$ is large, the update becomes smoother and more conservative.

## Projection Back to the Sampling Family

However, $p^\star(x)$ is generally not in the same parametric family as $p_{\theta^k}(x)$. For example, even if $p_{\theta^k}(x)$ is Gaussian, the product

$$
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda} f(x)\right)
$$

is Gaussian only for special choices of $f(x)$, such as a quadratic cost. For a general nonlinear cost function, the ideal target distribution $p^\star(x)$ can be non-Gaussian and even multimodal. Therefore, after constructing $p^\star(x)$, we project it back to the sampling family. If our sampling family is $\{p_\theta(x)\}$, one common projection is

$$
\theta^{k+1}
=
\underset{\theta}{\operatorname{argmin}}\;
D_{\mathrm{KL}}\left(p^\star(x)\,\Vert\,p_{\theta}(x)\right).
$$

This projection gives the next sampling distribution $p_{\theta^{k+1}}(x)$. Let us now compute this projection explicitly for a simple Gaussian case. Suppose we keep the covariance matrix fixed and only update the mean:

$$
p_{\theta}(x)
=
\mathcal{N}(\bar{x}, \Sigma^k).
$$

The projection problem becomes

$$
\bar{x}^{k+1}
=
\underset{\bar{x}}{\operatorname{argmin}}\;
D_{\mathrm{KL}}
\left(
p^\star(x)
\;\Vert\;
\mathcal{N}(\bar{x}, \Sigma^k)
\right).
$$

Expanding the KL divergence, we have

$$
\begin{aligned}
D_{\mathrm{KL}}
\left(
p^\star(x)
\;\Vert\;
\mathcal{N}(\bar{x}, \Sigma^k)
\right)
&=
\int p^\star(x)
\log
\frac{p^\star(x)}
{\mathcal{N}(x;\bar{x}, \Sigma^k)}
dx \\
&=
\int p^\star(x)\log p^\star(x)dx
-
\int p^\star(x)\log \mathcal{N}(x;\bar{x}, \Sigma^k)dx.
\end{aligned}
$$

The first term does not depend on $\bar{x}$. To simplify the second term, recall the density of a multivariate Gaussian distribution with mean $\bar{x}$ and positive definite covariance $\Sigma^k$:

$$
\mathcal{N}(x;\bar{x}, \Sigma^k)
=
\frac{1}
{(2\pi)^{n/2}\sqrt{\det(\Sigma^k)}}
\exp
\left(
-\frac{1}{2}
\left(x-\bar{x}\right)^\top
\left(\Sigma^k\right)^{-1}
\left(x-\bar{x}\right)
\right).
$$

The exponential term measures the squared distance between $x$ and the mean $\bar{x}$, scaled by the inverse covariance matrix. The coefficient in front only normalizes the density so that it integrates to one. Taking the logarithm gives

$$
\log \mathcal{N}(x;\bar{x}, \Sigma^k)
=
-\frac{1}{2}
\left(x-\bar{x}\right)^\top
\left(\Sigma^k\right)^{-1}
\left(x-\bar{x}\right)
-
\frac{n}{2}\log(2\pi)
-
\frac{1}{2}\log\det(\Sigma^k).
$$

Since $\Sigma^k$ is fixed, the normalization terms $\frac{n}{2}\log(2\pi)$ and $\frac{1}{2}\log\det(\Sigma^k)$ also do not depend on $\bar{x}$. Therefore,

$$
\begin{aligned}
-\int p^\star(x)\log \mathcal{N}(x;\bar{x}, \Sigma^k)dx
&=
\frac{1}{2}
\int p^\star(x)
\left(x-\bar{x}\right)^\top
\left(\Sigma^k\right)^{-1}
\left(x-\bar{x}\right)
dx
+ \mathrm{const} \\
&=
\frac{1}{2}
\mathbb{E}_{p^\star}
\left[
\left(x-\bar{x}\right)^\top
\left(\Sigma^k\right)^{-1}
\left(x-\bar{x}\right)
\right]
+ \mathrm{const}.
\end{aligned}
$$

After dropping all terms independent of $\bar{x}$, the projection is equivalent to minimizing

$$
\bar{x}^{k+1}
=
\underset{\bar{x}}{\operatorname{argmin}}\;
\frac{1}{2}
\mathbb{E}_{p^\star}
\left[
\left(x-\bar{x}\right)^\top
\left(\Sigma^k\right)^{-1}
\left(x-\bar{x}\right)
\right].
$$

Taking the gradient with respect to $\bar{x}$ gives

$$
\nabla_{\bar{x}}
\left(
\frac{1}{2}
\mathbb{E}_{p^\star}
\left[
\left(x-\bar{x}\right)^\top
\left(\Sigma^k\right)^{-1}
\left(x-\bar{x}\right)
\right]
\right)
=
\left(\Sigma^k\right)^{-1}
\left(
\bar{x}
-
\mathbb{E}_{p^\star}[x]
\right).
$$

The first-order optimality condition is therefore

$$
\left(\Sigma^k\right)^{-1}
\left(
\bar{x}^{k+1}
-
\mathbb{E}_{p^\star}[x]
\right)
=
0,
$$

which gives

$$
\bar{x}^{k+1}
=
\mathbb{E}_{p^\star}[x].
$$

## Monte Carlo Approximation

Thus, the weighted mean update is the result of projecting the ideal target distribution $p^\star(x)$ back to the Gaussian sampling family. Using the expression of $p^\star(x)$, we obtain

$$
\begin{aligned}
\bar{x}^{k+1}
&=
\mathbb{E}_{p^\star}[x] \\
&=
\int \textcolor{blue}{x} p^\star(x)dx \\
&=
\frac{1}{\eta}
\int
\textcolor{blue}{x}
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda} f(x)\right)
dx,
\end{aligned}
$$

with

$$
\eta
=
\int
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda} f(x)\right)
dx.
$$

Since both the numerator and the denominator are integrals with respect to the current sampling distribution $p_{\theta^k}(x)$. Therefore, we can rewrite the update as a ratio of expectations under the distribution that we can sample from:

$$
\bar{x}^{k+1}
=
\frac{
\mathbb{E}_{p_{\theta^k}}
\left[
\textcolor{blue}{x}
\exp\left(-\frac{1}{\lambda} f(x)\right)
\right]
}{
\mathbb{E}_{p_{\theta^k}}
\left[
\exp\left(-\frac{1}{\lambda} f(x)\right)
\right]
}.
$$

This form is important because it only requires samples from the current distribution. We draw

$$
x_i \sim p_{\theta^k}(x),
\quad i = 1,\ldots,N,
$$

and evaluate the cost

$$
f_i = f(x_i).
$$

The denominator is approximated by

$$
\mathbb{E}_{p_{\theta^k}}
\left[
\exp\left(-\frac{1}{\lambda} f(x)\right)
\right]
\approx
\frac{1}{N}
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda} f_j\right),
$$

and the numerator is approximated by

$$
\mathbb{E}_{p_{\theta^k}}
\left[
x
\exp\left(-\frac{1}{\lambda} f(x)\right)
\right]
\approx
\frac{1}{N}
\sum_{i=1}^{N}
x_i
\exp\left(-\frac{1}{\lambda} f_i\right).
$$

The factors $\frac{1}{N}$ cancel in the ratio, giving

$$
\bar{x}^{k+1}
\approx
\frac{
\sum_{i=1}^{N}
x_i
\exp\left(-\frac{1}{\lambda} f_i\right)
}{
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda} f_j\right)
}.
$$

Equivalently, we define the normalized weight of each sample as

$$
w_i
:=
\frac{
\exp\left(-\frac{1}{\lambda} f_i\right)
}{
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda} f_j\right)
}.
$$

With this notation, the Monte Carlo projection update becomes

$$
\bar{x}^{k+1}
\approx
\sum_{i=1}^{N} w_i x_i.
$$

This is the same weighted average derived above, written in a compact form. In implementations, the samples are often generated by perturbing the current mean:

$$
x_i = \bar{x}^k + \epsilon_i,
\quad
\epsilon_i \sim \mathcal{N}(0, \Sigma^k),
$$

then the update can be written equivalently as

$$
\bar{x}^{k+1}
=
\bar{x}^k
+
\sum_{i=1}^{N} w_i \epsilon_i.
$$

This algorithm does not simply choose the best sample. Instead, it forms a soft weighted average of all samples, where low-cost samples contribute more strongly to the update.

<div class="algorithm" markdown="1">
**Algorithm 1:** Exponential Weighted Sampling Update
<hr style="margin: 0.5em 0;">

Given current distribution $p_{\theta^k}(x) = \mathcal{N}(\bar{x}^k, \Sigma^k)$  
Sample $\epsilon_i \sim \mathcal{N}(0, \Sigma^k)$ for $i=1,\ldots,N$  
Set $x_i \gets \bar{x}^k + \epsilon_i$  
Evaluate $f_i \gets f(x_i)$  
Compute

$$
w_i \gets
\frac{
\exp\left(-\frac{1}{\lambda} f_i\right)
}{
\sum_{j=1}^{N}
\exp\left(-\frac{1}{\lambda} f_j\right)
}
$$

Update

$$
\bar{x}^{k+1}
\gets
\bar{x}^k + \sum_{i=1}^{N} w_i \epsilon_i
$$
</div>

## Connection to MPPI

In model predictive control, $x$ can be chosen to be the whole control sequence over a planning horizon. In that case, the exponential weighted update above becomes the standard sampling update used in Model Predictive Path Integral (MPPI) control.

The main limitation of this standard update is that the weights are determined by the full cost $f(x_i)$. If most samples have much larger cost than a few lucky samples, the weights become highly concentrated. In that case, the update

$$
\sum_{i=1}^{N} w_i \epsilon_i
$$

is dominated by only a small number of samples. As a result, different batches of random samples may produce very different update directions. This is the high variance here: it refers to the variance of the Monte Carlo estimate of the update.

Additionally, in standard MPPI, the covariance matrix $\Sigma^k$ is often fixed. This makes the algorithm simple, but it also limits adaptivity. If $\Sigma^k$ is too small, the samples stay close to the current mean and may fail to explore better regions. If $\Sigma^k$ is too large, the samples spread over a wide region, and many of them may land in areas with very high cost. Since the weights depend exponentially on the cost,

$$
w_i \propto \exp\left(-\frac{1}{\lambda}f_i\right),
$$

these high-cost samples receive weights that are nearly zero. Then only a few low-cost samples effectively contribute to the update. This reduces the effective sample size. The update direction is mostly determined by those few samples, so a different random batch may produce a noticeably different direction. Thus, a fixed covariance matrix can make the method sensitive to tuning and can worsen sample inefficiency when the cost landscape changes across iterations.

This tutorial was written with the assistance of ChatGPT-5.


<!-- ## References

{% bibliography --cited %} -->
