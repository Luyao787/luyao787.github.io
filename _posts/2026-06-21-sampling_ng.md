---
layout: post
title: Natural Gradient View of Sampling-Based Optimization
date: 2026-06-24
description: A direct natural-gradient derivation of sampling distribution updates
tags: sampling optimization natural-gradient control
categories:
  - Sampling-based Optimization
related_posts: false
---

In the previous post, we derived an exponential weighted sampling update from a KL-regularized distribution optimization problem. The key idea was to first construct an ideal target distribution

$$
p^\star(x)
\propto
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda}f(x)\right),
$$

and then project this target distribution back to a tractable sampling family.
There is another natural way to look at the same problem. Instead of introducing an ideal target distribution first, we can directly optimize the parameters of the sampling distribution. This leads to a gradient-based view of sampling optimization.

The goal of this post is to derive this direct update.

## Parameterized Objective

Recall the deterministic optimization problem

$$
\begin{equation}
\label{eq:ng_original_problem}
    x^\star
    \in
    \underset{x \in \mathbb{R}^n}{\operatorname{argmin}}\;
    f(x).
\end{equation}
$$

In sampling-based optimization, instead of updating a single candidate point, we maintain a probability distribution over candidates. Suppose this distribution belongs to a parameterized family

$$
\mathcal{P}_{\Theta}
=
\left\{
p_{\theta}(x)
\;\middle|\;
\theta \in \Theta
\right\}.
$$

Then we can consider the parameterized distribution optimization problem

$$
\begin{equation}
\label{eq:ng_parametric_distribution_problem}
    \underset{\theta \in \Theta}{\operatorname{minimize}}\;
    J(\theta)
    :=
    \mathbb{E}_{p_{\theta}}\left[f(x)\right].
\end{equation}
$$

This objective asks for a sampling distribution whose average cost is small. If the family is expressive enough, minimizing this objective pushes the distribution toward low-cost regions. If the family is limited, for example if we restrict ourselves to a Gaussian distribution, then the solution is the best distribution within that family.

The important point is that \eqref{eq:ng_parametric_distribution_problem} is now a finite-dimensional optimization problem in the parameter $\theta$. Therefore, a natural question is:

<p style="text-align: center;"><em>Can we apply gradient descent directly to $J(\theta)$?</em></p>

The answer is yes. Moreover, because $\theta$ parameterizes a probability distribution, the $\textbf{natural gradient}$ is often the more appropriate gradient.

## Score Function Gradient

Assume that $p_\theta(x)$ is differentiable with respect to $\theta$. If $f(x)$ does not explicitly depend on $\theta$, then

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=
\nabla_\theta
\int
p_\theta(x) f(x) dx \\
&=
\int
f(x) \nabla_\theta p_\theta(x) dx.
\end{aligned}
$$

To rewrite this expression as an expectation, we use a simple identity involving the log density. The chain rule gives

$$
\nabla_\theta \log p_\theta(x)
=
\frac{1}{p_\theta(x)}
\nabla_\theta p_\theta(x).
$$

Multiplying both sides by $p_\theta(x)$ gives

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)
\nabla_\theta \log p_\theta(x).
$$

Substituting this identity into the gradient expression above gives

$$
\begin{aligned}
\nabla_\theta J(\theta)
&=
\int
f(x)
\nabla_\theta p_\theta(x) dx \\
&=
\int
f(x)
p_\theta(x)
\nabla_\theta \log p_\theta(x) dx \\
&=
\int
p_\theta(x)
\left[
f(x)
\nabla_\theta \log p_\theta(x)
\right] dx.
\end{aligned}
$$

By the definition of expectation under $p_\theta$, we obtain

$$
\begin{equation}
\label{eq:score_function_gradient}
\nabla_\theta J(\theta)
=
\mathbb{E}_{p_\theta}
\left[
f(x)
\nabla_\theta \log p_\theta(x)
\right].
\end{equation}
$$

This is called the score function gradient, which is useful because it only requires evaluating the cost $f(x)$ and the score $\nabla_\theta \log p_\theta(x)$. We do not need the derivative of $f(x)$ with respect to $x$. This formula is also closely related to the policy gradient estimator in reinforcement learning.

Given samples

$$
x_i \sim p_\theta(x),
\quad
i = 1,\ldots,N,
$$

the gradient can be approximated by

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{N}
\sum_{i=1}^{N}
f(x_i)
\nabla_\theta \log p_\theta(x_i).
$$

<!-- In practice, it is common to subtract a baseline $b$ from the cost:

$$
\begin{equation}
\label{eq:score_function_gradient_baseline}
\nabla_\theta J(\theta)
=
\mathbb{E}_{p_\theta}
\left[
\left(f(x)-b\right)
\nabla_\theta \log p_\theta(x)
\right].
\end{equation}
$$

This does not change the expected gradient, because

$$
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\right]
=
\int
p_\theta(x)
\nabla_\theta \log p_\theta(x)dx
=
\int
\nabla_\theta p_\theta(x)dx
=
\nabla_\theta 1
=
0.
$$

The baseline can reduce the variance of the Monte Carlo estimator. A simple choice is the sample average cost. -->

## Why Natural Gradient?

The gradient in \eqref{eq:score_function_gradient} is the ordinary Euclidean gradient with respect to the parameter vector $\theta$. Before introducing the natural gradient, let us first recall how classical optimization methods choose an update direction. At the current parameter $\theta^k$, consider a small step $\delta\theta$. A first-order Taylor approximation gives

$$
J(\theta^k+\delta\theta)
\approx
J(\theta^k)
+
\nabla_\theta J(\theta^k)^\top \delta\theta.
$$

If we only minimize this linear approximation, the objective can be decreased without bound by taking an arbitrarily large step in the negative gradient direction. Gradient descent avoids this by penalizing the Euclidean length of the step:

$$
\delta\theta^\star
=
\underset{\delta\theta}{\operatorname{argmin}}\;
\nabla_\theta J(\theta^k)^\top \delta\theta
+
\frac{1}{2\alpha}
\left\Vert \delta\theta \right\Vert^2.
$$

Taking the derivative with respect to $\delta\theta$ gives

$$
\nabla_\theta J(\theta^k)
+
\frac{1}{\alpha}
\delta\theta^\star
=
0,
$$

and therefore

$$
\delta\theta^\star
=
-
\alpha
\nabla_\theta J(\theta^k).
$$

Applying this displacement gives the usual gradient descent update:

$$
\theta^{k+1}
=
\theta^k
-
\alpha
\nabla_\theta J(\theta^k).
$$

This derivation shows that gradient descent is not only about the gradient direction. It also makes a $\textbf{geometric}$ choice: the size of a step is measured by the Euclidean norm

$$
\left\Vert \delta\theta \right\Vert^2
=
\delta\theta^\top I \delta\theta.
$$

Newton's method uses a different local model. Instead of using only the first-order approximation, it uses the second-order Taylor approximation

$$
J(\theta^k+\delta\theta)
\approx
J(\theta^k)
+
\nabla_\theta J(\theta^k)^\top \delta\theta
+
\frac{1}{2}
\delta\theta^\top
H(\theta^k)
\delta\theta,
$$

where

$$
H(\theta^k)
=
\nabla_\theta^2 J(\theta^k)
$$

is the Hessian of the objective. If $H(\theta^k)$ is positive definite, minimizing this quadratic model gives

$$
\delta\theta^\star
=
-
H(\theta^k)^{-1}
\nabla_\theta J(\theta^k),
$$

and therefore

$$
\theta^{k+1}
=
\theta^k
-
H(\theta^k)^{-1}
\nabla_\theta J(\theta^k).
$$

Gradient descent and Newton's method can both be viewed as local model minimization. Gradient descent uses the Euclidean geometry of parameter space, while Newton's method uses the local curvature of the objective. More generally, we can choose a positive definite matrix $M(\theta^k)$ to define the local size of a step:

$$
\delta\theta^\top
M(\theta^k)
\delta\theta.
$$

The local regularized problem then becomes

$$
\delta\theta^\star
=
\underset{\delta\theta}{\operatorname{argmin}}\;
\nabla_\theta J(\theta^k)^\top \delta\theta
+
\frac{1}{2\alpha}
\delta\theta^\top
M(\theta^k)
\delta\theta.
$$

Solving this problem gives

$$
\delta\theta^\star
=
-
\alpha
M(\theta^k)^{-1}
\nabla_\theta J(\theta^k).
$$

Ordinary gradient descent corresponds to $M(\theta^k)=I$. Newton's method corresponds to using $M(\theta^k)=H(\theta^k)$, with $\alpha=1$, when the Hessian is positive definite.

For a parameterized probability distribution, we would like the metric $M(\theta)$ to measure changes in the distribution $p_\theta$, not just changes in the raw parameter vector. Natural gradient descent chooses the Fisher information matrix

$$
\begin{equation}
\label{eq:fisher_information}
F(\theta)
=
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top
\right]
\end{equation}
$$

as the metric:

$$
M(\theta)
=
F(\theta).
$$

This gives

$$
\begin{equation}
\label{eq:natural_gradient_step}
\theta^{k+1}
=
\theta^k
-
\alpha
F(\theta^k)^{-1}
\nabla_\theta J(\theta^k).
\end{equation}
$$

This is the natural gradient update. Compared with ordinary gradient descent, the gradient is preconditioned by $F(\theta^k)^{-1}$, so the step is scaled according to the local geometry of the distribution family.

It remains to explain why the Fisher matrix is the right metric for a probability distribution and how to derive it. 
The key issue of a small Euclidean change in $\theta$ is that it does not necessarily correspond to a small change in the distribution $p_\theta(x)$.
The Fisher metric fixes this by measuring steps in distribution space instead of raw parameter space. A natural way to measure the difference between two probability distributions is the KL divergence. Let us consider a small perturbation of the parameter:

$$
\theta
\mapsto
\theta+\delta\theta.
$$

The KL divergence from the current distribution to the perturbed distribution is

$$
\begin{aligned}
D_{\mathrm{KL}}
\left(
p_\theta
\;\Vert\;
p_{\theta+\delta\theta}
\right)
&=
\int
p_\theta(x)
\log
\frac{
p_\theta(x)
}{
p_{\theta+\delta\theta}(x)
}
dx \\
&=
\mathbb{E}_{p_\theta}
\left[
\log p_\theta(x)
-
\log p_{\theta+\delta\theta}(x)
\right].
\end{aligned}
$$

Now expand $\log p_{\theta+\delta\theta}(x)$ around $\theta$ using a second-order Taylor expansion:

$$
\begin{aligned}
\log p_{\theta+\delta\theta}(x)
&\approx
\log p_\theta(x)
+
\nabla_\theta \log p_\theta(x)^\top \delta\theta
+
\frac{1}{2}
\delta\theta^\top
\nabla_\theta^2 \log p_\theta(x)
\delta\theta.
\end{aligned}
$$

Substituting this into the KL divergence gives

$$
\begin{aligned}
D_{\mathrm{KL}}
\left(
p_\theta
\;\Vert\;
p_{\theta+\delta\theta}
\right)
&\approx
-
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\right]^\top
\delta\theta 
-
\frac{1}{2}
\delta\theta^\top
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta^2 \log p_\theta(x)
\right]
\delta\theta.
\end{aligned}
$$

The first-order term vanishes because the expected score is zero. To see this, write the expectation as an integral:

$$
\begin{aligned}
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\right]
&=
\int
p_\theta(x)
\nabla_\theta \log p_\theta(x)dx \\
&=
\int
\nabla_\theta p_\theta(x)dx \\
&=
\nabla_\theta
\int
p_\theta(x)dx \\
&=
\nabla_\theta 1 \\
&=
0.
\end{aligned}
$$

Therefore, the leading term is second-order:

$$
D_{\mathrm{KL}}
\left(
p_\theta
\;\Vert\;
p_{\theta+\delta\theta}
\right)
\approx
-
\frac{1}{2}
\delta\theta^\top
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta^2 \log p_\theta(x)
\right]
\delta\theta.
$$

The negative expected Hessian of the log density is equal to the Fisher information matrix:

$$
-\mathbb{E}_{p_\theta}
\left[
\nabla_\theta^2 \log p_\theta(x)
\right]
=
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top
\right].
$$

One way to see this identity is to differentiate the expected score. From the previous calculation,

$$
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\right]
=
\int
p_\theta(x)
\nabla_\theta \log p_\theta(x)dx
=
0.
$$

Now differentiate this vector identity with respect to $\theta$. Since both $p_\theta(x)$ and $\nabla_\theta \log p_\theta(x)$ depend on $\theta$, the product rule gives

$$
\begin{aligned}
0
&=
\nabla_\theta
\int
p_\theta(x)
\nabla_\theta \log p_\theta(x)dx \\
&=
\int
\nabla_\theta p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top dx
+
\int
p_\theta(x)
\nabla_\theta^2 \log p_\theta(x)dx.
\end{aligned}
$$

The first term comes from differentiating the density $p_\theta(x)$, and the second term comes from differentiating the score $\nabla_\theta \log p_\theta(x)$. Using

$$
\nabla_\theta p_\theta(x)
=
p_\theta(x)
\nabla_\theta \log p_\theta(x),
$$

we obtain

$$
\begin{aligned}
0
&=
\int
p_\theta(x)
\nabla_\theta \log p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top dx \\
&\quad
+
\int
p_\theta(x)
\nabla_\theta^2 \log p_\theta(x)dx \\
&=
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top
\right]
+
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta^2 \log p_\theta(x)
\right].
\end{aligned}
$$

Therefore,

$$
-\mathbb{E}_{p_\theta}
\left[
\nabla_\theta^2 \log p_\theta(x)
\right]
=
\mathbb{E}_{p_\theta}
\left[
\nabla_\theta \log p_\theta(x)
\nabla_\theta \log p_\theta(x)^\top
\right]
=
F(\theta).
$$

Hence, for a small perturbation $\delta\theta$,

$$
D_{\mathrm{KL}}
\left(
p_\theta
\;\Vert\;
p_{\theta+\delta\theta}
\right)
=
\frac{1}{2}
\delta\theta^\top
F(\theta)
\delta\theta
+ o(\Vert \delta\theta \Vert^3).
$$

This shows why the Fisher matrix gives a local metric for the distribution family: it tells us how large a small parameter displacement $\delta\theta$ is in terms of the induced change in the probability distribution. In other words, natural gradient descent is what we obtain when the Euclidean step penalty in gradient descent is replaced by this local KL-based metric.

## Gaussian Mean Update

Let us now compute the natural gradient update for a simple Gaussian sampling distribution. Suppose the covariance is fixed and we only update the mean:

$$
p_\theta(x)
=
\mathcal{N}(\bar{x}, \Sigma),
\quad
\theta = \bar{x}.
$$

The log density is

$$
\log p_\theta(x)
=
-
\frac{1}{2}
\left(x-\bar{x}\right)^\top
\Sigma^{-1}
\left(x-\bar{x}\right)
+ \mathrm{const},
$$

where the constant does not depend on $\bar{x}$. The score with respect to the mean is

$$
\nabla_{\bar{x}}
\log p_\theta(x)
=
\Sigma^{-1}
\left(x-\bar{x}\right).
$$

Therefore, the ordinary gradient of the expected cost is

$$
\begin{aligned}
\nabla_{\bar{x}} J(\bar{x})
&=
\mathbb{E}_{p_\theta}
\left[
f(x)
\Sigma^{-1}
\left(x-\bar{x}\right)
\right] \\
&=
\Sigma^{-1}
\mathbb{E}_{p_\theta}
\left[
f(x)
\left(x-\bar{x}\right)
\right].
\end{aligned}
$$

The Fisher information matrix for the mean parameter is

$$
\begin{aligned}
F(\bar{x})
&=
\mathbb{E}_{p_\theta}
\left[
\Sigma^{-1}
\left(x-\bar{x}\right)
\left(x-\bar{x}\right)^\top
\Sigma^{-1}
\right] \\
&=
\Sigma^{-1}
\mathbb{E}_{p_\theta}
\left[
\left(x-\bar{x}\right)
\left(x-\bar{x}\right)^\top
\right]
\Sigma^{-1} \\
&=
\Sigma^{-1}
\Sigma
\Sigma^{-1} \\
&=
\Sigma^{-1}.
\end{aligned}
$$

Thus, the natural gradient is

$$
\begin{aligned}
F(\bar{x})^{-1}
\nabla_{\bar{x}}J(\bar{x})
&=
\Sigma
\Sigma^{-1}
\mathbb{E}_{p_\theta}
\left[
f(x)
\left(x-\bar{x}\right)
\right] \\
&=
\mathbb{E}_{p_\theta}
\left[
f(x)
\left(x-\bar{x}\right)
\right].
\end{aligned}
$$

The natural gradient descent update is therefore

$$
\begin{equation}
\label{eq:gaussian_mean_natural_gradient}
\bar{x}^{k+1}
=
\bar{x}^k
-
\alpha
\mathbb{E}_{p_{\theta^k}}
\left[
f(x)
\left(x-\bar{x}^k\right)
\right].
\end{equation}
$$

## Closing Remarks

In this post, we started from the parameterized distribution objective

$$
J(\theta)
=
\mathbb{E}_{p_\theta}
\left[
f(x)
\right],
$$

and derived a natural gradient update for the sampling distribution. The main message is that once the decision variable is a probability distribution, the geometry of the update should come from the distribution family rather than from the raw Euclidean coordinates of $\theta$.

This point of view gives another way to understand sampling-based optimization. In the next post, we will connect this natural-gradient perspective back to the MPPI update introduced earlier, where samples are reweighted exponentially according to their costs. We will also start looking at algorithms such as CMA-ES, which adapt not only the mean of the sampling distribution but also its covariance, making the search distribution itself more responsive to the local structure of the cost landscape.


This tutorial was written with the assistance of ChatGPT-5.

<!-- With a baseline $b$, this can be written as

$$
\begin{equation}
\label{eq:gaussian_mean_natural_gradient_baseline}
\bar{x}^{k+1}
=
\bar{x}^k
-
\alpha
\mathbb{E}_{p_{\theta^k}}
\left[
\left(f(x)-b\right)
\left(x-\bar{x}^k\right)
\right].
\end{equation}
$$

Using samples

$$
x_i
=
\bar{x}^k
+
\epsilon_i,
\quad
\epsilon_i
\sim
\mathcal{N}(0,\Sigma),
$$

we obtain the Monte Carlo update

$$
\bar{x}^{k+1}
\approx
\bar{x}^k
-
\alpha
\frac{1}{N}
\sum_{i=1}^{N}
\left(f_i-b\right)
\epsilon_i.
$$

Equivalently,

$$
\begin{equation}
\label{eq:natural_gradient_sampling_update}
\bar{x}^{k+1}
\approx
\bar{x}^k
+
\alpha
\frac{1}{N}
\sum_{i=1}^{N}
\left(b-f_i\right)
\epsilon_i.
\end{equation}
$$

This formula has an intuitive interpretation. Samples with cost lower than the baseline have $b-f_i > 0$, so they pull the mean toward themselves. Samples with cost higher than the baseline have $b-f_i < 0$, so they push the mean away. -->

<!-- <div class="algorithm" markdown="1">
**Algorithm 1:** Natural Gradient Sampling Update for a Gaussian Mean
<hr style="margin: 0.5em 0;">

Given current distribution $p_{\theta^k}(x) = \mathcal{N}(\bar{x}^k, \Sigma)$  
Sample $\epsilon_i \sim \mathcal{N}(0,\Sigma)$ for $i=1,\ldots,N$  
Set $x_i \gets \bar{x}^k + \epsilon_i$  
Evaluate $f_i \gets f(x_i)$  
Choose a baseline $b$, for example

$$
b \gets \frac{1}{N}\sum_{i=1}^{N} f_i
$$

Update

$$
\bar{x}^{k+1}
\gets
\bar{x}^k
+
\alpha
\frac{1}{N}
\sum_{i=1}^{N}
\left(b-f_i\right)
\epsilon_i
$$
</div> -->

<!-- ## Connection to Exponential Weighted Updates

Now let us connect this natural gradient update to the exponential weighted update from the previous post.

For fixed covariance Gaussian sampling, the exponential weighted update is

$$
\bar{x}^{k+1}
=
\frac{
\mathbb{E}_{p_{\theta^k}}
\left[
x
\exp\left(-\frac{1}{\lambda}f(x)\right)
\right]
}{
\mathbb{E}_{p_{\theta^k}}
\left[
\exp\left(-\frac{1}{\lambda}f(x)\right)
\right]
}.
$$

Let

$$
\beta
=
\frac{1}{\lambda}.
$$

When $\lambda$ is large, $\beta$ is small, and we can use the first-order approximation

$$
\exp(-\beta f(x))
\approx
1-\beta f(x).
$$

The numerator becomes

$$
\begin{aligned}
\mathbb{E}
\left[
x
\exp(-\beta f(x))
\right]
&\approx
\mathbb{E}
\left[
x
\left(1-\beta f(x)\right)
\right] \\
&=
\bar{x}^k
-
\beta
\mathbb{E}
\left[
x f(x)
\right].
\end{aligned}
$$

The denominator becomes

$$
\begin{aligned}
\mathbb{E}
\left[
\exp(-\beta f(x))
\right]
&\approx
\mathbb{E}
\left[
1-\beta f(x)
\right] \\
&=
1
-
\beta
\mathbb{E}
\left[
f(x)
\right].
\end{aligned}
$$

Using

$$
\frac{1}{1-\beta \mathbb{E}[f(x)]}
\approx
1+\beta \mathbb{E}[f(x)],
$$

we obtain

$$
\begin{aligned}
\bar{x}^{k+1}
&\approx
\left(
\bar{x}^k
-
\beta
\mathbb{E}
\left[
x f(x)
\right]
\right)
\left(
1
+
\beta
\mathbb{E}
\left[
f(x)
\right]
\right) \\
&=
\bar{x}^k
-
\beta
\left(
\mathbb{E}
\left[
x f(x)
\right]
-
\bar{x}^k
\mathbb{E}
\left[
f(x)
\right]
\right) \\
&=
\bar{x}^k
-
\beta
\mathbb{E}
\left[
\left(f(x)-\mathbb{E}[f(x)]\right)
\left(x-\bar{x}^k\right)
\right].
\end{aligned}
$$

This is exactly the Gaussian mean natural gradient update with the baseline

$$
b
=
\mathbb{E}_{p_{\theta^k}}
\left[
f(x)
\right],
$$

and step size

$$
\alpha
=
\beta
=
\frac{1}{\lambda}.
$$

Therefore, the exponential weighted update can be viewed as a finite-step distribution update, while the natural gradient update is its small-step approximation.

## Comparing the Two Views

The natural gradient view and the KL-regularized view are two closely related ways to describe sampling-based optimization.

In the KL-regularized view, we solve a proximal problem in the space of probability distributions:

$$
p^\star
=
\underset{q}{\operatorname{argmin}}\;
\mathbb{E}_{q}
\left[
f(x)
\right]
+
\lambda
D_{\mathrm{KL}}
\left(
q
\;\Vert\;
p_{\theta^k}
\right).
$$

This gives the exponential reweighting formula

$$
p^\star(x)
\propto
p_{\theta^k}(x)
\exp\left(-\frac{1}{\lambda}f(x)\right).
$$

In the natural gradient view, we directly optimize

$$
J(\theta)
=
\mathbb{E}_{p_\theta}
\left[
f(x)
\right],
$$

but we measure local movement using the Fisher metric, which is the second-order approximation of KL divergence.

Thus, both views are based on the same geometric idea: probability distributions should not be updated using the raw Euclidean geometry of their parameters. They should be updated according to the geometry of the distribution space.

The difference is that the exponential weighted update takes a finite KL-regularized step and then projects back to the sampling family, while natural gradient descent takes an infinitesimal or local KL-regularized step directly in parameter space.

## Practical Remarks

The natural gradient update derived above has signed weights:

$$
b-f_i.
$$

Low-cost samples attract the mean, and high-cost samples repel it. In contrast, the exponential weighted update uses positive normalized weights:

$$
w_i
\propto
\exp\left(-\frac{1}{\lambda}f_i\right).
$$

This difference matters in implementation. Exponential weights are often stable because the update is an average of sampled candidates, but the weights can collapse onto a few samples when the temperature is small or the cost variance is large. Natural gradient updates avoid normalized exponential weights, but they require a step size and can be sensitive to cost scaling. In practice, baselines, cost normalization, step-size tuning, and covariance adaptation are all important.

The derivation in this post only updated the Gaussian mean while keeping the covariance fixed. One can also update the covariance or use richer distribution families, but then the Fisher matrix and the parameterization of positive definite matrices must be handled carefully.

The main takeaway is simple:

$$
\text{exponential weighted sampling}
\quad
\approx
\quad
\text{natural gradient descent on }
\mathbb{E}_{p_\theta}[f(x)]
$$

when the distribution update is small. -->


<!-- ## References

{% bibliography --cited %} -->
