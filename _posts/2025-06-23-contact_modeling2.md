---
layout: post
title: Contact Dynamics
date: 2025-06-23
description: Cone Complementarity Problem formulation for contact modeling
tags:
categories:
  - Simulation
related_posts: false
---

<style>
.post-content p:not([align]), .post-content li {
  text-align: justify;
  text-justify: inter-word;
}
.post-content p[align="center"] {
  text-align: center;
}
</style>

In the previous [blog post]({% post_url 2025-06-01-contact_modeling %}), we introduced how to represent contact dynamics using nonlinear complementarity problems (NCPs). 
To get around the computational challenges, we relaxed the complementarity constraint and reformulated the problem as a cone complementarity problem (CCP). 
In this post, we delve deeper into the CCP formulation and present two types of corresponding convex optimization problems.
We begin by recapping the CCP for the $i$-*th* contact:

$$
\begin{equation} \label{eq:CCP}
    K_{\mu, i} \in \lambda_i \perp \left( c_i - \hat{c}_i \right) \in K^*_{\mu, i},
\end{equation}
$$

where $K_{\mu, i}$ is the friction cone, $K_{\mu, i}^*$ is the corresponding dual cone, $\lambda_i \in \mathbb{R}^3$ represents the contact impulse, and $c_i \in \mathbb{R}^3$ is the contact velocity, and $\hat{c}_i \in \mathbb{R}^3$ is the so-called stabilization velocity expressed as 

$$
\hat{c}_i = \begin{bmatrix}
    0 \\ 0 \\ -\frac{\phi_i(q)}{\Delta t}
\end{bmatrix},
$$

where $\phi_i(q)$ is the signed distance at the previous step, and $\Delta t$ is discretization step. 
Let $N_c$ denote the number of contacts. 
Next, we derive two optimization problems whose KKT conditions are equivalent to the problem in \eqref{eq:CCP}. 

### Optimization on the dual

We aim to formulate an optimization problem in terms of the contact impulse $\lambda$, which serves as a dual variable in constrained multibody dynamics.
To achieve this, we first replace the contact velocity $c$ with the contact impulse $\lambda$. 
To achieve this, we revisit the discretized multibody dynamics:

$$
M(q_t)(v_{t+1} - v_t) = \Delta t\left(\tau_t - h(q_t, v_t)\right) + J(q_t)^\top \boldsymbol{\lambda},
$$

By rearranging the equation, we derive the expression for $v_{t+1}$ in terms of $\lambda$:

$$
\begin{align}
M v_{t+1} &= J^\top \boldsymbol{\lambda} + M v_t + \Delta t\left(\tau_t - h \right) \label{eq:dynamics} \\
v_{t+1} &= M^{-1} J^\top \boldsymbol{\lambda} + \underbrace{v_t + M^{-1} \Delta t\left(\tau_t - h \right)}_{v^f} \nonumber
\end{align}
$$

where the inertia matrix $M$ is always invertiable, and $v^f$ is the free motion velocity that the system would have in the absence of contact constraints. We then compute the contact velocity by multiplying both sides by $J$:

$$
\begin{equation} \label{eq:c_lambda}  
    \begin{aligned}
        \boldsymbol{c} &= J v \\ 
        &= J M^{-1} J^\top \boldsymbol{\lambda} + J v^f \\
        &= G \boldsymbol{\lambda} + J v^f, 
    \end{aligned}
\end{equation}
$$

where we ignore the subscript, $\boldsymbol{c} = [c_1^\top, \dots, c_{N_c}^\top]$ is the collection of all contact velocities, and $G := J M^{-1} J^\top$ is the so-called Delassus matrix. 
Substituting \eqref{eq:c_lambda} into \eqref{eq:CCP}, we rewrite the CCP as

$$
\begin{equation} \label{eq:CCP_dual}
    \begin{aligned}
        G \boldsymbol{\lambda} + g &= \boldsymbol{c} - \hat{\boldsymbol{c}}  \\
        \lambda_i &\in K_{\mu, i},\; i = 1, \dots, N_c \\
        c_i - \hat{c}_i &\in K_{\mu, i}^*,\; i = 1, \dots, N_c \\
         \boldsymbol{\lambda}^\top (\boldsymbol{c} - \hat{\boldsymbol{c}})  &= 0,
    \end{aligned}
\end{equation}
$$

where $g = J v^f + \hat{\boldsymbol{c}}$. In fact, the conditions above can be viewed as the KKT conditions of the following optimization problems in terms of $\lambda$:

$$
\begin{aligned}
    \min_{\boldsymbol{\lambda}}\; &\dfrac{1}{2} \boldsymbol{\lambda}^\top G \boldsymbol{\lambda} + g^\top \boldsymbol{\lambda} \\
    \text{s.t.}\; & \boldsymbol{\lambda} \in \mathcal{K} := K_{\mu, 1} \times \dots \times K_{\mu, N_c}.
\end{aligned}
$$

The Lagrangian function is given by

$$
\mathcal{L}(\boldsymbol{\lambda}, \boldsymbol{y}) =  \dfrac{1}{2} \boldsymbol{\lambda}^\top G \boldsymbol{\lambda} + g^\top \boldsymbol{\lambda} - \boldsymbol{y}^\top \boldsymbol{\lambda}, 
$$

where $\boldsymbol{y}$ is the Lagrange multiplier vector. The corresponding KKT conditions are then

$$
\begin{aligned}
    G \boldsymbol{\lambda} + g &= \boldsymbol{y} &&\text{(stationarity)} \\
    \boldsymbol{\lambda} &\in \mathcal{K} &&\text{(primal feasibility)} \\
    \boldsymbol{y} &\in \mathcal{K}^* &&\text{(dual feasibility)} \\
    \boldsymbol{y}^\top \boldsymbol{\lambda} &= 0 &&\text{(complementary slackness)}
\end{aligned}
$$

This formulation is equivalent to \eqref{eq:CCP_dual} when $\boldsymbol{c} - \hat{\boldsymbol{c}}$ is interpreted as the Lagrange multiplier vector. 

### Optimization on the primal

We now introduce an alternative optimization problem, where the decision variable is the generalized velocity $v$.

From \eqref{eq:dynamics}, we have

$$
M v = J^\top \boldsymbol{\lambda} + M v^f. 
$$

We then reformulate \eqref{eq:CCP} as

$$
\begin{equation}
    \begin{aligned}
        M (v - v^f) - J^\top \boldsymbol{\lambda} &= 0 &&\text{(stationarity)} \\
        J v - \hat{\boldsymbol{c}} &\in \mathcal{K}^* &&\text{(primal feasibility)} \\
        \boldsymbol{\lambda}  &\in \mathcal{K} &&\text{(dual feasibility)} \\
        \boldsymbol{\lambda}^\top (J v - \hat{\boldsymbol{c}}) &= 0 &&\text{(complementary slackness)}
    \end{aligned}
\end{equation}
$$

Again, the conditions above corresponds an convex optimization problem:

$$
\begin{aligned}
    \min_{v}\; &\dfrac{1}{2} (v - v^f)^\top M (v - v^f) \\
    \text{s.t.}\; & J v - \hat{\boldsymbol{c}} \in \mathcal{K}^*.
\end{aligned}
$$

Its Lagrangian function is given by

$$
\mathcal{L}(v, \boldsymbol{\lambda}) = \dfrac{1}{2} (v - v^f)^\top M (v - v^f) - \boldsymbol{\lambda}^\top (J v - \hat{\boldsymbol{c}}),
$$

where the vector of contact impulses $\boldsymbol{\lambda}$ coincides with the dual variable of the optimization problem.
