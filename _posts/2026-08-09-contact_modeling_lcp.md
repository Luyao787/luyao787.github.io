---
layout: post
title: "Contact Dynamics: LCP"
date: 2026-08-09
description: Deriving a linear complementarity formulation for frictional contact dynamics
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
.post-content {
  counter-reset: contact-figure;
}
.post-content .contact-figure {
  counter-increment: contact-figure;
  text-align: center;
}
.post-content .contact-figure img {
  height: auto;
  max-width: 60%;
}
.post-content .contact-figure figcaption::before {
  content: "Figure " counter(contact-figure) ". ";
  font-weight: 600;
}
</style>

In the previous [blog post]({% post_url 2025-06-01-contact_modeling %}), we started from the *Signorini condition* and the *maximum dissipation principle* and obtained the nonlinear cone complementarity formulation

$$
K_\mu \ni \lambda \perp
\begin{bmatrix}
    c_T \\
    \dfrac{1}{\Delta t}\phi(q_t)+c_N+\mu\Vert c_T\Vert_2
\end{bmatrix}
\in K_\mu^*.
$$

In this post, we take a different route: we approximate the circular friction cone by a polyhedral cone and retain the multiplier associated with the friction bound as an explicit variable. The resulting optimality conditions are affine in the unknowns and can therefore be written as a Linear Complementarity Problem (LCP).

<figure class="contact-figure">
  <img src="/assets/img/Contact/lcp_friction_pyramid.png" width="760" alt="Circular Coulomb friction cone, an inscribed four-direction friction pyramid, and its projection onto the tangent plane"/>
  <figcaption class="caption">The circular Coulomb cone and an inscribed four-direction friction pyramid. Its vertices lie on the circular cross-section, while the dashed quadrilateral is their projection onto the tangent plane. The vectors $d_1,\dots,d_4$ generate the polyhedral tangential impulse set used by the LCP.</figcaption>
</figure>

The construction follows the velocity-impulse time-stepping formulation used by Anitescu and Hart {% cite anitescu2004constraint %}. We first derive the contact conditions for one contact and then assemble the full LCP.

## Recap: Signorini and Maximum Dissipation

To simplify the notation, we write $v:=v_{t+1}$ and define the discrete normal gap velocity

$$
g_N := \frac{1}{\Delta t}\phi(q_t)+c_N.
$$

The discretized Signorini condition is

$$
\begin{equation} \label{eq:lcp_signorini}
0\leq\lambda_N\perp g_N\geq0.
\end{equation}
$$

Given a normal impulse $\lambda_N$, the maximum dissipation principle selects the tangential impulse from the Coulomb disk:

$$
\begin{aligned}
\underset{\gamma_T\in\mathbb{R}^2}{\operatorname{minimize}}\quad
    & \gamma_T^\top c_T \\
\operatorname{subject\ to}\quad
    & \Vert\gamma_T\Vert_2\leq\mu\lambda_N.
\end{aligned}
$$

<!-- For the exact circular cone, the multiplier associated with the friction bound can be eliminated analytically. When sliding occurs, this multiplier is $\Vert c_T\Vert_2$, which produces the nonlinear term $\mu\Vert c_T\Vert_2$ in the compact formulation above. An LCP cannot contain this norm because an LCP residual must be affine in its unknown variables. -->

## Polyhedral Approximation of the Friction Cone

Choose $m$ unit vectors in the tangent plane and collect them as

$$
D :=
\begin{bmatrix}
    d_1 & d_2 & \cdots & d_m
\end{bmatrix}
\in\mathbb{R}^{2\times m}.
$$

We assume that the directions are balanced: for every $d_k$, the opposite direction $-d_k$ is also included. The circular cross-section of the friction cone is then approximated by the convex hull of these directions. We represent the tangential impulse as

$$
\begin{equation} \label{eq:tangential_decomposition}
\lambda_T=D\alpha,
\qquad
\alpha\geq0,
\end{equation}
$$

where $\alpha\in\mathbb{R}^{m}$ contains the nonnegative impulse magnitudes along the selected directions. Let

$$
e:=\mathbf{1}_{m}
=
\begin{bmatrix}
1 & \cdots & 1
\end{bmatrix}^{\top}
\in\mathbb{R}^{m}.
$$

That is, $e$ is the $m$-dimensional column vector whose entries are all one. Consequently, $e^\top\alpha=\sum_{k=1}^m\alpha_k$ is the sum of all impulse coefficients.

For a fixed normal impulse $\lambda_N$, the polygonal cross-section is 

$$
\mu\lambda_N\operatorname{conv}\{d_1,\dots,d_m\}.
$$ 

Because the directions are balanced, the origin lies in this polygon. Hence, any admissible tangential impulse can be written as

$$
\lambda_T
=r\sum_{k=1}^m\theta_kd_k,
\qquad
0\leq r\leq\mu\lambda_N,
\qquad
\theta_k\geq0,
\qquad
\sum_{k=1}^m\theta_k=1.
$$

Define $\alpha_k:=r\theta_k$. Then

$$
\lambda_T=D\alpha,
\qquad
e^\top\alpha
=\sum_{k=1}^m\alpha_k
=r
\leq\mu\lambda_N.
$$

Consequently, the exact Coulomb cone is approximated by

$$
K_{\mu}^{\mathrm{poly}}
:=
\left\{
(D\alpha,\lambda_N)
\;\middle|\;
\alpha\geq0,\;
e^\top\alpha\leq\mu\lambda_N
\right\}.
$$


## Maximum Dissipation as a Linear Program

After the polyhedral approximation, the maximum dissipation problem becomes

$$
\begin{aligned}
\underset{\gamma\in\mathbb{R}^{m}}{\operatorname{minimize}}\quad
    & \gamma^\top D^\top c_T \\
\operatorname{subject\ to}\quad
    & \gamma\geq0, \\
    & e^\top\gamma\leq\mu\lambda_N.
\end{aligned}
$$

This is a linear program. Let $\beta\geq0$ be the scalar multiplier associated with the friction bound, and let $\rho\in\mathbb{R}^m_{\geq0}$ be the vector multiplier associated with $\gamma\geq0$. Its Lagrangian is

$$
\mathcal{L}(\gamma,\beta,\rho)
=\gamma^\top D^\top c_T
+\beta\left(e^\top\gamma-\mu\lambda_N\right)
-\rho^\top\gamma.
$$

At the optimum $\gamma=\alpha$, the KKT conditions are

$$
\begin{aligned}
\alpha\geq0,\quad e^\top\alpha\leq\mu\lambda_N
&\qquad\text{(primal feasibility)}\\
\rho\geq0,\quad\beta\geq0
&\qquad\text{(dual feasibility)}\\
D^\top c_T+e\beta-\rho=0
&\qquad\text{(stationarity)}\\
\alpha^\top\rho=0,\quad
\beta\left(\mu\lambda_N-e^\top\alpha\right)=0
&\qquad\text{(complementary slackness)}.
\end{aligned}
$$

Stationarity gives $\rho=D^\top c_T+e\beta$. Substituting this expression into $\alpha\geq0$, $\rho\geq0$, and $\alpha^\top\rho=0$ eliminates $\rho$ and gives the first complementarity pair:

$$
\begin{equation} \label{eq:direction_complementarity}
0\leq\alpha
\perp
D^\top c_T+e\beta
\geq0.
\end{equation}
$$

Primal feasibility, dual feasibility, and complementary slackness for the friction bound give the second pair:

$$
\begin{equation} \label{eq:friction_bound_complementarity}
0\leq\beta
\perp
\mu\lambda_N-e^\top\alpha
\geq0.
\end{equation}
$$

Equations $\eqref{eq:direction_complementarity}$ and $\eqref{eq:friction_bound_complementarity}$ are the polyhedral version of maximum dissipation.

<!-- ### The role of the additional variable

The variable $\beta$ is not a new physical impulse. It is the multiplier of the friction bound. From $\eqref{eq:direction_complementarity}$,

$$
\beta\geq-d_k^\top c_T,
\qquad k=1,\dots,m.
$$

Whenever $\alpha_k>0$, complementarity forces equality for that direction. During sliding,

$$
\beta
=\max_k\left(-d_k^\top c_T\right).
$$

This maximum is the polyhedral dual norm induced by the selected friction directions. As the directions become dense on the unit circle, it approaches $\Vert c_T\Vert_2$.

We could eliminate $\beta$ and substitute this maximum explicitly, just as the previous post eliminated the exact-cone multiplier and obtained $\Vert c_T\Vert_2$. Doing so, however, would introduce a piecewise-linear maximum into the residual. Keeping $\beta$ as an independent variable is a lifting that makes every complementarity residual affine. -->

Combining Signorini with the KKT conditions of maximum dissipation gives

$$
\begin{equation} \label{eq:single_contact_complementarity}
\begin{aligned}
0\leq\lambda_N
&\perp
\frac{1}{\Delta t}\phi(q_t)+c_N
\geq0, \\
0\leq\alpha
&\perp
D^\top c_T+e\beta
\geq0, \\
0\leq\beta
&\perp
\mu\lambda_N-e^\top\alpha
\geq0, \\
\lambda_T&=D\alpha.
\end{aligned}
\end{equation}
$$

At this point the contact conditions are componentwise complementarity conditions, but they are not yet a closed LCP: the contact velocities $c_N$ and $c_T$ still depend on the unknown generalized velocity. We next use the discrete equations of motion to express those velocities in terms of the contact variables.

## Coupling the Contacts to Multibody Dynamics

We now restore the contact index $i$. Let $N_c$ be the number of candidate contacts and let

$$
m:=\sum_{i=1}^{N_c}m_i.
$$

Let $v$ denote the generalized velocity of the
multibody system. Multiplying it by the contact
Jacobians gives the relative contact velocities, so we stack the normal and
tangential components as

$$
\boldsymbol{c}_N=J_Nv,
\quad
\boldsymbol{c}_T=J_Tv,
$$

where

$$
J_N\in\mathbb{R}^{N_c\times n_v},
\quad
J_T\in\mathbb{R}^{2N_c\times n_v}.
$$

We define the block matrices as follows:

$$
\begin{aligned}
D&:=\operatorname{blkdiag}(D_1,\dots,D_{N_c})
    \in\mathbb{R}^{2N_c\times m},\\
E&:=\operatorname{blkdiag}(e_1,\dots,e_{N_c})
    \in\mathbb{R}^{m\times N_c},\\
\mathrm{M}_{\mu}&:=\operatorname{diag}(\mu_1,\dots,\mu_{N_c}).
\end{aligned}
$$

Here each column of $E$ contains the vector of ones associated with one contact. With

$$
\boldsymbol{\lambda}_N
=
\begin{bmatrix}
\lambda_{N,1}&\cdots&\lambda_{N,N_c}
\end{bmatrix}^\top,
$$

and the stacked vector $\boldsymbol{\alpha}$, the generalized contact impulse is

$$
J_N^\top\boldsymbol{\lambda}_N
+J_T^\top D\boldsymbol{\alpha}.
$$

Recall the discrete equation of motion

$$
M(v_{t+1}-v_t)
=\Delta t\left(\tau_t-h(q_t,v_t)\right)
+J_N^\top\boldsymbol{\lambda}_N
+J_T^\top D\boldsymbol{\alpha}.
$$

Collecting the terms that do not involve contact defines the free-motion
velocity

$$
v^f
:=
v_t+M^{-1}\Delta t\left(\tau_t-h(q_t,v_t)\right).
$$

Thus, the discrete dynamics can be written as

$$
\begin{equation} \label{eq:lcp_dynamics}
M(v_{t+1}-v^f)
=J_N^\top\boldsymbol{\lambda}_N
+J_T^\top D\boldsymbol{\alpha}.
\end{equation}
$$

Define the complementarity residuals

$$
\begin{aligned}
\boldsymbol{w}_N
&:=J_Nv_{t+1}+\frac{\boldsymbol{\phi}(q_t)}{\Delta t},\\
\boldsymbol{w}_{\alpha}
&:=D^\top J_Tv_{t+1}+E\boldsymbol{\beta},\\
\boldsymbol{w}_{\beta}
&:=\mathrm{M}_{\mu}\boldsymbol{\lambda}_N-E^\top\boldsymbol{\alpha}.
\end{aligned}
$$

The dynamics and contact conditions are therefore

$$
\begin{equation} \label{eq:mixed_lcp}
\begin{bmatrix}
M & -J_N^\top & -J_T^\top D & 0\\
J_N & 0 & 0 & 0\\
D^\top J_T & 0 & 0 & E\\
0 & \mathrm{M}_{\mu} & -E^\top & 0
\end{bmatrix}
\begin{bmatrix}
v_{t+1}\\
\boldsymbol{\lambda}_N\\
\boldsymbol{\alpha}\\
\boldsymbol{\beta}
\end{bmatrix}
+
\begin{bmatrix}
-Mv^f\\
\boldsymbol{\phi}(q_t)/\Delta t\\
0\\
0
\end{bmatrix}
=
\begin{bmatrix}
0\\
\boldsymbol{w}_N\\
\boldsymbol{w}_{\alpha}\\
\boldsymbol{w}_{\beta}
\end{bmatrix}.
\end{equation}
$$

The first block row is an equality. The remaining rows are the multi-contact version of the conditions in $\eqref{eq:single_contact_complementarity}$, and the remaining variables and residuals satisfy

$$
\begin{bmatrix}
\boldsymbol{\lambda}_N\\
\boldsymbol{\alpha}\\
\boldsymbol{\beta}
\end{bmatrix}
\geq0,
\qquad
\begin{bmatrix}
\boldsymbol{w}_N\\
\boldsymbol{w}_{\alpha}\\
\boldsymbol{w}_{\beta}
\end{bmatrix}
\geq0,
$$

and

$$
\boldsymbol{\lambda}_N^\top\boldsymbol{w}_N
+\boldsymbol{\alpha}^\top\boldsymbol{w}_{\alpha}
+\boldsymbol{\beta}^\top\boldsymbol{w}_{\beta}
=0.
$$

For a standard LCP, all unknowns must be nonnegative complementarity variables. Since $M$ is positive definite, we can eliminate $v$ from $\eqref{eq:lcp_dynamics}$:

$$
v_{t+1}
=v^f+M^{-1}
\left(
J_N^\top\boldsymbol{\lambda}_N
+J_T^\top D\boldsymbol{\alpha}
\right).
$$

Substituting this expression into the contact residuals gives

$$
\begin{aligned}
\boldsymbol{w}_N
={}&J_NM^{-1}J_N^\top\boldsymbol{\lambda}_N
+J_NM^{-1}J_T^\top D\boldsymbol{\alpha}
+J_Nv^f
+\frac{\boldsymbol{\phi}(q_t)}{\Delta t},\\
\boldsymbol{w}_{\alpha}
={}&D^\top J_TM^{-1}J_N^\top\boldsymbol{\lambda}_N
+D^\top J_TM^{-1}J_T^\top D\boldsymbol{\alpha}
+E\boldsymbol{\beta}
+D^\top J_Tv^f,\\
\boldsymbol{w}_{\beta}
={}&\mathrm{M}_{\mu}\boldsymbol{\lambda}_N
-E^\top\boldsymbol{\alpha}.
\end{aligned}
$$

Define

$$
\boldsymbol{z}:=
\begin{bmatrix}
\boldsymbol{\lambda}_N\\
\boldsymbol{\alpha}\\
\boldsymbol{\beta}
\end{bmatrix}.
$$

The standard LCP is

$$
\begin{equation} \label{eq:standard_lcp}
0\leq \boldsymbol{z} \perp A \boldsymbol{z} + b\geq0,
\end{equation}
$$

with

$$
\begin{equation} \label{eq:lcp_matrix}
A=
\begin{bmatrix}
J_NM^{-1}J_N^\top
    & J_NM^{-1}J_T^\top D
    & 0\\
D^\top J_TM^{-1}J_N^\top
    & D^\top J_TM^{-1}J_T^\top D
    & E\\
\mathrm{M}_{\mu}
    & -E^\top
    & 0
\end{bmatrix},
\qquad
b=
\begin{bmatrix}
J_Nv^f+\boldsymbol{\phi}(q_t)/\Delta t\\
D^\top J_Tv^f\\
0
\end{bmatrix}.
\end{equation}
$$

This is now in the form expected by a standard LCP solver: find $\boldsymbol{z}\geq0$ such that $A\boldsymbol{z}+b\geq0$ and $\boldsymbol{z}^\top \left( A\boldsymbol{z}+b \right) =0$.

<!-- The upper-left block of $A$ is a Delassus matrix expressed in polyhedral contact coordinates. Indeed, if we define

$$
\bar{J}:=
\begin{bmatrix}
J_N\\
D^\top J_T
\end{bmatrix},
$$

then

$$
\bar{G}:=\bar{J}M^{-1}\bar{J}^\top
=
\begin{bmatrix}
J_NM^{-1}J_N^\top
    & J_NM^{-1}J_T^\top D\\
D^\top J_TM^{-1}J_N^\top
    & D^\top J_TM^{-1}J_T^\top D
\end{bmatrix}.
$$

Thus, the derivation uses exactly the same velocity-to-impulse elimination as the Delassus formulation; only the tangential coordinates have changed from two Cartesian components to nonnegative coefficients along the friction-pyramid directions. -->

<!-- ## Interpretation of the Contact Modes

The LCP recovers the familiar contact modes directly from complementarity.

- **Take-off.** If $w_{N,i}>0$, then $\lambda_{N,i}=0$. The friction bound $\mu_i\lambda_{N,i}-e_i^\top\alpha_i\geq0$ then forces $\alpha_i=0$, so the entire contact impulse vanishes.

- **Sticking.** If $\lambda_{N,i}>0$, then $w_{N,i}=0$. When $c_{T,i}=0$, maximum dissipation permits an impulse strictly inside the polyhedral friction cone. In this case the friction-bound multiplier can be $\beta_i=0$.

- **Sliding.** If $\beta_i>0$, then $\eqref{eq:friction_bound_complementarity}$ forces

  $$
  e_i^\top\alpha_i=\mu_i\lambda_{N,i},
  $$

  so the tangential impulse lies on the boundary of the friction pyramid. Equation $\eqref{eq:direction_complementarity}$ selects the face or edge that opposes the sliding velocity and maximizes dissipation. -->

<!-- Unlike the [CCP relaxation discussed separately]({% post_url 2025-06-23-contact_modeling_ccp %}), this construction keeps the discrete Signorini pair

$$
0\leq\lambda_N
\perp
\frac{\phi(q_t)}{\Delta t}+c_N
\geq0
$$

explicit. Its approximation is instead geometric: the circular Coulomb cone is replaced by a friction pyramid. Increasing the number of tangential directions improves the friction approximation but enlarges the LCP through additional entries in $\boldsymbol{\alpha}$. -->

## References

{% bibliography --cited %}
