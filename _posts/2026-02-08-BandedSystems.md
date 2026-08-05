---
layout: post
title: Parallel Methods for Solving Block Banded Systems
date: 2026-02-08
description:
tags:
categories:
  - Optimal Control
related_posts: false
---

Let us consider the following block banded system:

$$
\begin{equation}
\label{eq:banded-system}
    \begin{bmatrix}
    D_{1} & F_{1} & & & & \\
    E_{2} & D_{2} & F_{2} & & &\\
    & E_{3} & D_{3} & F_{3} & &\\
    & & \ddots & \ddots & \ddots &\\
    & & & E_{N-2} & D_{N-1} & F_{N-1}\\
    & & & & E_{N} & D_{N}
    \end{bmatrix}
    \begin{bmatrix}
    x_{1}\\
    x_{2}\\
    x_{3}\\
    \vdots\\
    x_{N-1}\\
    x_{N}
    \end{bmatrix}
    =
    \begin{bmatrix}
    g_{1}\\
    g_{2}\\
    g_{3}\\
    \vdots\\
    g_{N-1}\\
    g_{N}
    \end{bmatrix},
\end{equation}
$$

where $D_k \in \mathbb{R}^{n \times n}$ are positive definite matrices, and $E_k, F_k \in \mathbb{R}^{n \times n}$ are off-diagonal blocks satisfying $E_{k+1} = F_{k}^\top$. Such systems are often solved in MPC problems {% cite jordana2025SQP Wang2010FastMPC %}. We begin by introducing a serial method. At the initial stage, we have 

$$
\begin{aligned}
    D_1 x_1 + F_1 x_2 &= g_1 \\ 
    x_1 &= D_1^{-1} \left( g_1 - F_1 x_2 \right) \\
    &= L_1^{-\top} L_1^{-1} \left( g_1 - F_1 x_2 \right) \\
    &= L_1^{-\top} \left( L_1^{-1} g_1 - L_1^{-1}F_1 x_2 \right),
\end{aligned}
$$

where $L_1$ is a lower triangular matrix obtained from the cholesky factorization, such that $D_1=L_1L_1^\top$. For notational simplicity, we define

$$
\begin{aligned}
    y_1 &:= L_1^{-1} g_1 \\
    Y_1 &:= F_1^\top L_1^{-\top} = E_2 L_1^{-\top}.
\end{aligned}
$$

We then move to the second stage and obtain

$$
\begin{aligned}
    E_2 x_1 + D_2 x_2 + F_2 x_3 &= g_2 \\ 
    E_2 L_1^{-\top} \left( y_1 - Y_1^\top x_2 \right) + D_2 x_2 + F_2 x_3 &= g_2 \\
    Y_1 \left( y_1 - Y_1^\top x_2 \right) + D_2 x_2 + F_2 x_3 &= g_2 \\
    \left( D_2 -  Y_1 Y_1^\top \right) x_2 &= g_2 - Y_1 y_1 - F_2 x_3 \\
    x_2 &= L_2^{-\top} \left( \underbrace{L_2^{-1} g_2 - L_2^{-1} Y_1 y_1}_{y_2} - \underbrace{L_2^{-1}F_2}_{Y_2^\top} x_3 \right),
\end{aligned}
$$

where $D_2 -  Y_1 Y_1^\top=L_2L_2^\top$. At stage $k$, we have

$$
\begin{aligned}
    x_k = L_k^{-\top} \left( y_k - Y_k^{\top} x_{k+1} \right), 
\end{aligned}
$$

where

$$
\begin{aligned}
    L_k L_k^{\top} &= D_k - Y_{k-1}Y_{k-1}^\top \\
    Y_k &= E_{k+1} L_k^{-\top}\\
    y_k &= L_k^{-1}\left( g_k - Y_{k-1} y_{k-1} \right).\\
\end{aligned}
$$

At the final stage, we have 

$$
\begin{aligned}
    E_{N} x_{N-1} + D_N x_N &= g_N \\
    E_{N} L_{N-1}^{-\top} \left( y_{N-1} - Y_{N-1}^{\top} x_{N} \right) + D_N x_N &= g_N \\
    x_N &= L_N^{-\top}\left( \underbrace{L_N^{-1}g_N - L_N^{-1}Y_{N-1}y_{N-1}}_{y_{N}} \right),
\end{aligned}
$$

where $D_N - Y_{N-1}Y_{N-1}^\top = L_N L_N^{\top}$. This is the so-called factorization phase, where we perform cholesky factorizations and compute $Y_k$ and $y_k$ to get prepared for the solve phase. We summarize the algorithm as follows.

<div class="algorithm" markdown="1">
**Algorithm 1:** Serial Forward-Backward Sweep
<hr style="margin: 0.5em 0;">

$L_1 \gets \texttt{chol}(D_1)$  
$Y_1 \gets E_2 L_1^{-\top}$  
$y_1 \gets L_1^{-1} g_1$  
**for** $k = 2, 3, \ldots, N-1$ **do**  
&emsp;$L_k \gets \texttt{chol}(D_k - Y_{k-1}Y_{k-1}^\top)$  
&emsp;$Y_k \gets E_{k+1} L_k^{-\top}$  
&emsp;$y_k \gets L_k^{-1}\left( g_k - Y_{k-1} y_{k-1}\right)$  
$L_N \gets \texttt{chol}(D_N - Y_{N-1}Y_{N-1}^\top)$  
$y_N \gets L_{N}^{-1}\left( g_N - Y_{N-1} y_{N-1}\right)$  

$x_N = L_N^{-\top} y_N$  
**for** $k = N-1, \ldots, 1$ **do**  
&emsp;$x_k = L_k^{-\top} \left( y_k - Y_{k}^{\top} x_{k+1} \right)$ 
</div>


Next, we discuss a parallel method. After observing the banded system in \eqref{eq:banded-system}, we find that the process of eliminating odd equations in \eqref{eq:banded-system} can be performed in parallel. Specifically, we consider a set of equations associated with stages $i-1$, $i$, and $i+1$:

$$
\begin{align}
    E_{i-1} x_{i-2} + D_{i-1} x_{i-1} + F_{i-1} x_{i} &= g_{i-1} \\
    E_{i} x_{i-1} + D_{i} x_{i} + F_{i} x_{i+1} &= g_{i} \label{eq:odd_equations} \\
    E_{i+1} x_{i} + D_{i+1} x_{i+1} + F_{i+1} x_{i+2} &= g_{i+1}.
\end{align}
$$

We express $x_{i-1}$ using $x_{i-2}$ and $x_{i}$:

$$
\begin{aligned}
    x_{i-1} &= D_{i-1}^{-1} \left( g_{i-1} - E_{i-1} x_{i-2} - F_{i-1} x_{i} \right) \\
    &= L_{i-1}^{-\top}L_{i-1}^{-1} \left( g_{i-1} - E_{i-1} x_{i-2} - F_{i-1} x_{i} \right) \\
    &= L_{i-1}^{-\top} \left( L_{i-1}^{-1} g_{i-1} - L_{i-1}^{-1} E_{i-1} x_{i-2} - L_{i-1}^{-1} F_{i-1} x_{i} \right) \\
    &= L_{i-1}^{-\top} \left( L_{i-1}^{-1} g_{i-1} - Y_{i-1}^\top x_{i-2} - U_{i-1}^\top x_{i} \right),
\end{aligned}
$$

where $Y_{i-1} = E_{i-1}^{\top}L_{i-1}^{-\top},\; U_{i-1} = F_{i-1}^{\top} L_{i-1}^{-\top}$. We express $x_{i+1}$ in the same manner:

$$
\begin{aligned}
    x_{i+1} &= D_{i+1}^{-1} \left( g_{i+1} - E_{i+1} x_{i} - F_{i+1} x_{i+2} \right) \\
    &= L_{i+1}^{-\top} \left( L_{i+1}^{-1} g_{i+1} - Y_{i+1}^\top x_{i} - U_{i+1}^\top x_{i+2} \right),
\end{aligned}
$$

where $Y_{i+1} = E_{i+1}^\top L_{i+1}^{-\top},\; U_{i+1} = F_{i+1}^\top L_{i+1}^{-\top}$. Plugging the expressions for $x_{i-1}$ and $x_{i+1}$ into \eqref{eq:odd_equations}, we have 

$$
\begin{aligned}
    E_{i} x_{i-1} + D_{i} x_{i} + F_{i} x_{i+1} &= g_{i} \\
    \textcolor{blue}{E_{i}L_{i-1}^{-\top}} \left( L_{i-1}^{-1} g_{i-1} - Y_{i-1}^\top x_{i-2} - U_{i-1}^\top x_{i} \right) + D_i x_i + \textcolor{blue}{F_i L_{i+1}^{-\top}} \left( L_{i+1}^{-1} g_{i+1} - Y_{i+1}^\top x_{i} - U_{i+1}^\top x_{i+2} \right) &= g_i \\
    \textcolor{blue}{U_{i-1}} \left( L_{i-1}^{-1} g_{i-1} - Y_{i-1}^\top x_{i-2} - U_{i-1}^\top x_{i} \right) + D_i x_i + \textcolor{blue}{Y_{i+1}} \left( L_{i+1}^{-1} g_{i+1} - Y_{i+1}^\top x_{i} - U_{i+1}^\top x_{i+2} \right) &= g_i. 
\end{aligned}
$$

We rearrange the equation above and obtain
$$
\begin{aligned}
    -U_{i-1}Y_{i-1}^\top x_{i-2} + \left( D_i - U_{i-1}U_{i-1}^\top - Y_{i+1}Y_{i+1}^\top \right) x_i - Y_{i+1} U_{i+1}^\top x_{i+2}  &= g_i - U_{i-1} L_{i-1}^{-1} g_{i-1} - Y_{i+1} L_{i+1}^{-1} g_{i+1} \\
    \textcolor{blue}{E_{i}^{(1)}}x_{i-2} + \textcolor{blue}{D_{i}^{(1)}}x_{i} + \textcolor{blue}{F_{i}^{(1)}}x_{i+2} &= g_i - U_{i-1}\textcolor{blue}{y_{i-1}^{(1)}} - Y_{i+1}\textcolor{blue}{y_{i+1}^{(1)}},
\end{aligned}
$$

where the superscript $(\cdot)$ denotes the level index. This new equation is associated with $x_{i-2}$, $x_{i}$, and $x_{i+2}$. The figure below shows the elimination process.

<p align="center">
  <img src="/assets/img/bandedsystems/fact_single.svg" width="500"/>
</p>

Once all odd equations are eliminated, we obtain a reduced banded system:

$$
\begin{equation*}
\begin{bmatrix}
    D_{2}^{(1)} & F_{2}^{(1)} & & & \\
    E_{4}^{(1)} & D_{4}^{(1)} & F_{4}^{(1)} & & \\
    & E_{6}^{(1)} & D_{6}^{(1)} & F_{6}^{(1)} & \\
    && \ddots & \ddots & \ddots\\
    & & & E_{M}^{(1)} & D_{M}^{(1)}
    \end{bmatrix}
    \begin{bmatrix}
    x_{2}\\
    x_{4}\\
    x_{6}\\
    \vdots\\
    x_{M}
    \end{bmatrix}
    =
    \begin{bmatrix}
    g_{2}^{(1)}\\
    g_{4}^{(1)}\\
    g_{6}^{(1)}\\
    \vdots\\
    g_{M}^{(1)}
\end{bmatrix}.
\end{equation*}
$$

We then continue to eliminate $x_2, x_6, x_{10}, \dots$. This procedure can be repeated iteratively until the smallest system is reached. The complete reduction process is illustrated below.  

<p align="center">
  <img src="/assets/img/bandedsystems/fact.svg" width="800"/>
</p>

Next, we compute the unknown variables backwards using $L$, $Y$, and $U$. We start from the highest level $\kappa_{\text{max}}$:

$$
\begin{aligned}
    L_{i}^{(\kappa_{\max})} L_{i}^{(\kappa_{\max})\top} x_i &= g_i^{(\kappa_{\max})} \\
    L_{i}^{(\kappa_{\max})\top} x_i &= \left(L_{i}^{(\kappa_{\max})}\right)^{-1} g_i^{(\kappa_{\max})} \\
    x_i &= L_{i}^{(\kappa_{\max})-\top} \textcolor{blue}{y_i^{(\kappa_{\max})}}, 
\end{aligned}
$$

where $i = 2^{\kappa_{\max}}$ and $y_i^{(\kappa_{\max})}=\left(L_{i}^{(\kappa_{\max})}\right)^{-1} g_i^{(\kappa_{\max})}$. At level $\kappa$, we compute $x_i$ as follows:

$$
\begin{aligned}
    x_{i} &= L_{i}^{(\kappa)-\top} \left( L_{i}^{(\kappa)-1} g_{i}^{(\kappa)} - Y_{i}^{(\kappa)\top} x_{i-2^{\kappa}} - U_{i}^{(\kappa)\top} x_{i+2^{\kappa}} \right) \\
    &= L_{i}^{(\kappa)-\top} \left( \textcolor{blue}{y_i^{(\kappa)}} - Y_{i}^{(\kappa)\top} x_{i-2^{\kappa}} - U_{i}^{(\kappa)\top} x_{i+2^{\kappa}} \right),
\end{aligned}
$$

where $x_{i-2^{\kappa}}$ and $x_{i+2^{\kappa}}$ are obtained at level $\kappa+1$.

<p align="center">
  <img src="/assets/img/bandedsystems/single_solve.svg" width="400"/>
</p>

<p align="center">
  <img src="/assets/img/bandedsystems/solve.svg" width="800"/>
</p>

<br/><br/>

## References

{% bibliography --cited %}
