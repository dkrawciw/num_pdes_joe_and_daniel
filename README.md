# Numerical PDEs Final Project - Schrodinger in 2D with periodic BCs

Joseph Huston

Daniel Krawciw

## Setup

**Schrodinger's Equation**

$$
\begin{equation}
    i \hbar \frac{\partial}{\partial t} \psi = \hat{H} \psi
\end{equation}
$$
where $\psi(\underline{x},t) \in \mathbb{C}$ is the wave function and $\hat{H}$ is defined as 
$$
\begin{equation}
    \hat{H} = - \frac{\hbar^2}{2m} \nabla^2 + V(\underline{x},t).
\end{equation}
$$
Here, $V(\underline{x},t)$ is the potential function.

For now, we will write the equation out like:
$$
\begin{equation}
    i \hbar \frac{\partial}{\partial t} \psi = - \frac{\hbar^2}{2m} \nabla^2\psi + V(\underline{x},t) \psi
\end{equation}
$$

**Dr. Sprinkle's Equation**

$$
\begin{equation}
    i \psi_t = \nabla^2 \psi + \psi |\psi|^2
\end{equation}
$$

## Numerical Schemes

The following is CN-AB2 which was given in class. The linear component is Crank Nicolson and the nonlinear part is AB2.

$$
\begin{equation}
    \frac{u^{n+1} - u^n}{\Delta t} = A\left[ \frac{1}{2} (u^{n+1} + u^n) \right] + \frac{3}{2} N(u^n) - \frac{1}{2} N(u^{n-1})
\end{equation}
$$
where $A$ represents a linear operator and $N(u)$ represents the nonlinear operator.

solved for $u^{n+1}$

$$
\begin{equation}
    u^{n+1} = [I-\frac{\Delta t}{2}A]^{-1} ([I+\frac{\Delta t}{2}A] u^n + \frac{3 \Delta t}{2}N(u^n) - \frac{\Delta t}{2}N(u^{n-1}))
\end{equation}
$$