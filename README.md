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