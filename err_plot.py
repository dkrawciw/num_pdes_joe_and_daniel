import numpy as np
from scipy.sparse import csr_matrix, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve, splu
# from sksparse.cholmod import cholesky

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

"""Plot setup"""
sns.set_style("whitegrid")
sns.set_color_codes(palette="colorblind")

plt.rcParams.update({
	"text.usetex": False,  # keep False to avoid requiring a LaTeX installation
	"mathtext.fontset": "cm",  # Computer Modern (LaTeX-like)
	"font.family": "serif",
	"font.serif": ["Computer Modern Roman", "DejaVu Serif"],
    "axes.labelsize": 14,      # increase axis label size
    "axes.titlesize": 16,
    "xtick.labelsize": 14,     # increase tick / bin label size
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
})

"""Spacial Setup"""
N = 50
x_range = (0, 2*np.pi)
y_range = (0, 2*np.pi)
x_vals = np.linspace(x_range[0], x_range[1], N,endpoint=False)
y_vals = np.linspace(y_range[0], y_range[1], N,endpoint=False)
X,Y = np.meshgrid(x_vals,y_vals, indexing='ij')
h = x_vals[1] - x_vals[0]

"""Construct the periodic Laplacian"""
D = np.zeros((N,N))
D += np.eye(N, k=0) * -2
D += np.eye(N, k=1)
D += np.eye(N, k=-1)
D[-1,0] = 1
D[0,-1] = 1
D /= h**2

D = csc_matrix(D)

L = kron(eye(N, format="csc"), D, format="csc") + kron(D, eye(N, format="csc"), format="csc")

"""Setting up the number of times"""
N_t_list = list(range(10,2000,100))
t_range = [0, 1]
err_list = []

nonlinear_func = lambda u: u * np.abs(u)**2
forcing_fxn = lambda x,y,t: np.exp(-1j*t)*(3*np.cos(x)*np.sin(y) - (np.cos(x)*np.sin(y))**3)

exact_soln_func = lambda x,y,t: np.exp(-1j*t)*np.cos(x)*np.sin(y)
exact_soln = exact_soln_func(X,Y,t_range[1]).ravel(order="F")

for N_t in tqdm(N_t_list, desc="Solving at different time steps", colour="blue"):
    """Time Setup"""
    # t_range = [0, 1]
    dt = (t_range[1] - t_range[0]) / N_t

    """Creating the solution array"""
    U = np.zeros((N,N,N_t), dtype=complex)
    U[:,:,0] = exact_soln_func(X,Y,0)

    """Begin evolving the solution in time"""
    F = forcing_fxn(X,Y,0)
    f = F.ravel(order="F")

    LHS = eye(N**2, format="csc") + 1j*dt/2 * L
    LHS_lu = splu(LHS)
    
    # Run forward 1 time step
    u_now = U[:,:,0].ravel(order="F")
    # LHS = eye(N**2, format="csr") - dt/2 * L
    RHS = (eye(N**2, format="csc") - 1j*dt/2 * L)@u_now - 1j*dt*nonlinear_func(u_now) - 1j*dt*f
    # u_then = spsolve(LHS,RHS)
    u_then = LHS_lu.solve(RHS)
    U[:,:,1] = u_then.reshape((N,N), order="F")

    """Actually move it forward with CN-AB2"""
    for k in tqdm(range(1,N_t-1), desc="Evolving SE"):
        u_now = U[:,:,k].ravel(order="F")
        u_prev = U[:,:,k-1].ravel(order="F")

        F = forcing_fxn(X,Y,k*dt)
        f = F.ravel(order="F")

        RHS = (eye(N**2, format="csc") - 1j*dt/2 * L)@u_now - 1j*3/2 * dt * nonlinear_func(u_now) + 1j*dt/2 * nonlinear_func(u_prev) - 1j*dt*f

        # u_then = spsolve(LHS,RHS)
        u_then = LHS_lu.solve(RHS)
        # u_then = u_then / np.linalg.norm(u_then)

        U[:,:,k+1] = u_then.reshape((N,N), order="F")
    
    u_last = U[:,:,-1].ravel(order="F")
    err = np.linalg.norm(u_last - exact_soln, ord=2) / np.linalg.norm(exact_soln, ord=2)
    err_list.append(err)

plt.figure(figsize=(8,5))

plt.loglog(N_t_list, err_list, "-o", linewidth=4, markersize=8, label=r"$2$-Norm Error")
plt.loglog(N_t_list,  1/(np.array(N_t_list)), "k", label=r"Reference $O(n)$ Convergence")

plt.xlabel("Number of Time Points")
# plt.ylabel(r"Error")
plt.ylabel(r"$\log \log$ Error")
plt.title("Error Convergence of Our Given Method as\n# of Time Points Increases")
plt.legend()

plt.tight_layout()
plt.savefig("output/error_convergence.svg")

plt.clf()

""" Plotting the Actual and Numerical """
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Actual Solution Plot
U_real = exact_soln_func(X,Y,t_range[1])
c1 = ax[0].contourf(X, Y, np.abs(U_real), levels=50, cmap='viridis')
fig.colorbar(c1, ax=ax[0], label=r'$u(x,y)$')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Actual Solution', fontsize=16)

# Final Condition Plot
U_final_squared = np.abs(U[:,:,-1].reshape((N,N), order="F"))
c2 = ax[1].contourf(X, Y, U_final_squared, levels=50, cmap='viridis')
fig.colorbar(c2, ax=ax[1], label=r'$u(x,y)$')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].set_title('Numerical Solution', fontsize=16)

plt.suptitle('Evolution of Schrodinger Equation Through Time', fontsize=18)

plt.tight_layout()
# plt.savefig("output/problem3.png", dpi=300)
plt.savefig("output/Method_of_Manufactured_Solution_comparison.svg")