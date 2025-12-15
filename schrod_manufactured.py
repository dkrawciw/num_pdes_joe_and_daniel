import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve


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

"""Time Setup"""
N_t = 5000
t_range = [0, 1]
dt = (t_range[1] - t_range[0]) / N_t

"""Construct the periodic Laplacian"""
D = np.zeros((N,N))
D += np.eye(N, k=0) * -2
D += np.eye(N, k=1)
D += np.eye(N, k=-1)
D[-1,0] = 1
D[0,-1] = 1

D = csr_matrix(D)

L = kron(eye(N, format="csr"), D, format="csr") + kron(D, eye(N, format="csr"), format="csr")

nonlinear_func = lambda u: np.multiply(u,np.multiply(u,np.conjugate(u)))


"""Creating the solution array"""
U = np.zeros((N,N,N_t), dtype=complex)
# U[0,N//2,0] = 1
U[:,:,0] = np.multiply(np.sin(X),np.sin(Y))
"""since f is separable in relevent cases, can make a function of t"""
f = lambda t: -1j*np.sin(t)*np.sin(X)*np.sin(Y) - 2*np.cos(t)*np.sin(X)*np.sin(Y) + pow(np.cos(t),3)*pow(np.sin(X),3)*pow(np.sin(Y),3)
print(np.shape(f(0).ravel(order="F")))
print(np.shape(U[:,:,0].ravel(order="F")))


# Run forward 1 time step
u_now = U[:,:,0].ravel(order="F")
F = f(0)
f_now = F.ravel(order="F")
LHS = eye(N**2, format="csr") - dt/2 * L
RHS = (eye(N**2, format="csr") + dt/2 * L)@u_now + dt*nonlinear_func(u_now) + dt*f_now
u_then = spsolve(LHS,RHS)
U[:,:,1] = u_then.reshape((N,N), order="F")

for k in tqdm(range(1,N_t-1), desc="Evolving SE"):
    u_now = U[:,:,k].ravel(order="F")
    u_prev = U[:,:,k-1].ravel(order="F")

    t = k*dt
    F = f(t)
    f_now = F.ravel(order="F")

    LHS = eye(N**2, format="csr") - 1j*dt/2 * L
    RHS = (eye(N**2, format="csr") + 1j*dt/2 * L)@u_now + 1j*3/2 * dt * nonlinear_func(u_now) - 1j*dt/2 * nonlinear_func(u_prev) + dt*f_now

    u_then = spsolve(LHS,RHS)

    U[:,:,k+1] = u_then.reshape((N,N), order="F")


"""Plotting the first and last solution """
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Initial Condition Plot
U_init_squared = np.abs(U[:,:,0].reshape((N,N), order="F"))
c1 = ax[0].contourf(X, Y, U_init_squared, levels=50, cmap='viridis')
fig.colorbar(c1, ax=ax[0], label=r'$u(x,y)$')
ax[0].set_xlabel(r'$x$')
ax[0].set_ylabel(r'$y$')
ax[0].set_title('Initial Condition', fontsize=16)

# Final Condition Plot
U_final_squared = np.abs(U[:,:,-1].reshape((N,N), order="F"))
c2 = ax[1].contourf(X, Y, U_final_squared, levels=50, cmap='viridis')
fig.colorbar(c2, ax=ax[1], label=r'$u(x,y)$')
ax[1].set_xlabel(r'$x$')
ax[1].set_ylabel(r'$y$')
ax[1].set_title('Final State', fontsize=16)

plt.suptitle('Evolution of Schrodinger Equation Through Time', fontsize=18)

plt.tight_layout()
# plt.savefig("output/problem3.png", dpi=300)
plt.savefig("output/problem3.svg")
U_real = np.cos(t)*np.multiply(np.sin(X),np.sin(Y))

print(np.linalg.norm(U[:,:,-1] - U_real))