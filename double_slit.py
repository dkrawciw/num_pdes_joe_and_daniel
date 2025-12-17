import numpy as np
from scipy.sparse import csr_matrix, kron, eye, csc_matrix
from scipy.sparse.linalg import spsolve, splu

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

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
N = 100
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
# D[-1,0] = 1
# D[0,-1] = 1
D /= h**2

D = csc_matrix(D)

L = kron(eye(N, format="csc"), D, format="csc") + kron(D, eye(N, format="csc"), format="csc")

"""Time Setup"""
N_t = 1000
t_range = [0, 0.2]
dt = (t_range[1] - t_range[0]) / N_t

"""Potential Function"""
def potential_func(u, delta=4):
    U = u.reshape((N,N), order="F")
    N_x = U.shape[0]
    N_y = U.shape[1]
    potential = 0*U
    

    potential[N_x//2-delta:N_x//2+delta, :] = 800
    potential[N_x//2-delta:N_x//2+delta, N_y//4-delta:N_y//4+delta] = 0
    potential[N_x//2-delta:N_x//2+delta, 3*N_y//4-delta:3*N_y//4+delta] = 0

    return potential.ravel(order="F")


"""Creating the solution array"""
U = np.zeros((N,N,N_t), dtype=complex)
U[0,:,0] = 1 / N

"""Begin evolving the solution in time"""
# nonlinear_func = lambda u: u * (np.abs(u)**2 + potential_func(u))
nonlinear_func = lambda u: u * (potential_func(u))

LHS = eye(N**2, format="csc") + 1j*dt/2 * L
LHS_lu = splu(LHS)

# Run forward 1 time step
u_now = U[:,:,0].ravel(order="F")
RHS = (eye(N**2, format="csc") - 1j*dt/2 * L)@u_now - 1j*dt*nonlinear_func(u_now) - 1j*dt
u_then = LHS_lu.solve(RHS)
U[:,:,1] = u_then.reshape((N,N), order="F")

"""Actually move it forward with CN-AB2"""
for k in tqdm(range(1,N_t-1), desc="Evolving SE", colour="green"):
    u_now = U[:,:,k].ravel(order="F")
    u_prev = U[:,:,k-1].ravel(order="F")

    RHS = (eye(N**2, format="csc") - 1j*dt/2 * L)@u_now - 1j*3/2 * dt * nonlinear_func(u_now) + 1j*dt/2 * nonlinear_func(u_prev)

    # u_then = spsolve(LHS,RHS)
    u_then = LHS_lu.solve(RHS)
    u_then = u_then / np.linalg.norm(u_then)

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

plt.suptitle('Evolution of a Particle Running Through a Double Slit', fontsize=18)

plt.tight_layout()
# plt.savefig("output/problem3.png", dpi=300)
plt.savefig("output/initial_and_final_double_slit.svg")

"""Create animation of the evolution"""
fig, ax = plt.subplots(figsize=(12, 8))

# # Pre-compute potential for overlay
# x0, y0 = np.pi, np.pi
# sigma = 0.3
# A = 50
# potential = A * np.exp( - ((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2) )

def animate(frame):
    ax.clear()
    U_frame = np.abs(U[:,:,frame])
    c = ax.contourf(X, Y, U_frame, levels=50, cmap='viridis')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_title(f'Evolving Wave Function at {dt*frame:.2f} s')
    plt.tight_layout()
    return c

# Animate every 10th frame to speed up
frames = range(0, N_t, 10)
anim = FuncAnimation(fig, animate, frames=frames, interval=100, blit=False)

# Save as GIF
anim.save("output/double_slit_animation.gif", writer='pillow', fps=10)
print("Animation saved as output/double_slit_animation.gif")

# """Plotting the first and last solution """
# fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# # # Initial Condition Plot
# # U_init_squared = np.abs(U[:,:,1].reshape((N,N), order="F"))
# # c1 = ax[0].contourf(X, Y, U_init_squared, levels=50, cmap='viridis')
# # fig.colorbar(c1, ax=ax[0], label=r'$| \psi |$')
# # ax[0].set_xlabel(r'$x$')
# # ax[0].set_ylabel(r'$y$')
# # ax[0].set_title('Initial Condition', fontsize=16)

# # Initial Condition Plot
# U_middle = np.abs(U[:,:,N_t//4].reshape((N,N), order="F"))
# c1 = ax[0].contourf(X, Y, U_middle, levels=50, cmap='inferno')
# fig.colorbar(c1, ax=ax[0], label=r'$| \psi |$')
# ax[0].set_xlabel(r'$x$')
# ax[0].set_ylabel(r'$y$')
# ax[0].set_title('Before Hitting Double Slit', fontsize=16)

# # Final Condition Plot
# U_final_squared = np.abs(U[:,:,-60].reshape((N,N), order="F"))
# c2 = ax[1].contourf(X, Y, U_final_squared, levels=50, cmap='inferno')
# fig.colorbar(c2, ax=ax[1], label=r'$| \psi |$')
# ax[1].set_xlabel(r'$x$')
# ax[1].set_ylabel(r'$y$')
# ax[1].set_title('Final State', fontsize=16)

# plt.suptitle('Evolution of Schrodinger Equation Through Time\nThrough a Double Slit', fontsize=18)

# plt.tight_layout()
# plt.savefig("output/first_middle_last_double_slit.png", dpi=600)

# plt.cla()
# plt.figure(figsize=(7,5))

# last_side = np.abs(U[-1,:,-60])

# plt.plot(y_vals, last_side, '-o', linewidth=4, markersize=9)
# plt.ylabel(r"$| \psi |$")
# plt.xlabel(r"y")
# plt.title("Probability Distribution at the Right Wall\nafter Travelling through Double Slit")

# plt.tight_layout()
# plt.savefig("output/edge_probability.svg")