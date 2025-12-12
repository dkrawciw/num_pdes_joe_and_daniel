import numpy as np
from scipy.sparse import csr_matrix, kron, eye
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import seaborn as sns

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
N = 5
x_range = (0, 2*np.pi)
y_range = (0, 2*np.pi)
x_vals = np.linspace(x_range[0], x_range[1], N,endpoint=False)
y_vals = np.linspace(y_range[0], y_range[1], N,endpoint=False)
X,Y = np.meshgrid(x_vals,y_vals, indexing='ij')
h = x_vals[1] - x_vals[0]

"""Time Setup"""
N_t = 100
t_range = [0, 50]
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
