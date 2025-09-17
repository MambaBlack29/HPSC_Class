# Assignment 1
**Name:** Aryan Agarwal
**Roll No.:** 22EC39006  

### Experimental Setup
- `.msh` files generated using `gmsh` software and converted to `fem.c` readable format using the given `C` code files
- `fem.c` updated for the given equation and boundary conditions and run for meshes with around $40$, $80$ and $160$ points
- Convergence achieved iteratively using **Jacobi**, **Gauss-Seidel** and **Successive Over-Relaxation** for a variety of mesh sizes and $\omega$ values
- Initial guess $x_0$ picked randomly, and average over $10k$ convergences taken to find **number of iterations** till convergence
- An error tolerance of $10^{-8}$ was considered for convergence

### Actual Mesh Sizes
| Desired Mesh Size | True Mesh Size |
|:-----------------:|:--------------:|
| 40                | 44             |
| 80                | 74             |
| 160               | 167            |

### Jacobi
- Iterative step:
$$ 
\begin{array}{rcl}
D\, x^{k+1} = b - (L + U)\ x^k &
\Longleftrightarrow &
x^{k+1} = G_J\, x^k + c_J
\end{array} \\
\begin{array}{rl}
G_J = -D^{-1} (L + U), &
c_J = D^{-1} \, b
\end{array} 
$$
- Where $D = \text{diag}(A)$, $L = \text{lower\_triangle}(A)$ and $U = \text{upper\_triangle}(A)$

| N (mesh) | Spectral Radius $\rho(G)$ | Convergence Rate = $-ln(\rho)$ | # Iterations |
|:--------:|:--------------------:|:------------------:|:------------:|
| 40       | 0.8132               | 0.2068             | 71           |
| 80       | 0.8824               | 0.1251             | 109          |
| 160      | 0.9495               | 0.0518             | 238          |

### Gauss–Seidel
- Iterative step:
$$ 
\begin{array}{rcl}
(D + L)\, x^{k+1} = b - U x^k &
\Longleftrightarrow &
x^{k+1} = G_{GS}\, x^k + c_{GS}
\end{array} \\
\begin{array}{rl}
G_{GS} = -(D + L)^{-1}\, U, &
c_{GS} = (D + L)^{-1}\, b
\end{array} 
$$
- Where $D = \text{diag}(A)$, $L = \text{lower\_triangle}(A)$ and $U = \text{upper\_triangle}(A)$

| N (mesh) | Spectral Radius $\rho(G)$ | Convergence Rate = $-ln(\rho)$ | # Iterations |
|:--------:|:--------------------:|:------------------:|:------------:|
| 40       | 0.6671               | 0.4048             | 39           |
| 80       | 0.7821               | 0.2457             | 60           |
| 160      | 0.9024               | 0.1027             | 129          |

### Successive Over-Relaxation (SOR) 
- Iterative step:
$$ 
\begin{array}{rcl}
(D + \omega L)\, x^{k+1} = \omega b + [(1 - \omega)D - \omega U]\, x^k &
\Longleftrightarrow &
x^{k+1} = G_{SOR}\, x^k + c_{SOR}
\end{array} \\
\begin{array}{rl}
G_{SOR}(\omega) = (D + \omega L)^{-1}[(1 - \omega)D - \omega U], &
c_{SOR}(\omega) = \omega(D + \omega L)^{-1}\, b
\end{array} 
$$
- Where $D = \text{diag}(A)$, $L = \text{lower\_triangle}(A)$ and $U = \text{upper\_triangle}(A)$

#### $(\omega = 1.2)$:
| N (mesh) | Spectral Radius $\rho(G)$ | Convergence Rate = $-ln(\rho)$ | # Iterations |
|:--------:|:--------------------:|:------------------:|:------------:|
| 40       | 0.4969               | 0.6993             | 25           |
| 80       | 0.6749               | 0.3931             | 40           |
| 160      | 0.8543               | 0.1575             | 89           |

#### $(\omega = 1.5)$:
| N (mesh) | Spectral Radius $\rho(G)$ | Convergence Rate = $-ln(\rho)$ | # Iterations |
|:--------:|:--------------------:|:------------------:|:------------:|
| 40       | 0.5347               | 0.6260             | 35           |
| 80       | 0.5498               | 0.5982             | 36           |
| 160      | 0.6914               | 0.3690             | 43           |

#### $(\omega = 1.9)$:
| N (mesh) | Spectral Radius $\rho(G)$ | Convergence Rate = $-ln(\rho)$ | # Iterations |
|:--------:|:--------------------:|:------------------:|:------------:|
| 40       | 0.9085               | 0.0960             | 225          |
| 80       | 0.9087               | 0.0957             | 231          |
| 160      | 0.9093               | 0.0950             | 236          |
