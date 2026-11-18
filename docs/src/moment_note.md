# Moment calculations

The integration assumes each angular bin has the same velocity direction
$\hat{\mathbf e}(\Omega)$ at every energy. Spacecraft potential changes the
particle speed while preserving this direction, giving
$\mathbf v_\infty = \sqrt{2E_\infty/m}\,\hat{\mathbf e}(\Omega)$.

Since $E_\infty = E + q\Phi_{sc}$, we have $dE_\infty = dE$.

```math
d^3v_\infty = \sqrt{\frac{2E_\infty}{m^3}}\, dE_\infty\, d\Omega_\infty = \sqrt{\frac{2E_\infty}{m^3}}\, dE\, d\Omega.
```

## Angular weights

Each angle bin $i$ is the patch $\theta_i \pm \Delta\theta_i/2$, $\phi_i \pm \Delta\phi_i/2$, on which $f_E$ is taken constant.
The angular factors in the sums below are exact integrals over the patch:

```math
\Delta\Omega_i = \int_i d\Omega, \qquad
\Omega_{j,i} = \int_i \hat e_j\, d\Omega, \qquad
\Omega_{jk,i} = \int_i \hat e_j \hat e_k\, d\Omega.
```

## Number density

The ambient number density is given by

```math
n = \int f(\mathbf v_∞) d^3v_\infty
```

where $f(\mathbf v_∞)$ is the ambient phase-space distribution function.

The **differential number flux**, which is measured by particle instruments, is related to the velocity distribution by

```math
j(E,\Omega) dE d\Omega = v f(\mathbf v_∞) d^3v .
```

Where the **measured** energy $E$ is related to the **measured** speed $v$ by $E=m v^2 / 2$. We get

```math
j(E,\Omega) = \frac{2E}{m^2} f(\mathbf v_∞).
```

The **differential energy flux** therefore satisfies

```math
f_E(E,\Omega) = E j(E,\Omega) = \frac{2E^2}{m^2} f(\mathbf v_∞)
```

so $f(\mathbf v_\infty) = \frac{m^2}{2E^2} f_E$. Substituting the relation between $f$ and $f_E$ into $n$ yields

```math
n = \sqrt{\frac{m}{2}} \int f_E(E,\Omega) \frac{\sqrt{E_∞}}{E^2} dE d\Omega.
```

And after discretization in energy and angle,

```math
n = \sqrt{\frac{m}{2}} \sum_i f_{E,i}\, a_i\, \frac{Δ E_i}{E_i ^ 2} \sqrt{E_{∞,i}}  Δ \Omega_i
```

where the weight $a_i$ suppresses bins below the spacecraft potential, which hold photoelectrons:

```math
a_i = \mathrm{clamp}\!\left(\frac{E_i + q\Phi_{sc}}{ Δ E_i} + \frac{1}{2},\; 0,\; 1\right).
```

## Flux

The particle flux is the first velocity-space moment (vector):

```math
\mathbf F = \int \mathbf v_\infty\, f(\mathbf v_\infty)\, d^3v_\infty,
\qquad
F_j = \int v_{\infty,j}\, f(\mathbf v_\infty)\, d^3v_\infty.
```

We obtain

```math
F_j = \int \left(\sqrt{\frac{2E_\infty}{m}}\, \hat e_j\right)
\left(\sqrt{\frac{2E_\infty}{m^3}}\right)
\frac{m^2}{2E^2} f_E \, dE d\Omega
= \int f_E \, \frac{E_\infty}{E^2}\, \hat e_j\, dE\, d\Omega.
```

After discretization in energy and angle, this becomes

```math
F_j = \sum_i f_{E,i}\, a_i\, \frac{\Delta E_i}{E_i ^2}\, E_{\infty,i}\; \Omega_{j,i}
\qquad j \in \{x,y,z\}
```

Compared to density, the integrand acquires one extra factor of $E_\infty/E$, and $\Omega_{j,i}$ replaces $\Delta\Omega_i$.

## Energy flux

The energy flux is the energy-weighted first moment:

```math
\mathbf F_E = \int E_\infty\, \mathbf v_\infty\, f(\mathbf v_\infty)\, d^3v_\infty,
\qquad
F_{E,j} = \int E_\infty\, v_{\infty,j}\, f(\mathbf v_\infty)\, d^3v_\infty.
```

Using the same substitutions as above yields

```math
F_{E,j} = \int f_E\, \frac{E_\infty^2}{E^2}\, \hat e_j\, dE\, d\Omega.
```

After discretization in energy and angle,

```math
F_{E,j} = \sum_i f_{E,i}\, a_i\, \frac{\Delta E_i}{E_i^2}\, E_{\infty,i}^2\; \Omega_{j,i},
\qquad j \in \{x,y,z\}.
```

## Momentum flux

The (symmetric) momentum flux tensor is the second velocity-space moment (rank-2 tensor):

```math
\mathrm{MF}_{jk} = \int m v_{\infty,j}\, v_{\infty,k}\, f(\mathbf v_\infty)\, d^3 v_\infty.
```

Using the relation between (1) $f$ and $f_E$, (2) $d^3v_\infty$ and $dE\ d\Omega$, we obtain

```math
\mathrm{MF}_{jk}
= \int m \left(\sqrt{\frac{2E_\infty}{m}}\,\hat e_j\right)
\left(\sqrt{\frac{2E_\infty}{m}}\,\hat e_k\right)
\left(\sqrt{\frac{2E_\infty}{m^3}}\right)
\frac{m^2}{2E^2} f_E \, dE\, d\Omega
= \sqrt{2 m} \int f_E \, \frac{E_\infty^{3/2}}{E^2}\, \hat e_j\,\hat e_k\, dE\, d\Omega.
```

After discretization in energy and angle, this becomes

```math
\mathrm{MF}_{jk} = \sqrt{2 m}\sum_i
f_{E,i}\, a_i\, \frac{\Delta E_i}{E_i^2}\, E_{\infty,i}^{3/2}\; \Omega_{jk,i},
\qquad (j,k) \in \{x,y,z\}^2.
```

In code and many data products, the symmetric tensor is stored as
$(xx, yy, zz, xy, xz, yz)$.

## Pressure tensor

The pressure tensor is the second *central* moment (momentum flux minus the bulk-flow/ram contribution):

```math
P_{jk} = \mathrm{MF}_{jk} - V_j\,F_k\, m,
```

## Heat flux

The heat flux is defined as the third velocity-space moment of the *thermal* (bulk-frame)
velocity $\mathbf w_\infty \equiv \mathbf v_\infty - \mathbf V$:

```math
\mathbf q = \int \tfrac{1}{2} m\,|\mathbf v_\infty - \mathbf V|^2\,(\mathbf v_\infty - \mathbf V)\, f(\mathbf v_\infty)\, d^3v_\infty,
\qquad
q_j = \int \tfrac{1}{2} m\,|\mathbf v_\infty - \mathbf V|^2\,(v_{\infty,j} - V_j)\, f(\mathbf v_\infty)\, d^3v_\infty.
```

Here $\mathbf V = \mathbf F/n$ is the bulk velocity. Expanding $|\mathbf w|^2\mathbf w$ with $\mathbf w = \mathbf v_\infty - \mathbf V$
and integrating term by term expresses $\mathbf q$ through the lab-frame moments above:

```math
\mathbf q = \mathbf F_E - \mathrm{MF}\,\mathbf V - \tfrac{1}{2}\operatorname{tr}(\mathrm{MF})\,\mathbf V + m\,n\,|\mathbf V|^2\,\mathbf V,
```

using $\mathbf F_E = \int \tfrac12 m |\mathbf v_\infty|^2 \mathbf v_\infty f\, d^3v_\infty$ and $\mathbf F = n\mathbf V$.
Each term is integrated exactly over the bins, so $\mathbf q$ is too.
SPEDAS instead evaluates $|\mathbf w|^2\mathbf w$ at bin centers, which is off by $O(\Delta\theta^2, \Delta\phi^2)$
and fails for wide bins such as a full-circle $\phi$ bin.
