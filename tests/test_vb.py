"""Checks of the estimator against exact results. Run: python -m pytest"""
import numpy as np
from scipy import integrate

from dynamic_covar import vb


def test_gig_moments():
    for a, b in [(10.5, 0.02), (2.0, 3.0), (50.0, 1e-4)]:
        dens = lambda z: z ** -0.5 * np.exp(-0.5 * (a * z + b / z))
        Z = integrate.quad(dens, 0, np.inf, limit=500)[0]
        num = [integrate.quad(lambda z: f(z) * dens(z) / Z, 0, np.inf, limit=500)[0]
               for f in (lambda z: z, lambda z: 1 / z, np.log)]
        assert np.allclose(num, [m[0] for m in vb.gig_half_moments(np.array([a]), np.array([b]))[:3]], rtol=1e-6)


def test_smoother_matches_joint_gaussian():
    rng = np.random.default_rng(1)
    T, p = 40, 3
    X = np.column_stack([np.ones(T), rng.standard_normal((T, p - 1))])
    y, R = rng.standard_normal(T), rng.uniform(0.5, 2, T)
    om, mu0, V0 = rng.uniform(0.01, 0.1, p), rng.standard_normal(p), np.diag(rng.uniform(0.5, 2, p))
    Qi, Prec, h = np.diag(1 / om), np.zeros((T * p, T * p)), np.zeros(T * p)
    for t in range(T):
        s = slice(t * p, (t + 1) * p)
        Prec[s, s] += np.outer(X[t], X[t]) / R[t]; h[s] += X[t] * y[t] / R[t]
        if t == 0:
            Prec[s, s] += np.linalg.inv(V0); h[s] += np.linalg.solve(V0, mu0)
        else:
            u = slice((t - 1) * p, t * p)
            Prec[s, s] += Qi; Prec[u, u] += Qi; Prec[s, u] -= Qi; Prec[u, s] -= Qi
    Cov = np.linalg.inv(Prec)
    m, _, xVx, var = vb.kalman_smoother(y, R, X, om, mu0, V0, X)
    blocks = [Cov[t * p:(t + 1) * p, t * p:(t + 1) * p] for t in range(T)]
    assert np.allclose(m.ravel(), Cov @ h)
    assert np.allclose(xVx, [X[t] @ blocks[t] @ X[t] for t in range(T)])
    assert np.allclose(var, [np.diag(B) for B in blocks])


def test_elbo_never_decreases_and_recovers_drift():
    rng = np.random.default_rng(0)
    T, q = 1500, 0.05
    X = np.column_stack([np.ones(T), rng.standard_normal(T)])
    b = np.array([0.0, 1.0]) + np.cumsum(0.02 * rng.standard_normal((T, 2)), axis=0)
    y = np.einsum("tp,tp->t", X, b) + rng.standard_t(5, T)
    model = vb._Model(y, X, q, np.full(2, 1e-4), np.zeros(2), 10 * np.eye(2))
    th, elbos = model.initial(np.array([-1.6, 1.0])), []
    for _ in range(30):
        th, e, _ = model.sweep(th)
        elbos.append(e)
    assert np.all(np.diff(elbos) > -1e-8 * np.abs(elbos[1:]))
    f = vb.fit(y, X, q)
    assert np.sqrt(np.mean((f["m"] - b) ** 2)) < np.sqrt(np.mean((f["qr_coef"] - b) ** 2))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print(name, "ok")
