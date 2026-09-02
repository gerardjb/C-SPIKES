import numpy as np
import scipy
import scipy.signal
from math import sqrt, log, exp
from .oasis_methods import constrained_oasisAR1, oasisAR1
from warnings import warn
from scipy.optimize import minimize, curve_fit


def _finite_scalar(value, name):
    """Return *value* as a finite float or raise a public-facing error."""

    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _normalize_g(g):
    """Normalize the public AR coefficient input without retaining caller-owned storage."""

    if g is None or np.ndim(g) == 0:
        values = (g,)
    else:
        array = np.asarray(g, dtype=object)
        if array.ndim != 1:
            raise ValueError("g must be a scalar or a one-dimensional sequence")
        values = tuple(array.tolist())

    if len(values) not in (1, 2):
        raise ValueError("g must contain one AR(1) coefficient or two AR(2) coefficients")

    missing = tuple(value is None for value in values)
    if any(missing):
        if not all(missing):
            raise ValueError("g coefficients must either all be provided or all be None")
        return tuple(None for _ in values)

    return tuple(_finite_scalar(value, "g") for value in values)


def _validate_stable_g(g):
    """Validate coefficients supported by the logarithmic OASIS kernels."""

    if len(g) == 1:
        if not 0 < g[0] < 1:
            raise ValueError("AR(1) g must satisfy 0 < g < 1")
        return

    roots = np.roots((1.0, -g[0], -g[1]))
    if (np.max(np.abs(roots.imag)) > 1e-12 or
            np.any(roots.real <= 0) or np.any(roots.real >= 1)):
        raise ValueError("AR(2) g must have two real roots strictly between 0 and 1")


def deconvolve(y, g=(None,), sn=None, b=None, b_nonneg=True,
               optimize_g=0, penalty=0, **kwargs):
    """Infer spikes from a fluorescence trace using OASIS methods.

    Solves the noise-constrained sparse non-negative deconvolution problem
    ``min |s|_q`` subject to ``|c-y|^2 = sn^2 T`` and ``s = Gc >= 0`` where
    ``q`` is either 1 or 0. AR(1) uses the compiled active-set solver; AR(2)
    retains the comparator's Python ONNLS backend.

    ``y`` may be a floating-point list or one-dimensional array and is copied
    to float64 before inference. ``g`` contains one or two per-bin AR
    coefficients; use all-``None`` coefficients to estimate them. If ``b`` is
    supplied, it is treated as the authoritative fluorescence baseline and is
    subtracted before inference. The returned calcium trace therefore excludes
    the baseline, while the returned baseline remains ``b``. ``b_nonneg`` only
    constrains an estimated baseline.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("Input trace must be a one-dimensional array")
    if y.size < 3:
        raise ValueError("Input trace must contain at least three samples")
    if not np.issubdtype(y.dtype, np.floating):
        raise TypeError("Input trace should be a floating point array")
    y = y.astype(np.double, copy=True)
    if not np.isfinite(y).all():
        raise ValueError("Input trace must contain only finite values")

    g = _normalize_g(g)
    estimate_g = g[0] is None

    if sn is not None:
        sn = _finite_scalar(sn, "sn")
        if sn < 0:
            raise ValueError("sn must be non-negative")
    if b is not None:
        b = _finite_scalar(b, "b")
    if not isinstance(b_nonneg, (bool, np.bool_)):
        raise TypeError("b_nonneg must be a boolean")
    if (isinstance(optimize_g, (bool, np.bool_)) or
            not isinstance(optimize_g, (int, np.integer)) or optimize_g < 0):
        raise ValueError("optimize_g must be a non-negative integer")
    optimize_g = int(optimize_g)
    if (isinstance(penalty, (bool, np.bool_)) or
            not isinstance(penalty, (int, np.integer)) or penalty not in (0, 1)):
        raise ValueError("penalty must be either 0 (L0) or 1 (L1)")
    penalty = int(penalty)

    solver_kwargs = dict(kwargs)
    decimate = solver_kwargs.get("decimate", 1 if len(g) == 1 else 5)
    if (isinstance(decimate, (bool, np.bool_)) or
            not isinstance(decimate, (int, np.integer)) or decimate < 1):
        raise ValueError("decimate must be a positive integer")
    if decimate > len(y):
        raise ValueError("decimate cannot exceed the trace length")
    if "decimate" in solver_kwargs:
        solver_kwargs["decimate"] = int(decimate)

    working_y = y if b is None else y - b

    if estimate_g or sn is None:
        minimum_length = 11 + len(g)
        scale = max(1.0, float(np.max(np.abs(working_y))))
        if len(y) < minimum_length:
            raise ValueError(
                f"Automatic parameter estimation for AR({len(g)}) requires at least "
                f"{minimum_length} samples"
            )
        if np.std(working_y) <= np.finfo(np.double).eps * scale:
            raise ValueError("Automatic parameter estimation requires a non-constant trace")

    if estimate_g or sn is None:
        fudge_factor = .97 if (optimize_g and len(g) == 1) else .98
        est = estimate_parameters(working_y, p=len(g), fudge_factor=fudge_factor)
        if estimate_g:
            g = tuple(float(value) for value in np.asarray(est[0]).reshape(-1))
        if sn is None:
            sn = float(est[1])

    _validate_stable_g(g)
    if not np.isfinite(sn) or sn < 0:
        raise ValueError("Estimated sn must be finite and non-negative")

    if len(g) == 1:
        c, s, fitted_b, fitted_g, lam = constrained_oasisAR1(
            working_y, g[0], sn,
            optimize_b=True if b is None else False,
            b_nonneg=b_nonneg,
            optimize_g=optimize_g,
            penalty=penalty,
            **solver_kwargs
        )
        fitted_g = float(fitted_g)
        _validate_stable_g((fitted_g,))
        return c, s, fitted_b + (0.0 if b is None else b), fitted_g, lam

    if len(g) == 2:
        if optimize_g > 0:
            warn("Optimization of AR parameters is already fairly stable for AR(1), "
                 "but slower and more experimental for AR(2)")
        c, s, fitted_b, fitted_g, lam = constrained_onnlsAR2(
            working_y, list(g), sn,
            optimize_b=True if b is None else False,
            b_nonneg=b_nonneg,
            optimize_g=optimize_g,
            penalty=penalty,
            **solver_kwargs
        )
        fitted_g = tuple(float(value) for value in fitted_g)
        _validate_stable_g(fitted_g)
        return c, s, fitted_b + (0.0 if b is None else b), fitted_g, lam


def _nnls(KK, Ky, s=None, mask=None, tol=1e-9, max_iter=None):
    """Solve non-negative least squares ``argmin_s ||Ks-y||_2`` for ``s>=0``."""

    if mask is None:
        mask = np.ones(len(KK), dtype=bool)
    else:
        KK = KK[mask][:, mask]
        Ky = Ky[mask]

    if s is None:
        s = np.zeros(len(KK))
        l = Ky.copy()
        P = np.zeros(len(KK), dtype=bool)
    else:
        s = s[mask]
        P = s > 0
        l = Ky - KK[:, P].dot(s[P])

    if max_iter is None:
        max_iter = len(KK)

    for _ in range(max_iter):
        w = np.argmax(l)
        P[w] = True
        try:
            mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
        except Exception:
            mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
            print(r'added $\epsilon$I to avoid singularity')
        while len(mu > 0) and min(mu) < 0:
            a = min(s[P][mu < 0] / (s[P][mu < 0] - mu[mu < 0]))
            s[P] += a * (mu - s[P])
            P[s <= tol] = False
            try:
                mu = np.linalg.inv(KK[P][:, P]).dot(Ky[P])
            except Exception:
                mu = np.linalg.inv(KK[P][:, P] + tol * np.eye(P.sum())).dot(Ky[P])
                print(r'added $\epsilon$I to avoid singularity')
        s[P] = mu.copy()
        l = Ky - KK[:, P].dot(s[P])
        if max(l) < tol:
            break

    tmp = np.zeros(len(mask))
    tmp[mask] = s
    return tmp


def onnls(y, g, lam=0, shift=100, window=None, mask=None, tol=1e-9, max_iter=None):
    """Infer spikes for AR(2) by solving sparse non-negative deconvolution."""

    T = len(y)
    if mask is None:
        mask = np.ones(T, dtype=bool)

    if window is None:
        w = max(200, len(g) if len(g) > 2 else
                int(-5 / log(g[0] if len(g) == 1 else
                             (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2)))
    else:
        w = window
    w = min(T, w)

    K = np.zeros((w, w))
    if len(g) == 1:  # kernel for AR(1)
        _y = y - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        h = np.exp(log(g[0]) * np.arange(w))
        for i in range(w):
            K[i:, i] = h[:w - i]
    elif len(g) == 2:  # kernel for AR(2)
        _y = y - lam * (1 - g[0] - g[1])
        _y[-2] = y[-2] - lam * (1 - g[0])
        _y[-1] = y[-1] - lam
        d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
        r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
        if d == r:
            h = np.exp(log(d) * np.arange(1, w + 1)) * np.arange(1, w + 1)
        else:
            h = (np.exp(log(d) * np.arange(1, w + 1)) -
                 np.exp(log(r) * np.arange(1, w + 1))) / (d - r)
        for i in range(w):
            K[i:, i] = h[:w - i]
    else:  # arbitrary kernel
        h = g
        for i in range(w):
            K[i:, i] = h[:w - i]
        if lam:
            a = np.linalg.inv(K).sum(0)
            _y = y - lam * a[0]
            _y[-w:] = y[-w:] - lam * a
        else:
            _y = y

    s = np.zeros(T)
    KK = K.T.dot(K)
    for i in range(0, max(1, T - w), shift):
        s[i:i + w] = _nnls(
            KK,
            K.T.dot(_y[i:i + w]),
            s[i:i + w],
            mask=mask[i:i + w],
            tol=tol,
            max_iter=max_iter,
        )[:w]
        _y[i:i + w] -= K[:, :shift].dot(s[i:i + shift])

    s[i + shift:] = _nnls(
        KK[-(T - i - shift):, -(T - i - shift):],
        K[:T - i - shift, :T - i - shift].T.dot(_y[i + shift:]),
        s[i + shift:],
        mask=mask[i + shift:]
    )

    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, T - t)]
    return c, s


def constrained_onnlsAR2(y, g, sn, optimize_b=True, b_nonneg=True, optimize_g=0, decimate=5,
                         shift=100, window=None, tol=1e-9, max_iter=1, penalty=1):
    """Infer spikes for AR(2) using a noise-constrained ONNLS procedure."""

    T = len(y)
    d = (g[0] + sqrt(g[0] * g[0] + 4 * g[1])) / 2
    r = (g[0] - sqrt(g[0] * g[0] + 4 * g[1])) / 2
    if window is None:
        window = int(min(T, max(200, -5 / log(d))))
    if not optimize_g:
        g11 = (np.exp(log(d) * np.arange(1, T + 1)) * np.arange(1, T + 1)) \
            if d == r else \
            (np.exp(log(d) * np.arange(1, T + 1)) -
             np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
        g12 = np.append(0, g[1] * g11[:-1])
        g11g11 = np.cumsum(g11 * g11)
        g11g12 = np.cumsum(g11 * g12)
        Sg11 = np.cumsum(g11)
        f_lam = 1 - g[0] - g[1]
    elif decimate == 0:
        decimate = 1
    thresh = sn * sn * T

    if decimate > 0:
        _, s, b, aa, lam = constrained_oasisAR1(
            y[:len(y) // decimate * decimate].reshape(-1, decimate).mean(1),
            d**decimate, sn / sqrt(decimate),
            optimize_b=optimize_b, b_nonneg=b_nonneg, optimize_g=optimize_g)
        if optimize_g:
            d = aa**(1. / decimate)
            if decimate > 1:
                s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
            r = estimate_time_constant(s, 1, fudge_factor=.98)[0]
            g[0] = d + r
            g[1] = -d * r
            g11 = (np.exp(log(d) * np.arange(1, T + 1)) -
                   np.exp(log(r) * np.arange(1, T + 1))) / (d - r)
            g12 = np.append(0, g[1] * g11[:-1])
            g11g11 = np.cumsum(g11 * g11)
            g11g12 = np.cumsum(g11 * g12)
            Sg11 = np.cumsum(g11)
            f_lam = 1 - g[0] - g[1]
        elif decimate > 1:
            s = oasisAR1(y - b, d, lam=lam * (1 - aa) / (1 - d))[1]
        lam *= (1 - d**decimate) / f_lam
        ff = np.ravel([a + np.arange(-2, 2) for a in np.where(s > s.max() / 10.)[0]])
        ff = np.unique(ff[(ff >= 0) * (ff < T)]).astype(int)
        mask = np.zeros(T, dtype=bool)
        mask[ff] = True
    else:
        b = np.percentile(y, 15) if optimize_b else 0
        lam = 2 * sn * np.linalg.norm(g11)
        mask = None

    if b_nonneg:
        b = max(b, 0)

    c, s = onnls(y - b, g, lam=lam, mask=mask,
                 shift=shift, window=window, tol=tol)
    g_converged = False
    if not optimize_b:
        for _ in range(max_iter - 1):
            res = y - c
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4:
                break
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):
                l = ls[i + 1] - f - 1
                if i == len(ls) - 2:
                    tmp[f] = (1. / f_lam if l == 0 else
                              (Sg11[l] + g[1] / f_lam * g11[l - 1]
                               + (g[0] + g[1]) / f_lam * g11[l]
                               - g11g12[l] * tmp[f - 1]) / g11g11[l])
                elif i == len(ls) - 3 and ls[-2] == T - 1:
                    tmp[f] = (Sg11[l] + g[1] / f_lam * g11[l]
                              - g11g12[l] * tmp[f - 1]) / g11g11[l]
                else:
                    tmp[f] = (Sg11[l] - g11g12[l] * tmp[f - 1]) / g11g11[l]
                l += 1
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]

            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                dlam = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except Exception:
                dlam = -bb / aa
            lam += dlam / f_lam
            c, s = onnls(y, g, lam=lam, mask=mask, shift=shift, window=window, tol=tol)

            if optimize_g and (not g_converged):
                def getRSS(y_, opt):
                    ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d_, r_ = exp(ld), exp(lr)
                    g1_, g2_ = d_ + r_, -d_ * r_
                    tmp_ = onnls(y_, [g1_, g2_], lam, mask=(s > 1e-2 * s.max()))[0] - y_
                    return tmp_.dot(tmp_)

                result = minimize(lambda x: getRSS(y, x), (log(d), log(r)),
                                  bounds=((None, -1e-4), (None, -1e-3)), method='L-BFGS-B',
                                  options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - log(d)) < 1e-4:
                    g_converged = True
                ld, lr = result['x']
                d, r = exp(ld), exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)

    else:
        db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
        b += db
        lam -= db / (1 - g[0] - g[1])
        for _ in range(max_iter - 1):
            res = y - c - b
            RSS = res.dot(res)
            if np.abs(RSS - thresh) < 1e-4:
                break
            tmp = np.empty(T)
            ls = np.append(np.where(s > 1e-6)[0], T)
            l = ls[0]
            tmp[:l] = (1 + d) / (1 + d**l) * np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):
                l = ls[i + 1] - f
                tmp[f] = (Sg11[l - 1] - g11g12[l - 1] * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            tmp -= tmp.mean()
            aa = tmp.dot(tmp)
            bb = res.dot(tmp)
            cc = RSS - thresh
            try:
                db = (-bb + sqrt(bb * bb - aa * cc)) / aa
            except Exception:
                db = -bb / aa
            if b_nonneg:
                db = max(db, -b)
            b += db
            c, s = onnls(y - b, g, lam=lam, mask=mask,
                         shift=shift, window=window, tol=tol)
            db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
            b += db
            lam -= db / f_lam

            if optimize_g and (not g_converged):
                def getRSS(y_, opt):
                    b_, ld, lr = opt
                    if ld < lr:
                        return 1e3 * thresh
                    d_, r_ = exp(ld), exp(lr)
                    g1_, g2_ = d_ + r_, -d_ * r_
                    tmp_ = b_ + onnls(y_ - b_, [g1_, g2_], lam,
                                      mask=(s > 1e-2 * s.max()))[0] - y_
                    return tmp_.dot(tmp_)

                result = minimize(lambda x: getRSS(y, x), (b, log(d), log(r)),
                                  bounds=((0 if b_nonneg else None, None),
                                          (None, -1e-4), (None, -1e-3)), method='L-BFGS-B',
                                  options={'gtol': 1e-04, 'maxiter': 10, 'ftol': 1e-05})
                if abs(result['x'][1] - log(d)) < 1e-3:
                    g_converged = True
                b, ld, lr = result['x']
                d, r = exp(ld), exp(lr)
                g = (d + r, -d * r)
                c, s = onnls(y - b, g, lam=lam, mask=mask,
                             shift=shift, window=window, tol=tol)
                db = max(np.mean(y - c), 0 if b_nonneg else -np.inf) - b
                b += db
                lam -= db

    if penalty == 0:
        def c4smin(y_, s_, s_min):
            ls = np.append(np.where(s_ > s_min)[0], T)
            tmp = np.zeros_like(s_)
            l = ls[0]
            tmp[:l] = max(0, np.exp(log(d) * np.arange(l)).dot(y_[:l]) * (1 - d * d)
                          / (1 - d**(2 * l))) * np.exp(log(d) * np.arange(l))
            for i, f in enumerate(ls[:-1]):
                l = ls[i + 1] - f
                tmp[f] = (g11[:l].dot(y_[f:f + l])
                          - g11g12[l - 1] * tmp[f - 1]) / g11g11[l - 1]
                tmp[f + 1:f + l] = g11[1:l] * tmp[f] + g12[1:l] * tmp[f - 1]
            return tmp

        spikesizes = np.sort(s[s > 1e-6])
        i = len(spikesizes) // 2
        l = 0
        u = len(spikesizes) - 1
        while u - l > 1:
            s_min = spikesizes[i]
            tmp = c4smin(y - b, s, s_min)
            res = y - b - tmp
            RSS = res.dot(res)
            if RSS < thresh or i == 0:
                l = i
                i = (l + u) // 2
                res0 = tmp
            else:
                u = i
                i = (l + u) // 2
        if i > 0:
            c = res0
            s = np.append([0, 0], c[2:] - g[0] * c[1:-1] - g[1] * c[:-2])

    return c, s, b, g, lam


# functions to estimate AR coefficients and sn from
# https://github.com/agiovann/Constrained_NMF.git
def estimate_parameters(y, p=2, range_ff=[0.25, 0.5], method='mean', lags=10, fudge_factor=1., nonlinear_fit=False):
    """Estimate noise standard deviation and AR coefficients."""
    sn = GetSn(y, range_ff, method)
    g = estimate_time_constant(y, p, sn, lags, fudge_factor, nonlinear_fit)
    return g, sn


def estimate_time_constant(y, p=2, sn=None, lags=10, fudge_factor=1., nonlinear_fit=False):
    """Estimate AR model parameters through the autocovariance function."""

    if sn is None:
        sn = GetSn(y)

    lags += p
    y = y - y.mean()
    xc = np.array([y[i:].dot(y[:-i if i else None]) for i in range(1 + lags)]) / len(y)

    if nonlinear_fit and p <= 2:
        xc[0] -= sn**2
        g1 = xc[:-1].dot(xc[1:]) / xc[:-1].dot(xc[:-1])
        if p == 1:
            def func(x, a, g_):
                return a * g_**x
            popt, _ = curve_fit(func, list(range(len(xc))), xc, (xc[0], g1))
            return popt[1:2] * fudge_factor
        if p == 2:
            def func(x, a, d, r):
                return a * (d**(x + 1) - r**(x + 1) / (1 - r**2) * (1 - d**2))
            popt, _ = curve_fit(func, list(range(len(xc))), xc, (xc[0], g1, .1))
            d, r = popt[1:]
            d *= fudge_factor
            return np.array([d + r, -d * r])

    A = scipy.linalg.toeplitz(xc[np.arange(lags)],
                              xc[np.arange(p)]) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[1:, np.newaxis], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    gr = np.real((gr + gr.conjugate()) / 2.)
    gr[gr > 1] = 0.95
    gr[gr < 0] = 0.15
    g = np.poly(fudge_factor * gr)
    g = -g[1:]

    return g.flatten()


def GetSn(y, range_ff=[0.25, 0.5], method='mean'):
    """Estimate noise standard deviation from the power spectral density."""

    ff, Pxx = scipy.signal.welch(y)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1, ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind_: np.sqrt(np.mean(Pxx_ind_ / 2)),
        'median': lambda Pxx_ind_: np.sqrt(np.median(Pxx_ind_ / 2)),
        'logmexp': lambda Pxx_ind_: np.sqrt(np.exp(np.mean(np.log(Pxx_ind_ / 2))))
    }[method](Pxx_ind)

    return sn
