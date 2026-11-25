import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed

K2 = 0.93


# =================================================
# Lognormal PSD (vectorized, same interface)
# =================================================
def lognormal_N(r, Nt, sigma, r_m):
    lr = np.log(r)
    return (Nt / (r * sigma * np.sqrt(2*np.pi))) * \
           np.exp(-(lr - np.log(r_m))**2 / (2*sigma**2))


# =================================================
# Velocity (vectorized)
# =================================================
def velocity(r):
    g = 9.81
    rho_p = 1000.0
    rho_air = 1.293
    eta = 1.766e-5

    A = np.array([
        -3.18657, 0.992696, -1.53193e-3,
        -9.87059e-4, -5.78878e-4,
        8.55176e-5, -3.27815e-6
    ])

    Nd = (4/3) * ((g*rho_air*rho_p)/(eta**2)) * (2*r)**3
    logNd = np.log(Nd)

    # Horner's method
    poly = np.zeros_like(logNd)
    for a in A[::-1]:
        poly = poly * logNd + a

    return np.exp(poly) * eta / (rho_air * (2*r))


# =================================================
# PSD (vectorized, same behavior)
# =================================================
def bimodal_psd(r, cloud, drizzle):
    Nc = lognormal_N(r, cloud[0], cloud[1], cloud[2])
    Nd = lognormal_N(r, drizzle[0], drizzle[1], drizzle[2])

    Nc[r > 25e-6] = 0.0
    Nd[(r < 35e-6) | (r > 250e-6)] = 0.0

    return Nc, Nd


# =================================================
# HS filter (fast vectorized)
# =================================================
def hs_noise_filter(S):
    S_sorted = np.sort(S)

    cumsum = np.cumsum(S_sorted)
    cumsum2 = np.cumsum(S_sorted**2)

    n = np.arange(1, len(S_sorted)+1)

    mean = cumsum / n
    var = cumsum2 / n - mean**2

    idx = np.argmax(var > mean**2)

    if var[idx] <= mean[idx]**2:
        threshold = S_sorted[-1]
    else:
        threshold = S_sorted[idx]

    S_filtered = np.where(S >= threshold, S, 0.0)

    return S_filtered, threshold


def _eta_to_sq(
    eta_comp: np.ndarray,
    v_r: np.ndarray,
    v: np.ndarray,
    dr: float,
) -> np.ndarray:
    """Ideal Doppler spectrum ∂Z/∂v from component ``η(r)`` (no turbulence, no ``w``)."""
    idx = np.argsort(v_r)
    Z_r = np.cumsum(eta_comp[idx]) * dr
    Z_interp = np.interp(v, v_r[idx], Z_r)
    return np.gradient(Z_interp, v)


def _eta_to_sw_after_w(
    eta_comp: np.ndarray,
    v_r: np.ndarray,
    v: np.ndarray,
    dr: float,
    dv: float,
    sigma_t: float,
    w: float,
) -> np.ndarray:
    """Map linear reflectivity ``eta``(r) to Doppler-axis PSD after turbulence + ``w``."""
    idx = np.argsort(v_r)
    Z_r = np.cumsum(eta_comp[idx]) * dr
    Z_interp = np.interp(v, v_r[idx], Z_r)
    SQ = np.gradient(Z_interp, v)
    St = gaussian_filter1d(SQ, sigma_t / dv)
    v_shifted = v - w
    return np.interp(v, v_shifted, St, left=0.0, right=0.0)


def linear_psd_zeroth_first_moments(
    spectrum: np.ndarray,
    velocity_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-row ∫S dv and ⟨v⟩ = ∫vS dv / ∫S dv (trapezoid), NaN/negative → 0.

    Matches the convention used in ``decompose_doppler_spectra`` for measured
    linear PSD moments (``np.trapezoid`` on the velocity axis).
    """
    s = np.atleast_2d(np.asarray(spectrum, dtype=np.float64))
    v = np.asarray(velocity_axis, dtype=np.float64).ravel()
    n, nv = s.shape
    if v.size != nv:
        raise ValueError("velocity_axis length must match spectrum width")
    
    s_clean = np.where(np.isfinite(s) & (s > 0.0), s, 0.0)
    _trapz = getattr(np, "trapezoid", np.trapz)

    z_lin = _trapz(s_clean, v, axis=1)

    with np.errstate(invalid="ignore"):
        v_mean = _trapz(v * s_clean, v, axis=1) / z_lin
        
    v_mean = np.where(np.isfinite(z_lin) & (z_lin > 0.0), v_mean, np.nan)
    z_lin = np.where(np.isfinite(z_lin) & (z_lin > 0.0), z_lin, np.nan)
    return z_lin, v_mean


# =================================================
# FORWARD MODEL (same signature)
# =================================================
def doppler_spectrum(
        cloud,
        drizzle,
        nr=4000,
        nv=512,
        v_max=6.0,
        sigma_t=0.2,
        navg=8,
        SNR_dB=20.0,
        Z_max_db=0.0,
        w=0.0
):

    # ---- grids
    r = np.linspace(1e-6, 4e-3, nr, dtype=np.float32)
    dr = r[1] - r[0]

    v = np.linspace(-v_max, v_max, nv, dtype=np.float32)
    dv = v[1] - v[0]

    # ---- PSD
    Nc, Nd = bimodal_psd(r, cloud, drizzle)

    # ---- velocity
    v_r = -velocity(r)
    v_r = np.clip(v_r, -v_max, v_max)

    # ---- reflectivity (components + total; same Z_max cap scale on all)
    eta_c = Nc * K2 * (2 * r) ** 6 * 1e18
    eta_d = Nd * K2 * (2 * r) ** 6 * 1e18
    eta = eta_c + eta_d

    Z = np.sum(eta) * dr
    Zc = np.sum(eta_c) * dr
    Zd = np.sum(eta_d) * dr

    chi = Zc / Zd if Zd > 0 else np.inf

    # ---- reflectivity cap
    Z_db = 10*np.log10(Z + 1e-20)
    if Z_db > Z_max_db:
        scale = 10**(Z_max_db/10) / Z
        eta_c *= scale
        eta_d *= scale
        eta *= scale
        Z *= scale

    # ---- noise
    SNR_lin = 10**(SNR_dB / 10)
    PN_total = Z / SNR_lin

    # ---- mapping r → v (total spectrum for energy diagnostic)
    idx = np.argsort(v_r)
    Z_r = np.cumsum(eta[idx]) * dr

    Z_interp = np.interp(v, v_r[idx], Z_r)
    SQ = np.gradient(Z_interp, v)

    # ---- turbulence + vertical velocity (total + cloud/drizzle components)
    Sw = _eta_to_sw_after_w(eta, v_r, v, dr, dv, sigma_t, w)
    Sw_cloud = _eta_to_sw_after_w(eta_c, v_r, v, dr, dv, sigma_t, w)
    Sw_drizzle = _eta_to_sw_after_w(eta_d, v_r, v, dr, dv, sigma_t, w)
    SQ_cloud = _eta_to_sq(eta_c, v_r, v, dr)
    SQ_drizzle = _eta_to_sq(eta_d, v_r, v, dr)

    # ---- noise sampling
    PN_fft = PN_total / (nv * dv)

    rand = np.random.rand(navg, nv)
    S_noisy = np.mean((Sw + PN_fft) * (-np.log(rand)), axis=0)

    # ---- HS filtering
    S_hs, thr = hs_noise_filter(S_noisy)

    # ---- energy check
    energy_ok = abs(Z - np.sum(SQ)*dv) / (Z + 1e-20) < 1e-2

    return r, Nc, Nd, v, Sw, S_noisy, S_hs, chi, energy_ok, thr, Sw_cloud, Sw_drizzle, SQ_cloud, SQ_drizzle


# =================================================
# Balanced parameters (same API)
# =================================================
def generate_balanced_params():

    chi_db = np.random.uniform(-30, 30)
    chi_target = 10**(chi_db / 10)

    cloud = (
        10**np.random.uniform(6, 8),
        np.random.uniform(0.2, 0.5),
        np.random.uniform(5e-6, 15e-6)
    )

    drizzle = (
        10**np.random.uniform(2, 5),
        np.random.uniform(0.3, 0.6),
        np.random.uniform(30e-6, 200e-6)
    )

    # faster coarse grid
    r = np.linspace(1e-6, 4e-3, 1200)
    dr = r[1] - r[0]

    Nc, Nd = bimodal_psd(r, cloud, drizzle)

    eta_c = Nc*K2*(2*r)**6*1e18
    eta_d = Nd*K2*(2*r)**6*1e18

    Zc = np.sum(eta_c)*dr
    Zd = np.sum(eta_d)*dr

    if Zd > 0:
        drizzle = (drizzle[0]*(Zc/(Zd*chi_target)), drizzle[1], drizzle[2])

    return cloud, drizzle


# =================================================
# DATASET GENERATION (parallel, same API)
# =================================================
def generate_dataset(
        n,
        filename,
        nr=4000,
        nv=512,
        v_max=6.0,
        sigma_t=0.2,
        navg=8,
        SNR_dB=20.0,
        Z_max_db=0.0,
        w=0.0,
):

    dtype = np.float32

    def simulate(_):
        cloud, drizzle = generate_balanced_params()

        return doppler_spectrum(
            cloud, drizzle,
            nr=nr, nv=nv,
            v_max=v_max,
            sigma_t=sigma_t,
            navg=navg,
            SNR_dB=SNR_dB,
            Z_max_db=Z_max_db,
            w=w
        )

    results = Parallel(n_jobs=-1)(
        delayed(simulate)(i) for i in range(n)
    )

    v_coord = np.linspace(-v_max, v_max, nv, dtype=np.float64)
    sw = np.array([r[4] for r in results], dtype=dtype)
    s_noisy = np.array([r[5] for r in results], dtype=dtype)
    s_hs = np.array([r[6] for r in results], dtype=dtype)
    sw_cloud = np.array([r[10] for r in results], dtype=dtype)
    sw_drizzle = np.array([r[11] for r in results], dtype=dtype)
    sq_cloud = np.array([r[12] for r in results], dtype=dtype)
    sq_drizzle = np.array([r[13] for r in results], dtype=dtype)

    m0_noisy, m1_noisy = linear_psd_zeroth_first_moments(s_noisy, v_coord)
    m0_hs, m1_hs = linear_psd_zeroth_first_moments(s_hs, v_coord)
    m0_sw, m1_sw = linear_psd_zeroth_first_moments(sw, v_coord)

    # build dataset
    ds = xr.Dataset(
        {
            "Sw": (("sample","velocity"), sw),
            "S_noisy": (("sample","velocity"), s_noisy),
            "S_hs": (("sample","velocity"), s_hs),
            "Sw_cloud": (("sample","velocity"), sw_cloud),
            "Sw_drizzle": (("sample","velocity"), sw_drizzle),
            "SQ_cloud": (("sample","velocity"), sq_cloud),
            "SQ_drizzle": (("sample","velocity"), sq_drizzle),
            "Nc": (("sample","radius"), np.array([r[1] for r in results], dtype=dtype)),
            "Nd": (("sample","radius"), np.array([r[2] for r in results], dtype=dtype)),
            "chi": (("sample",), np.array([r[7] for r in results], dtype=dtype)),
            "noise_threshold": (("sample",), np.array([r[9] for r in results], dtype=dtype)),
            "energy_ok": (("sample",), np.array([r[8] for r in results])),
            "meas_Z0_S_noisy": (("sample",), m0_noisy.astype(np.float64)),
            "meas_vmean_S_noisy": (("sample",), m1_noisy.astype(np.float64)),
            "meas_Z0_S_hs": (("sample",), m0_hs.astype(np.float64)),
            "meas_vmean_S_hs": (("sample",), m1_hs.astype(np.float64)),
            "meas_Z0_Sw": (("sample",), m0_sw.astype(np.float64)),
            "meas_vmean_Sw": (("sample",), m1_sw.astype(np.float64)),
        },
        coords={
            "velocity": v_coord.astype(np.float32),
            "radius": np.linspace(1e-6, 4e-3, nr)
        }
    )
    _psd_u = "mm^6 m^-3 (m s)^-1"
    for _vn in ("Sw", "S_noisy", "S_hs", "Sw_cloud", "Sw_drizzle", "SQ_cloud", "SQ_drizzle"):
        ds[_vn].attrs["units"] = _psd_u
        if _vn.startswith("SQ_"):
            ds[_vn].attrs["long_name"] = "ideal linear Doppler PSD dZ/dv (no turbulence, no w)"
        else:
            ds[_vn].attrs["long_name"] = "linear Doppler PSD"
    ds.attrs["spectra_linear_units"] = _psd_u

    ds.to_netcdf(filename)
    print("✅ Saved:", filename)


# =================================================
# MAIN
# =================================================
if __name__ == "__main__":
    generate_dataset(
        n=2000,
        filename="doppler_dataset.nc",
        nr=3000,
        nv=512,
        v_max=5.0,
        sigma_t=0.25,
        navg=48,
        SNR_dB=20,
        w=0
    )
