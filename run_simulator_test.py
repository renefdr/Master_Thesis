import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from dsd_simulator_edited import doppler_spectrum
from dsd_simulator_edited import linear_psd_zeroth_first_moments
from dsd_simulator_edited import generate_balanced_params
from dsd_simulator_edited import generate_dataset

# -----------------------------------------------------------------------
# Initial cloud and drizzle configuration
# -----------------------------------------------------------------------

# PSD configurations
cloud   = (2e8, 0.35, 10e-6)       # Cloud: N = 200 cm^-3, r = 10 µm
drizzle = (1e4, 0.25, 50e-6)    # Drizzle: N = 0.01 cm^-3, r = 50 µm
#drizzle = (0, 0.25, 50e-6)        # Cloud-only spectrum


# np.random.seed(6)

r, Nc, Nd, v, Sw, S_noisy, S_hs, S_hs2, chi, energy_ok, thr, thr2, Sw_cloud, Sw_drizzle, SQ_cloud, SQ_drizzle = doppler_spectrum( cloud, drizzle, sigma_t=0.2, w=0.0, SNR_dB=10, navg=48, v_max=5.0, nv=512 )


# -----------------------------------------------------------------------
# Spectral moments
# -----------------------------------------------------------------------

# Moments of the clean spectrum
m0_clean, m1_clean, m2_clean, m3_clean, m4_clean = linear_psd_zeroth_first_moments(Sw, v)

# Moments of the Hildebrand-Sekhon filtered spectrum
m0_hs, m1_hs, m2_hs, m3_hs, m4_hs = linear_psd_zeroth_first_moments(S_hs, v)

# Moments of the updated HS-filtered spectrum following Kollias
m0_hs2, m1_hs2, m2_hs2, m3_hs2, m4_hs2 = linear_psd_zeroth_first_moments(S_hs2, v)

moments = pd.DataFrame({
    "Clean": [ 10 * np.log10(m0_clean).item(),
              m1_clean.item(),
              m2_clean.item(),
              m3_clean.item(),
              m4_clean.item() ],
    
    "HS": [ 10 * np.log10(m0_hs).item(),
           m1_hs.item(),
           m2_hs.item(),
           m3_hs.item(),
           m4_hs.item() ],
    
    "Updated HS": [ 10 * np.log10(m0_hs2).item(),
                   m1_hs2.item(),
                   m2_hs2.item(), 
                   m3_hs2.item(),
                   m4_hs2.item() ]}, 
                   
    index=[ "Reflectivity-related M0 (dB)",
           "Mean Doppler velocity",
           "Spectrum width",
           "Skewness",
           "Kurtosis" ])

print(moments.round(4).to_string())


# -----------------------------------------------------------------------
# Plot annotations
# -----------------------------------------------------------------------

psd_text = ( r"$\mathbf{Cloud\ PSD}$" + "\n"
            rf"$N_t = {cloud[0] * 1e-6:.0f}\ \mathrm{{cm}}^{{-3}}$" + "\n"
            rf"$\sigma = {cloud[1]:.2f}$" + "\n"
            rf"$r_m = {cloud[2] * 1e6:.1f}\ \mu\mathrm{{m}}$" + "\n\n"
            r"$\mathbf{Drizzle\ PSD}$" + "\n"
            rf"$N_t = {drizzle[0] * 1e-6:.3f}\ \mathrm{{cm}}^{{-3}}$" + "\n"
            rf"$\sigma = {drizzle[1]:.2f}$" + "\n"
            rf"$r_m = {drizzle[2] * 1e6:.1f}\ \mu\mathrm{{m}}$" )


# -----------------------------------------------------------------------
# Comparison of clean, noisy, and filtered Doppler spectra
# -----------------------------------------------------------------------

# For the different versions of the worklog, just set PSD configurations accordingly, as well as commenting out desired spectra

plt.figure(figsize=(8, 6))

plt.text( 0.2, 0.97, psd_text, transform=plt.gca().transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict( boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85 ) )
#plt.plot(v, 10 * np.log10(Sw + 1e-20), label="Clean", color="darkorange")
plt.plot(v, 10 * np.log10(S_noisy + 1e-20), label="Noisy", color="royalblue")
plt.plot(v, 10 * np.log10(S_hs + 1e-20), label="HS filtered", color="mediumseagreen", alpha=0.8)
plt.plot(v, 10 * np.log10(S_hs2 + 1e-20), label="HS filtered updated", color="red", alpha=0.4)
plt.xlabel("Doppler velocity [m/s]")
plt.ylabel("Spectral power [dB]")
plt.title("Doppler spectra")
plt.legend()
plt.ylim(-50, 5)
plt.grid(True, alpha=0.25)
plt.tight_layout()
# plt.savefig("test2.png", dpi=300)
plt.show()


# -----------------------------------------------------------------------
# Cloud and drizzle contributions to the Doppler spectrum
# -----------------------------------------------------------------------

plt.figure(figsize=(8, 6))

plt.text( 0.2, 0.97, psd_text, transform=plt.gca().transAxes,
    fontsize=9, verticalalignment="top", horizontalalignment="right",
    bbox=dict( boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.85 ) )
plt.plot(v, 10 * np.log10(Sw + 1e-20), label="clean", color="darkorange")
plt.plot(v, 10 * np.log10(Sw_cloud + 1e-20), label="clean cloud-only", color="turquoise")
plt.plot(v, 10 * np.log10(Sw_drizzle + 1e-20), label="clean drizzle-only", color="mediumpurple")
plt.xlabel("Doppler velocity [m/s]")
plt.ylabel("Spectral power [dB]")
plt.title("Cloud-only, drizzle-only and total clean Doppler spectra")
plt.ylim(-50, 5)
plt.legend()
plt.show()