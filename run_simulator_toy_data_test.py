import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dsd_simulator_edited import doppler_spectrum
from dsd_simulator_edited import linear_psd_zeroth_first_moments
from dsd_simulator_edited import generate_balanced_params
from dsd_simulator_edited import generate_dataset


#-----------------------------------------------------------------------
# Generate toy dataset
#-----------------------------------------------------------------------
                                       
rng = np.random.default_rng(11)     # Random generator for reproducibility 
n_cfg = 100     # Number of PSD configurations
cloud_only = rng.permutation(n_cfg) < 20        # 20 cloud-only 

configs = []

# np.random.default_rng.choice() takes the exact given numbers
# np.random.default_rng.uniform samples log-uniformly between given numbers

for i in range(n_cfg):
    Nc = rng.choice([75, 200]) * 1e6     # Cloud number concentration [m^-3]
    rc = rng.choice([5, 10]) * 1e-6      # Cloud median radius [m]
    if cloud_only[i]:
        Nd = 0.0
    else:
        Nd = 10**rng.uniform(np.log10(0.005), np.log10(0.3)) * 1e6   
    rd = rng.uniform(26, 150) * 1e-6   # Drizzle median radius [m]     
    configs.append(((Nc, 0.35, rc), (Nd, 0.35, rd)))

#print(configs)


#-----------------------------------------------------------------------
# Generate Doppler spectra and calculate spectral moments
# -----------------------------------------------------------------------

# Storage for PSDs, Doppler spectra, and metadata
psds = []
spectra = []
metadata = []

for i, (cloud, drizzle) in enumerate(configs):

    np.random.seed(i)

    # Run forward model
    result = doppler_spectrum(cloud, drizzle, sigma_t=0.1, w=0.0, SNR_dB=10, Z_max_db=50.0, navg=20, v_max=5.0, nv=512)

    (r, Nc, Nd, v, Sw, S_noisy, S_hs, S_hs2, chi, energy_ok, thr, thr2, Sw_cloud, Sw_drizzle, SQ_cloud, SQ_drizzle) = result

    # Store PSD configurations and their corresponding Doppler spectra
    psds.append({
    "spectrum_id": i,
    "r": r,
    "Nc": Nc,
    "Nd": Nd,
    "combined_psd": Nc + Nd,
    "v": v,
    "Sw": Sw,
    "Sw_cloud": Sw_cloud,
    "Sw_drizzle": Sw_drizzle
})

    # Calculate moments
    m_clean = linear_psd_zeroth_first_moments(Sw, v)
    m_hs = linear_psd_zeroth_first_moments(S_hs, v)
    m_hs2 = linear_psd_zeroth_first_moments(S_hs2, v)

    # Store spectra
    spectra.append({
        "clean": Sw,
        "noisy": S_noisy,
        "hs": S_hs,
        "hs2": S_hs2
    })

    # Store parameters and moments
    metadata.append({
        "spectrum_id": i,
        "cloud_N_cm3":   round(cloud[0]*1e-6, 3),   # for N [cm^-3]
        "cloud_N": round(cloud[0], 3),
        "cloud_sigma": cloud[1],
        "cloud_r_um":    round(cloud[2]*1e6, 3),   # for r [\mu m]
        "cloud_r": round(cloud[2], 3),
        "drizzle_N_cm3": round(drizzle[0]*1e-6, 3),   # for N [cm^-3]
        "drizzle_N": round(drizzle[0], 3),
        "drizzle_sigma": drizzle[1],
        "drizzle_r_um":  round(drizzle[2]*1e6, 3),   # for r [\mu m]
        "drizzle_r": round(drizzle[2], 3),
        "drizzle_present": int(drizzle[0] > 0),
        "chi": float(chi),
        "energy_ok": energy_ok,

        "Z_clean": float(np.asarray(10*np.log10(m_clean[0])).item()),
        "v_clean": float(np.asarray(m_clean[1]).item()),
        "sw_clean": float(np.asarray(m_clean[2]).item()),
        "skew_clean": float(np.asarray(m_clean[3]).item()),
        "kurt_clean": float(np.asarray(m_clean[4]).item()),

        "Z_hs": float(np.asarray(10*np.log10(m_hs[0])).item()),
        "v_hs": float(np.asarray(m_hs[1]).item()),
        "sw_hs": float(np.asarray(m_hs[2]).item()),
        "skew_hs": float(np.asarray(m_hs[3]).item()),
        "M4_hs": float(np.asarray(m_hs[4]).item()),

        "Z_hs2": float(np.asarray(10*np.log10(m_hs2[0])).item()),
        "v_hs2": float(np.asarray(m_hs2[1]).item()),
        "sw_hs2": float(np.asarray(m_hs2[2]).item()),
        "skew_hs2": float(np.asarray(m_hs2[3]).item()),
        "kurt_hs2": float(np.asarray(m_hs2[4]).item())
    })


#-----------------------------------------------------------------------
# Save and load datasets
#-----------------------------------------------------------------------

# Save moment data for the different drizzle radius cutoffs
# Note: have to always first set the gap accordingly in the dsd_simulator.py, then re-run every time with corresponding files commented out

#df = pd.DataFrame(metadata)
#df.to_csv("moments_gap26.csv", index=False)
#df.to_csv("moments_gap30.csv", index=False)
#df.to_csv("moments_gap35.csv", index=False)
#df.to_csv("moments_Z00.csv", index=False)
#df.to_csv("moments_Z50.csv", index=False)

df_26 = pd.read_csv("moments_gap26.csv")
df_30 = pd.read_csv("moments_gap30.csv")
df_35 = pd.read_csv("moments_gap35.csv")
df_capped = pd.read_csv("moments_Z00.csv")
df_uncapped = pd.read_csv("moments_Z50.csv")

# Save Doppler spectra for the selected drizzle radius cutoff

gap = 35
np.savez(f"spectra_gap{gap}.npz",
         v=psds[0]["v"],
         Sw=np.array([p["Sw"] for p in psds]),
         Sw_cloud=np.array([p["Sw_cloud"] for p in psds]),
         Sw_drizzle=np.array([p["Sw_drizzle"] for p in psds]))


#-----------------------------------------------------------------------
# Plot an overview of the toy dataset composition
#-----------------------------------------------------------------------

df = df_35      
dd = df[df["drizzle_present"] == 1]     # drizzle cases
chi_db = 10*np.log10(dd["chi"])

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# (a) drizzle parameter space, coloured by chi
sc = axs[0].scatter(dd["drizzle_r_um"], dd["drizzle_N_cm3"], c=chi_db, s=25, cmap="Greys")
axs[0].set_yscale("log")
axs[0].set_xlabel("Drizzle $r_m$ [µm]")
axs[0].set_ylabel("Drizzle $N_t$ [cm$^{-3}$]")
axs[0].set_title("Drizzle parameters (colour: $\\chi$ [dB])")
plt.colorbar(sc, ax=axs[0], label=r"$\chi$ = Z$_c$/Z$_d$ [dB]")
axs[0].grid(alpha=0.3)

# (b) the four cloud combinations and how often each occurs
counts = df.groupby(["cloud_N_cm3", "cloud_r_um"]).size()
labels = [f"{int(n)} cm$^{{-3}}$\n{int(r)} µm" for n, r in counts.index]
axs[1].bar(labels, counts.values, color="royalblue")
axs[1].set_ylabel("Number of spectra")
axs[1].set_title("Cloud combinations ($N_c$, $r_c$)")
axs[1].grid(alpha=0.3, axis="y")

# (c) chi distribution plus the cloud-only share
axs[2].hist(chi_db, bins=20, color="mediumseagreen", edgecolor="k")
axs[2].set_xlabel(r"$\chi$ [dB]")
axs[2].set_ylabel("Number of spectra")
n_co = (df["drizzle_present"] == 0).sum()
axs[2].set_title(f"Reflectivity ratio (no cloud-only cases)")
axs[2].grid(alpha=0.3)

plt.tight_layout()
fig.suptitle("Overview of the toy dataset's composition")
plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig("e4_toy_dataset_overview.png", dpi=300)
#plt.show()


#-----------------------------------------------------------------------
# Plot combined PSDs for all configurations
#-----------------------------------------------------------------------

plt.figure(figsize=(10, 6))

r_um = psds[0]["r"] * 1e6

for psd in psds:
    plt.plot(r_um, psd["combined_psd"] * 1e-12, alpha=0.2, linewidth=0.8)

plt.axvline(25, linestyle="--", color="black", label="Cloud cutoff")
plt.axvline(26, linestyle=":", color="red", label="Drizzle cutoff")
plt.xlabel("Particle radius [µm]")
plt.ylabel(r"$dN/dr$ [cm$^{-3}$ µm$^{-1}$]")
plt.title("Combined PSDs for 100 configurations, 1 µm gap")
plt.xlim(15, 100)
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("e4_psd_1mu.png", dpi=300)
#plt.show()


#-----------------------------------------------------------------------
# Plot Doppler spectra for all configurations
#-----------------------------------------------------------------------

# Sort spectra: cloud-only first, then ascending drizzle median radius

sort_order = sorted(
    range(len(configs)),
    key=lambda i: (
        configs[i][1][0] > 0,  # False (cloud-only) comes first
        configs[i][1][2] ) )      # Drizzle median radius

    
# Plot Doppler spectra in sorted order

fig, axes = plt.subplots(10, 10, figsize=(20, 16), sharex=True, sharey=True)

for position, idx in enumerate(sort_order):
    ax = axes.flat[position]
    v = psds[idx]["v"]
    ax.plot(v, 10*np.log10(psds[idx]["Sw"] + 1e-20), label="Total")
    ax.plot(v, 10*np.log10(psds[idx]["Sw_cloud"] + 1e-20), "--", label="Cloud")
    ax.plot(v, 10*np.log10(psds[idx]["Sw_drizzle"] + 1e-20), ":",  label="Drizzle")
    ax.set_title(f"ID {idx} | rd = {configs[idx][1][2]*1e6:.1f} µm", fontsize=7 )
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-50, 5)

fig.supxlabel("Doppler velocity [m/s]")
fig.supylabel("Spectral power [dB]")
fig.suptitle("Doppler spectra sorted by drizzle median radius, 1 µm gap")
plt.tight_layout()
#plt.savefig("e4_extra_doppler_spectra_1mu.png", dpi=300)
#plt.show()


#-----------------------------------------------------------------------
# Compare selected Doppler spectra for different radius gaps
#-----------------------------------------------------------------------

# Load previousl saved Doppler spectra

def load(filename):
    d = np.load(filename)
    return [{"v": d["v"], "Sw": d["Sw"][i], "Sw_cloud": d["Sw_cloud"][i],
             "Sw_drizzle": d["Sw_drizzle"][i]} for i in range(len(d["Sw"]))]

psds_26 = load("spectra_gap26.npz")
psds_35 = load("spectra_gap35.npz")

# Select specific Spectra ID for comparison
selected_ids = [91, 5, 24]

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)

for col, idx in enumerate(selected_ids):

    cloud, drizzle = configs[idx]

    Nc, _, rc = cloud
    Nd, _, rd = drizzle

    title = (f"ID {idx}: " 
             f"$r_c$ = {rc*1e6:.1f} µm, "
             f"$r_d$ = {rd*1e6:.1f} µm" )

    axes[0, col].set_title(title)

    # 1 µm gap
    axes[0, col].plot(psds_26[idx]["v"], 10*np.log10(psds_26[idx]["Sw"] + 1e-20), label="Total" )
    axes[0, col].plot(psds_26[idx]["v"], 10*np.log10(psds_26[idx]["Sw_cloud"] + 1e-20), "--", label="Cloud") 
    axes[0, col].plot(psds_26[idx]["v"], 10*np.log10(psds_26[idx]["Sw_drizzle"] + 1e-20), ":", label="Drizzle")

    # 10 µm gap
    axes[1, col].plot( psds_35[idx]["v"], 10*np.log10(psds_35[idx]["Sw"] + 1e-20), label="Total"  )
    axes[1, col].plot(psds_35[idx]["v"], 10*np.log10(psds_35[idx]["Sw_cloud"] + 1e-20), "--", label="Cloud")
    axes[1, col].plot(psds_35[idx]["v"], 10*np.log10(psds_35[idx]["Sw_drizzle"] + 1e-20), ":", label="Drizzle")

    axes[0, col].set_xlim(-1.5, 2.5)
    axes[0, col].set_ylim(-50, 5)

axes[0, 0].set_ylabel("1 µm gap\nSpectral power [dB]")
axes[1, 0].set_ylabel("10 µm gap\nSpectral power [dB]")

fig.supxlabel("Doppler velocity [m/s]")
fig.suptitle("Comparison of Doppler spectra for different cloud-drizzle radius gaps")

axes[0, 0].legend(fontsize=8)

plt.tight_layout()
plt.savefig("e4_extra_spectra_detailed.png", dpi=300)
plt.show()


#-----------------------------------------------------------------------
# Analyse skewness and kurtosis as a function of spectrum ID
#-----------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Skewness
axes[0].plot(df_26["spectrum_id"], df_26["skew_clean"],
             label="26 µm cutoff", marker="o", alpha=0.5, markersize=3, color="royalblue")
axes[0].plot(df_30["spectrum_id"], df_30["skew_clean"],
             label="30 µm cutoff", marker="o", alpha=0.5, markersize=3, color="darkorange")
axes[0].plot(df_35["spectrum_id"], df_35["skew_clean"],
             label="35 µm cutoff", marker="o", alpha=0.5, markersize=3, color="mediumseagreen")

axes[0].set_xlabel("Spectrum ID")
axes[0].set_ylabel("Skewness")
axes[0].set_title("Skewness per spectrum")
axes[0].legend()
axes[0].grid()

# Kurtosis
axes[1].plot(df_26["spectrum_id"], df_26["kurt_clean"],
             label="26 µm cutoff", marker="o", alpha=0.5, markersize=3, color="royalblue")
axes[1].plot(df_30["spectrum_id"], df_30["kurt_clean"],
             label="30 µm cutoff", marker="o", alpha=0.5, markersize=3, color="darkorange")
axes[1].plot(df_35["spectrum_id"], df_35["kurt_clean"],
             label="35 µm cutoff", marker="o", alpha=0.5, markersize=3, color="mediumseagreen")

axes[1].set_xlabel("Spectrum ID")
axes[1].set_ylabel("Kurtosis")
axes[1].set_title("Kurtosis per spectrum")
axes[1].legend()
axes[1].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Effect of drizzle radius cutoff on spectral moments Clean Spectra", fontsize=14)
#plt.savefig("e4_total_moments_drizzle_gap_clean_spectra_id2.png", dpi=300)
#plt.show()


#-----------------------------------------------------------------------
# Calculate differences in skewness and kurtosis relative to the 35 µm cutoff
#-----------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Skewness differences
axes[0].plot(df_26["spectrum_id"],
             df_26["skew_clean"] - df_35["skew_clean"],
             label="26 - 35 µm", marker="o", markersize=3, color="royalblue")

axes[0].plot(df_30["spectrum_id"],
             df_30["skew_clean"] - df_35["skew_clean"],
             label="30 - 35 µm", marker="o", markersize=3, color="darkorange")

axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Spectrum ID")
axes[0].set_ylabel("Skewness difference")
axes[0].set_title("Skewness differences per spectrum")
axes[0].legend()
axes[0].grid()

# Kurtosis differences
axes[1].plot(df_26["spectrum_id"],
             df_26["kurt_clean"] - df_35["kurt_clean"],
             label="26 - 35 µm", marker="o", markersize=3, color="royalblue")

axes[1].plot(df_30["spectrum_id"],
             df_30["kurt_clean"] - df_35["kurt_clean"],
             label="30 - 35 µm", marker="o", markersize=3, color="darkorange")

axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("Spectrum ID")
axes[1].set_ylabel("Kurtosis difference")
axes[1].set_title("Kurtosis differences per spectrum")
axes[1].legend()
axes[1].grid()

plt.tight_layout(rect=[0, 0, 1, 0.96])
fig.suptitle("Differences in spectral moments relative to the 35 µm cutoff Clean Spectra", fontsize=14)
#plt.savefig("e4_relative_differences_moments_drizzle_gap_clean_spectra_id2.png", dpi=300)
#plt.show()


#-----------------------------------------------------------------------
# Analyse moment differences as a function of drizzle median radius
#-----------------------------------------------------------------------

# Use 35 \mu m as reference
moments = [
    ("Z_clean", "Reflectivity"),
    ("v_clean", "Mean Doppler velocity"),
    ("sw_clean", "Spectral width"),
    ("skew_clean", "Skewness"),
    ("kurt_clean", "Kurtosis")
]


has_dr = df_35["drizzle_present"] == 1
rd = df_35["drizzle_r_um"][has_dr]          # x-axis: drizzle median radius in µm

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (column, title) in enumerate(moments):
    diff_26 = (df_26[column] - df_35[column])[has_dr]
    diff_30 = (df_30[column] - df_35[column])[has_dr]

    axes[i].plot(rd, diff_26, "o", ms=3, color="royalblue", label="1 µm gap")
    axes[i].plot(rd, diff_30, "o", ms=3, color="darkorange", alpha=0.7, label="5 µm gap")
    axes[i].axhline(0, color="black", lw=0.8)
    axes[i].set_xlabel("Drizzle $r_m$ [µm]")
    axes[i].set_ylabel("Difference to 10 µm gap")
    axes[i].set_title(title)
    axes[i].grid(alpha=0.3)
    axes[i].legend(loc="best")

axes[-1].axis("off")
plt.tight_layout()

fig.suptitle("Sensitivity of the clean Spectral Moments to Radius Gap\nas a Function of Drizzle Median Radius", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

#plt.savefig("e4_clean_differences_vs_rd.png", dpi=300)
plt.show()


#-----------------------------------------------------------------------
# Inspect configurations with large kurtosis differences
#-----------------------------------------------------------------------

column = "kurt_clean"

# Select drizzle cases with median radii between 26 and 55 µm
mask = df_35["drizzle_present"].eq(1) & df_35["drizzle_r_um"].between(26, 55)

# Collect PSD configuration and difference information
check = df_35.loc[mask].copy()

check["diff_1um"] = (df_26[column] - df_35[column])[mask]
check["diff_5um"] = (df_30[column] - df_35[column])[mask]

check["abs_diff_1um"] = check["diff_1um"].abs()
check["abs_diff_5um"] = check["diff_5um"].abs()

# Select relevant columns
columns = [
    "spectrum_id",
    "cloud_N_cm3", "cloud_r_um",
    "drizzle_N_cm3", "drizzle_r_um",
    "diff_1um", "diff_5um",
    "abs_diff_1um", "abs_diff_5um" ]

columns = [c for c in columns if c in check.columns]


# Print configurations with the largest differences
#print(check.sort_values("abs_diff_1um", ascending=False)[columns].head(10).to_string(index=False))


#Print the PSD parameters and clean kurtosis for spectrum ID 9 across the three drizzle radius cutoffs

for name, df in [("26 µm cutoff", df_26),
                 ("30 µm cutoff", df_30),
                 ("35 µm cutoff", df_35)]:

    row = df.loc[df["spectrum_id"] == 9].iloc[0]

    #print(f"\n{name}")
    #print("Cloud N:", row["cloud_N_cm3"])
    #print("Cloud r:", row["cloud_r_um"])
    #print("Drizzle N:", row["drizzle_N_cm3"])
    #print("Drizzle r:", row["drizzle_r_um"])
    #print("Clean kurtosis:", row["kurt_clean"])


# Prints the drizzle median radius for one spectrum on the three different radius gap sets
# Makes sure the same PSD configuration is used

#for name, df in [("26", df_26), ("30", df_30), ("35", df_35)]:
    #print(name, df.loc[df["spectrum_id"] == 9, "drizzle_r_um"].iloc[0])



#-----------------------------------------------------------------------
# Compare capped and uncapped spectral moments
#-----------------------------------------------------------------------

for moment in ["Z_clean", "v_clean", "sw_clean", "skew_clean", "kurt_clean",
               "Z_hs2", "v_hs2", "sw_hs2", "skew_hs2", "kurt_hs2"]:

    difference = df_uncapped[moment] - df_capped[moment]

    #print(f"{moment}: max absolute difference = {difference.abs().max():.6f}")