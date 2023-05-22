import numpy as np
import netCDF4 as nc
from scipy.interpolate import interp1d
from interp_apollo import interp_example

ORIG_BANDS = np.array(
    [
        (680, 690),
        (717, 727),
        (760, 765),
        (820, 825),
        (896, 980),
        (1107, 1181),
        # (1268, 1268),
        (1268, 1269),
        (1307, 1508),
        (1800, 1960),
        (2480, 2500),
    ]
)

BANDS = np.array(
    [
        (680, 690),
        (717, 727),
        # (760, 765),
        (755, 770),
        (820, 825),
        (896, 980),
        (1107, 1181),
        # (1268, 1268),
        (1268, 1269),
        (1307, 1508),
        # (1800, 1960),
        (1785, 1980),
        # (2480, 2500),
        (2480, 2500),
    ]
)

DS_PATH = "ds_ASD.nc"


def _get_apollo():
    # composite
    data = np.genfromtxt("./Composite.txt", delimiter=",")
    wavs = data[:, 0]
    refl = data[:, 1]
    wavs = wavs[np.where(np.isfinite(refl))]
    refl = refl[np.where(np.isfinite(refl))]
    valid_wavs = [i for i in range(350, 2501)]
    valid_ids = np.where(np.in1d(wavs, valid_wavs))[0]
    wavs = wavs[valid_ids]
    refl = refl[valid_ids]
    return wavs, refl


def interpolate_apollo(ini, stop, wlens, xp, fp):
    apollo_w, apollo_refl = _get_apollo()
    ap_xp = np.concatenate([apollo_w[:ini], apollo_w[stop:]])
    ap_fp = np.concatenate([apollo_refl[:ini], apollo_refl[stop:]]).T
    ie = interp_example(
        xp,
        fp,
        ap_xp,
        ap_fp,
        wlens[ini:stop],
        method="linear",
        method_hr="linear",
    )
    return ie


def extrapolate_apollo(dini, dstop, ini, stop, wlens, xp, fp):
    apollo_w, apollo_refl = _get_apollo()
    ap_xp = apollo_w[dini:dstop]
    ap_fp = apollo_refl[dini:dstop].T
    ie = interp_example(
        xp,
        fp,
        ap_xp,
        ap_fp,
        wlens[ini:stop],
        method="linear",
        method_hr="linear",
    )
    return ie


def interpolate(ini, stop, wlens, refl, use_apollo=True):
    xp = np.concatenate([wlens[:ini], wlens[stop:]])
    fps = np.concatenate([refl[:ini], refl[stop:]]).T
    if not use_apollo:
        refl[ini:stop] = np.array([np.interp(wlens[ini:stop], xp, fp) for fp in fps]).T
    else:
        refl[ini:stop] = np.array(
            [interpolate_apollo(ini, stop, wlens, xp, fp) for fp in fps]
        ).T


def extrapolate(ini, stop, wlens, refl, use_apollo=True):
    if ini > 0:
        xp = wlens[:ini]
        fps = refl[:ini].T
        dini = 0
        dstop = ini
    else:
        xp = wlens[stop:]
        fps = refl[stop:].T
        dini = stop
        dstop = len(wlens)
    if not use_apollo:
        interps = [interp1d(xp, fp, "linear", fill_value="extrapolate") for fp in fps]
        refl[ini:stop] = np.array([interp(wlens[ini:stop]) for interp in interps]).T
    else:
        refl[ini:stop] = np.array(
            [extrapolate_apollo(dini, dstop, ini, stop, wlens, xp, fp) for fp in fps]
        ).T


def extrapolate_constant(ini, stop, wlens, refl):
    if ini > 0:
        val = np.mean(refl[max(0, ini - 25) : ini].T, axis=1)
    else:
        val = np.mean(refl[stop : min(len(wlens), stop + 25)].T, axis=1)
    vals = np.array([val for _ in range(ini, stop)])
    refl[ini:stop] = vals


def drift_refl(wlens, refl, wlen_drift):
    sep = int(wlen_drift - wlens[0])
    reflt = refl
    diff = reflt[sep, :] - reflt[sep - 1, :]
    refl[sep:, :] = refl[sep:, :] - diff
    return refl


def fix_pos_spec_errors(wlens, refl):
    fixes = {
        (17, 32): {"interps": [1098]},
        (31, 44): {
            "interps": [646, 657, 749, 753, 790, 858, 862, 888],
            "drifts": [(620, 621)],
        },
    }
    for extrs in fixes:
        fix = fixes[extrs]
        e0 = extrs[0] + 90
        ef = extrs[1] + 90
        if "interps" in fix:
            finterps = np.array(fix["interps"]) - int(wlens[0])
            for wlen in finterps:
                refl[wlen, e0 : ef + 1] = (
                    refl[wlen - 1, e0 : ef + 1] + refl[wlen + 1, e0 : ef + 1]
                ) / 2
        if "drifts" in fix:
            for drpair in fix["drifts"]:
                refl = drift_refl(wlens, refl, drpair[1])
    return refl


def replace_too_corrupted_mpas(wlens, refl):
    replacers = {
        3: (4, 10),
        16: (11, 15),
        44: (45, 59),
    }
    extra_replacers = {-i: (i, i) for i in range(60, 90)}
    replacers = {**replacers, **extra_replacers}
    for rep in replacers:
        exts = replacers[rep]
        for ext in range(exts[0], exts[1] + 1):
            refl[:, ext + 90] = refl[:, rep + 90]
    return refl


def main():
    ds = nc.Dataset(DS_PATH, "r+")
    wlens = ds["wavelength"][:].data
    refl = ds["reflectance"][:].data
    dolp = ds["polarization"][:].data
    refl = replace_too_corrupted_mpas(wlens, refl)
    ds["reflectance"][:] = refl
    return
    refl = drift_refl(wlens, refl, 1001)
    bands = BANDS - wlens[0]
    for b in bands:
        ini = int(b[0])
        stop = int(b[1] + 1)
        if ini > 0 and stop < len(wlens):
            interpolate(ini, stop, wlens, refl)
            interpolate(ini, stop, wlens, dolp, False)
        else:
            extrapolate(ini, stop, wlens, refl)
            extrapolate_constant(ini, stop, wlens, dolp)
    ds["reflectance"][:] = refl
    ds["polarization"][:] = dolp
    ds.close()


if __name__ == "__main__":
    main()
