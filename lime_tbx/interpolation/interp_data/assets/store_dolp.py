import pandas as pd
import netCDF4 as nc
import numpy as np

CSV_PATH = "./dolp.csv"
CSV_PROCESSED_PATH = "./dolp_processed.csv"
NC_PATH = "./ds_ASD.nc"


def store_dolp_original(df):
    df = df.sort_values("mpa").reset_index()
    wlens = [f"{i}" for i in range(350, 2501)]
    df = pd.DataFrame(df[wlens].T.values, index=wlens, columns=df["mpa"].values)
    # print(df.columns)
    """
    -76.50106211984625,  -49.85643929803059, -35.672538961350085,
    -21.45596162213788,  -8.428740090236671,  -8.382861300750376,
    7.657853637133816,   8.025454918140133,   19.76725036737047,
    22.232507280735188,   32.48366720563654,   36.68248175569178,
    44.97968174924686,    50.4848074684456,   57.10322592899022,
    64.223405195593,   77.21484487619131,   90.03967225776074]
    """
    df.drop(df.columns[-1], axis=1, inplace=True)  # Drop column 90º mpa
    mpa_group = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11]
    mpa_group_df = pd.DataFrame(mpa_group, columns=["mpa_group"])
    df_T = pd.merge(
        df.T.reset_index(), mpa_group_df, left_index=True, right_index=True
    )  # .set_index('index')
    df_T = (
        df_T.groupby("mpa_group")
        .mean()
        .reset_index()
        .drop("mpa_group", axis=1)
        .set_index("index")
    )
    df = df_T.T
    print(df)
    ds = nc.Dataset(NC_PATH, "r+")
    phases = ds["phase_angle"][:].data
    dolp_mpas = np.array(list(map(float, df.columns)))
    print(dolp_mpas)
    print(phases)
    mpas_ids = np.array([np.argmin(np.abs(p - dolp_mpas)) for p in phases])
    print(mpas_ids)
    dolps = np.array([df[df.columns[mpa]] for mpa in mpas_ids])
    print(dolps.shape)
    ds["polarisation"][:] = dolps.T
    ds.close()


def store_dolp(df):
    wlens = [f"{i}" for i in range(350, 2501)]
    # print(df.columns)
    """
    -76.50106211984625,  -49.85643929803059, -35.672538961350085,
    -21.45596162213788,  -8.428740090236671,  -8.382861300750376,
    7.657853637133816,   8.025454918140133,   19.76725036737047,
    22.232507280735188,   32.48366720563654,   36.68248175569178,
    44.97968174924686,    50.4848074684456,   57.10322592899022,
    64.223405195593,   77.21484487619131]
    """
    mpa_group = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 10, 11]
    mpa_group_df = pd.DataFrame(mpa_group, columns=["mpa_group"])
    df_T = pd.merge(
        df.T.reset_index(), mpa_group_df, left_index=True, right_index=True
    ).rename(columns={"index": "mpa"})
    df_T["mpa"] = df_T["mpa"].astype(float)
    df_T = (
        df_T.groupby("mpa_group")
        .mean()
        .reset_index()
        .drop("mpa_group", axis=1)
        .set_index("mpa")
    )
    df = df_T.T
    print(df.head(3))
    ds = nc.Dataset(NC_PATH, "r+")
    phases = ds["phase_angle"][:].data
    dolp_mpas = np.array(list(map(float, df.columns)))
    print(dolp_mpas)
    print(phases)
    mpas_ids = np.array([np.argsort(np.abs(p - dolp_mpas))[:2] for p in phases])
    # print(mpas_ids)
    dolps = []
    for mpas, phase in zip(mpas_ids, phases):
        p0 = dolp_mpas[mpas[0]]
        p1 = dolp_mpas[mpas[1]]
        if phase > p0 and phase > p1:
            print(mpas[np.argmax([p0, p1])])
            val = df[df.columns[mpas[np.argmax([p0, p1])]]]
        elif phase < p0 and phase < p1:
            print(mpas[np.argmin([p0, p1])])
            val = df[df.columns[mpas[np.argmin([p0, p1])]]]
        else:
            dist = abs(p0 - p1)
            val = df[df.columns[mpas[0]]] * (dist - abs(phase - p0)) / dist
            val += df[df.columns[mpas[1]]] * (dist - abs(phase - p1)) / dist
        dolps.append(val)
    print((dolps[0] == dolps[60]).sum())
    #    np.array([df[df.columns[mpa]] for mpa in mpas_ids])
    dolps = np.array(dolps)
    print(dolps.shape)
    ds["polarisation"][:] = dolps.T
    ds.close()


def main():
    df = pd.read_csv(CSV_PROCESSED_PATH, index_col=0)
    store_dolp(df)


if __name__ == "__main__":
    main()
