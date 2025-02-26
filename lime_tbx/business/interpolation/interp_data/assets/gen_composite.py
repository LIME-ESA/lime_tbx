import numpy as np


def main():
    ap = np.genfromtxt("./Apollo16.txt", delimiter=",").T
    br = np.genfromtxt("./Breccia.txt", delimiter=",").T
    wlens = [i for i in range(350, 2501)]
    ai = np.interp(wlens, ap[0], ap[1])
    bi = np.interp(wlens, br[0], br[1])
    ci = ai * 0.95 + bi * 0.05
    for wlen, c in zip(wlens, ci):
        print(f"{wlen},{c}")


if __name__ == "__main__":
    main()
