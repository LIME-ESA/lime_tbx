from lime_tbx.gui import gui
from lime_tbx.datatypes import datatypes


def main():
    kp = datatypes.KernelsPath("kernels", "kernels")
    g = gui.GUI(kp, "eocfi_data")


if __name__ == "__main__":
    main()
