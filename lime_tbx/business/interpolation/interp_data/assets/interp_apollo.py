# There was a problem linting, so it was moved to another module


def interp_example(
    x_i,
    y_i,
    x_hr,
    y_hr,
    x,
    relative=True,
    method="linear",
    method_hr="linear",
):
    from comet_maths import interpolate_1d_along_example as cminterp_example

    return cminterp_example(
        x_i,
        y_i,
        x_hr,
        y_hr,
        x,
        relative,
        method,
        method_hr,
    )
