# ellipses.py

import numpy as np
import matplotlib.pyplot as plt
import ellipse

def fit_ellipse_v1(cnt_points_arr):
    """
    Fits an ellipse to the given contour points using least squares method.

    Parameters:
    cnt_points_arr (numpy.ndarray): Array of contour points.

    Returns:
    tuple: A tuple containing:
        - reg (ellipse.LsqEllipse): The fitted ellipse object.
        - center (tuple): The center coordinates of the ellipse.
        - width (float): The semimajor axis (horizontal dimension) of the ellipse.
        - height (float): The semiminor axis (vertical dimension) of the ellipse.
        - angle (float): The tilt angle of the ellipse in degrees.
    """
    reg = ellipse.LsqEllipse().fit(cnt_points_arr)
    center, width, height, phi = reg.as_parameters()
    angle = np.rad2deg(phi)

    return reg, center, width, height, angle

def fit_ellipse_v2(cnt_points):
    """
    Fits an ellipse to the given contour points using least squares method.

    Parameters:
    cnt_points (list or numpy.ndarray): List or array of contour points.

    Returns:
    tuple: A tuple containing:
        - alphas (numpy.ndarray): The coefficients of the fitted ellipse.
        - cnt_points_arr (numpy.ndarray): Reshaped array of contour points.
        - X (numpy.ndarray): X coordinates of contour points.
        - Y (numpy.ndarray): Y coordinates of contour points.
        - Z (numpy.ndarray): Calculated Z values for the ellipse equation.
    """
    cnt_points_arr = np.array(cnt_points).reshape(-1, 2, 1)
    X = cnt_points_arr[:, 0]
    Y = cnt_points_arr[:, 1]
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    alphas = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    Z = (
        alphas[0] * X**2
        + alphas[1] * X * Y
        + alphas[2] * Y**2
        + alphas[3] * X
        + alphas[4] * Y
    )

    return alphas, cnt_points_arr, X, Y, Z

def plot_ellipse(cnt_points_arr, alphas=np.array([]), img_show=False):
    """
    Plots an ellipse based on the given contour points and coefficients.

    Parameters:
    cnt_points_arr (numpy.ndarray): Array of contour points.
    alphas (numpy.ndarray): Coefficients of the fitted ellipse equation. Default is an empty array.
    img_show (bool): Flag to display the plot. Default is False.

    Returns:
    numpy.ndarray: Array of ellipse coordinates.
    """
    ellipse_coord = None

    X = cnt_points_arr[:, 0]
    Y = cnt_points_arr[:, 1]

    X_min = np.min(X)
    X_max = np.max(X)
    Y_min = np.min(Y)
    Y_max = np.max(Y)

    x_coord = np.linspace(X_min - 10, X_max + 10, 300)
    y_coord = np.linspace(Y_min - 10, Y_max + 10, 300)

    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)

    if len(alphas) > 0:
        Z_coord = (
            alphas[0] * X_coord**2
            + alphas[1] * X_coord * Y_coord
            + alphas[2] * Y_coord**2
            + alphas[3] * X_coord
            + alphas[4] * Y_coord
        )
        tt = plt.contour(
            X_coord, Y_coord, Z_coord, levels=[1], colors="r", linewidths=2
        )
        ellipse_coord = np.array(
            [
                [int(round(item[0])), int(round(item[1]))]
                for item in tt.allsegs[0][0]
            ]
        )

    if img_show:
        plt.scatter(X, Y, label="Data Points")
        if len(alphas) > 0:
            plt.contour(
                X_coord, Y_coord, Z_coord, levels=[1], colors="r", linewidths=2
            )
        plt.show()

    return ellipse_coord
