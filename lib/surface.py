"""Surface Construction"""

import numpy as np
import polars as pl
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import least_squares


def kernel_smoothing(
    X: np.ndarray,
    Y: np.ndarray,
    df: pl.DataFrame,
    h_x: float = 0.01,
    h_y: float = 0.1,
) -> np.ndarray:
    """
    Construct kernel smoothed surface.

    Args:
        X (np.ndarray): Moneyness coordinate matrix.
        Y (np.ndarray): Horizon coordinate matrix.
        df (pl.DataFrame): Option data.
        h_x (float): Moneyness bandwidth.
        h_y (float): Horizon bandwidth.
    """

    def kernel(dx, dy):
        exponent = (dx**2) / (2 * h_x) + (dy**2) / (2 * h_y)
        return np.exp(-exponent) / np.sqrt(2 * np.pi)

    # Vectorized!
    x = X.ravel().reshape(1, -1)
    y = Y.ravel().reshape(1, -1)
    dx = df["logMoneyness"].to_numpy().reshape(-1, 1) - x
    dy = np.log(df["daysToExpiry"].to_numpy().reshape(-1, 1) / y)
    w = df["weight"].to_numpy().reshape(-1, 1) * kernel(dx, dy)
    res = np.sum(df["markIV"].to_numpy()[:, np.newaxis] * w, axis=0)
    res /= np.sum(w, axis=0)
    Z = res.reshape(Y.shape)
    return Z


def svi(X: np.ndarray, Y: np.ndarray, df: pl.DataFrame):
    """
    Construct Stochastic Volatility Inspired (SVI) surface.

    Args:
        X (np.ndarray): Moneyness coordinate matrix.
        Y (np.ndarray): Horizon coordinate matrix.
        df (pl.DataFrame): Option data.
    """

    def raw_svi(k, a, b, s, rho, m):
        return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))

    def fit(data):
        """For a given time slice, return fitted IV curve."""
        log_moneyness = data[0].to_numpy()
        yrs_to_expiry = data[1][0] / 365
        mark_iv = data[2].to_numpy()
        weights = data[3].to_numpy()

        def residual(params):
            total_implied_var = raw_svi(log_moneyness, *params)
            iv = np.sqrt(total_implied_var / yrs_to_expiry)
            return (mark_iv - iv) * np.sqrt(weights)

        result = least_squares(
            fun=residual,
            x0=np.zeros(5),
            max_nfev=10_000,
            verbose=0,
            bounds=([-np.inf, 0, -1, -np.inf, 0], [np.inf, np.inf, 1, np.inf, np.inf]),
        )

        return np.sqrt(raw_svi(X[0], *result.x) / yrs_to_expiry)

    # Fit for each time slice.
    res = (
        df.group_by("daysToExpiry")
        .agg(pl.map_groups(["logMoneyness", "daysToExpiry", "markIV", "weight"], fit))
        .sort("daysToExpiry")
    )
    y_raw = res["daysToExpiry"].to_numpy()
    z_raw = res["logMoneyness"].explode().to_numpy()
    Z_raw = z_raw.reshape((len(y_raw), len(X[0]))).T

    # Interpolate over horizon with bivariate spline approximation.
    interpolate = RectBivariateSpline(X[0], y_raw, Z_raw, ky=2)
    Z = interpolate(X[0], Y[:, 0]).T
    return Z
