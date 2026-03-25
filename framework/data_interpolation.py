"""
Generate an Antimony piecewise function with linear or spline interpolation.

This script takes vectors of times and values and generates an Antimony
piecewise function that performs either linear or cubic spline interpolation 
between the points.
"""

import numpy as np
from typing import List, Union
from scipy.interpolate import CubicSpline, PchipInterpolator


def format_number(num: float, precision: int = 15) -> str:
    """
    Format a number for Antimony output, removing unnecessary trailing zeros.
    
    Parameters
    ----------
    num : float
        Number to format
    precision : int
        Maximum number of decimal places (default: 15)
    
    Returns
    -------
    str
        Formatted number string
    """
    # Use g format to remove trailing zeros, but limit precision
    formatted = f"{num:.{precision}g}"
    # Remove trailing decimal point if present
    if formatted.endswith('.'):
        formatted = formatted[:-1]
    return formatted


def _generate_linear_piecewise(
    times: Union[List[float], np.ndarray],
    data: Union[List[float], np.ndarray],
    data_name: str = "data",
    time_var: str = "time",
    default_after: Union[float, None] = None
) -> str:
    """
    Helper function to generate an Antimony piecewise function with linear interpolation.
    """
    times = np.array(times)
    data = np.array(data)
    
    # Validate inputs
    if len(times) != len(data):
        raise ValueError("times and data must have the same length")
    if len(times) < 2:
        raise ValueError("At least 2 points are required for interpolation")
    
    # Sort by time to ensure proper ordering
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    data = data[sort_idx]
    
    # Set default to the last data value for times after the last point
    if default_after is None:
        default_after = data[-1]
    
    # Build piecewise conditions and expressions
    pieces = []
    
    # Generate linear interpolation for each interval
    for i in range(len(times) - 1):
        t_i = times[i]
        t_next = times[i + 1]
        v_i = data[i]
        v_next = data[i + 1]
        
        # Skip if times are equal (would cause division by zero)
        if abs(t_next - t_i) < 1e-10:
            continue
        
        # Linear interpolation formula: v = v_i + (v_next - v_i) * (t - t_i) / (t_next - t_i)
        # For Antimony, we'll use: (v_i + (v_next - v_i) * (time_var - t_i) / (t_next - t_i))
        
        # Build the interpolation expression
        if abs(v_next - v_i) < 1e-10:
            # Constant value (no interpolation needed)
            interp_expr = format_number(v_i)
        else:
            # Linear interpolation: v_i + slope * (time_var - t_i) / dt
            slope = v_next - v_i
            dt = t_next - t_i
            v_i_str = format_number(v_i)
            slope_str = format_number(slope)
            t_i_str = format_number(t_i)
            dt_str = format_number(dt)
            interp_expr = f"({v_i_str} + ({slope_str} * ({time_var} - {t_i_str})) / {dt_str})"
        
        pieces.append(interp_expr)
        pieces.append(f"{time_var} < {format_number(t_next)}")
    
    # Handle times after last point
    pieces.append(format_number(default_after))
    pieces.append(f"{time_var} > {format_number(times[-1])}")
    
    # Join pieces into Antimony piecewise function
    piecewise_str = ", ".join(pieces)
    result = f"{data_name} := piecewise({piecewise_str})"
    
    return result


def _generate_spline_piecewise(
    times: Union[List[float], np.ndarray],
    data: Union[List[float], np.ndarray],
    data_name: str = "data",
    time_var: str = "time",
    antimony_continuation: bool = True,
    monotone: bool = False
) -> str:
    """
    Helper function to generate an Antimony piecewise function with spline interpolation.
    MIT License

    Copyright (c) 2026 UW Sauro Lab

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    x = np.asarray(times, dtype=float)
    y = np.asarray(data, dtype=float)

    if x.size != y.size:
        raise ValueError("times and data must have the same length.")
    if x.size < 3:
        raise ValueError(
            "At least 3 data points are required to generate a cubic spline."
        )
    if np.any(np.diff(x) <= 0):
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        if np.any(np.diff(x) <= 0):
            raise ValueError("time values must be strictly increasing.")

    if monotone:
        cs = PchipInterpolator(x, y)
    else:
        cs = CubicSpline(x, y, bc_type="natural")

    n_segments = len(x) - 1
    sep = ",\\ \n" if antimony_continuation else ",\n"
    segments = []

    for k in range(n_segments):
        xk  = x[k]
        xk1 = x[k + 1]
        c3  = cs.c[0, k]   # cubic
        c2  = cs.c[1, k]   # quadratic
        c1  = cs.c[2, k]   # linear
        c0  = cs.c[3, k]   # constant

        def fmt(v: float) -> str:
            return f"{v:+.6f}" if v < 0 else f"+{v:.6f}"

        delay = f"({time_var}-{xk:.6f})"
        expr = (
            f"(({c3:.6f}*{delay}"
            f"{fmt(c2)})*{delay}"
            f"{fmt(c1)})*{delay}"
            f"{fmt(c0)}"
        )
        condition = f"({time_var} >={xk:.6f}) && ({time_var} <= {xk1:.6f})"
        segments.append(f"{expr}, {condition}")

    body = sep.join(segments)
    return f"{data_name} := piecewise ({body})"


def generate_antimony_piecewise(
    times: Union[List[float], np.ndarray],
    data: Union[List[float], np.ndarray],
    data_name: str = "data",
    time_var: str = "time",
    interpolation_type: str = "linear",
    **kwargs
) -> str:
    """
    Generate an Antimony piecewise function with either linear or spline interpolation.
    
    Parameters
    ----------
    times : array-like
        Vector of time points (must be sorted in ascending order)
    data : array-like
        Vector of data values corresponding to each time point
    data_name : str
        Name of the data variable in Antimony output (default: "data")
    time_var : str
        Name of the time variable in Antimony output (default: "time")
    interpolation_type : str
        "linear" or "spline" (default: "linear")
    **kwargs
        Additional arguments passed to the specific interpolation function:
        - For linear: default_after (float or None)
        - For spline: antimony_continuation (bool), monotone (bool)
        
    Returns
    -------
    str
        Antimony piecewise function definition
    """
    if interpolation_type.lower() == "linear":
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ["default_after"]}
        return _generate_linear_piecewise(times, data, data_name, time_var, **valid_kwargs)
    elif interpolation_type.lower() == "spline":
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ["antimony_continuation", "monotone"]}
        return _generate_spline_piecewise(times, data, data_name, time_var, **valid_kwargs)
    else:
        raise ValueError(f"Unknown interpolation_type: {interpolation_type}. Must be 'linear' or 'spline'.")