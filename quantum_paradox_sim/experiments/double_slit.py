"""
Double-slit interference experiment with Fraunhofer diffraction model.

This module implements a physically grounded double-slit interference model
using Fraunhofer diffraction theory with sinc^2 envelope and two-slit interference.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional


@dataclass
class DoubleSlitParams:
    """Parameters for the double-slit experiment."""
    wavelength: float  # λ in meters
    slit_separation: float  # d in meters (distance between slit centers)
    slit_width: float  # a in meters (width of each slit)
    screen_distance: float  # L in meters
    x_range: float  # Half-width of screen region in meters
    n_points: int  # Number of grid points
    
    def __post_init__(self):
        """Validate parameters."""
        if self.wavelength <= 0:
            raise ValueError("Wavelength must be positive")
        if self.slit_separation <= 0:
            raise ValueError("Slit separation must be positive")
        if self.slit_width <= 0:
            raise ValueError("Slit width must be positive")
        if self.screen_distance <= 0:
            raise ValueError("Screen distance must be positive")
        if self.x_range <= 0:
            raise ValueError("x_range must be positive")
        if self.n_points < 10:
            raise ValueError("n_points must be at least 10")


def compute_single_slit_envelope(x: np.ndarray, params: DoubleSlitParams) -> np.ndarray:
    """
    Compute the single-slit diffraction envelope using Fraunhofer theory.
    
    The envelope is sinc^2(β) where:
    β = (π * a * sin(θ)) / λ
    and sin(θ) ≈ x / L (small-angle approximation)
    
    Using numpy.sinc: sinc(z) = sin(πz) / (πz)
    We want sinc(β) = sin(β) / β
    So: sinc(β) = np.sinc(β / π)
    
    Parameters:
    -----------
    x : np.ndarray
        Screen positions (meters)
    params : DoubleSlitParams
        Experiment parameters
    
    Returns:
    --------
    envelope : np.ndarray
        Single-slit envelope (sinc^2, normalized to peak at 1)
    """
    # Small-angle approximation: sin(θ) ≈ x / L
    sin_theta = x / params.screen_distance
    
    # β = (π * a * sin(θ)) / λ
    beta = (np.pi * params.slit_width * sin_theta) / params.wavelength
    
    # Avoid division by zero at center
    beta = np.where(np.abs(beta) < 1e-10, 1e-10, beta)
    
    # sinc(β) = sin(β) / β = np.sinc(β / π)
    sinc_beta = np.sinc(beta / np.pi)
    
    # Envelope is sinc^2
    envelope = sinc_beta ** 2
    
    return envelope


def compute_distribution(
    params: DoubleSlitParams,
    gamma: float,
    mode: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the intensity distribution for the double-slit experiment.
    
    Uses Fraunhofer diffraction with two-slit interference:
    - Single-slit envelope: sinc^2(β)
    - Two-slit interference phase: Δφ = (2π / λ) * d * sin(θ)
    
    With decoherence parameter γ:
    P_raw = |ψ1|² + |ψ2|² + 2 * γ * Re(ψ1* · ψ2)
    
    Parameters:
    -----------
    params : DoubleSlitParams
        Experiment parameters
    gamma : float
        Coherence parameter (0 to 1)
        γ=1: fully coherent (full interference)
        γ=0: fully decohered (no interference)
    mode : str
        'both', 'A', or 'B' - which slits are open
    
    Returns:
    --------
    x_grid : np.ndarray
        Screen positions (meters)
    P_raw : np.ndarray
        Intensity in arbitrary units (non-negative, not normalized)
    P_pdf : np.ndarray
        Normalized probability density (for sampling)
    """
    # Create x grid
    x_grid = np.linspace(-params.x_range, params.x_range, params.n_points)
    
    # Small-angle approximation: sin(θ) ≈ x / L
    sin_theta = x_grid / params.screen_distance
    
    # Compute single-slit envelope
    envelope = compute_single_slit_envelope(x_grid, params)
    
    # Wave number
    k = 2 * np.pi / params.wavelength
    
    # Phase difference for two-slit interference
    # Δφ = (2π / λ) * d * sin(θ)
    delta_phi = k * params.slit_separation * sin_theta
    
    # Initialize complex amplitudes
    psi1 = np.zeros_like(x_grid, dtype=complex)
    psi2 = np.zeros_like(x_grid, dtype=complex)
    
    # Build amplitudes based on mode
    if mode in ['both', 'A']:
        # Slit A: phase = +Δφ/2
        # sqrt(envelope) gives amplitude, phase gives interference
        psi1 = np.sqrt(envelope) * np.exp(1j * delta_phi / 2)
    
    if mode in ['both', 'B']:
        # Slit B: phase = -Δφ/2
        psi2 = np.sqrt(envelope) * np.exp(-1j * delta_phi / 2)
    
    # Compute intensity with decoherence
    # P_raw = |ψ1|² + |ψ2|² + 2 * γ * Re(ψ1* · ψ2)
    P_raw = np.abs(psi1)**2 + np.abs(psi2)**2 + 2 * gamma * np.real(np.conj(psi1) * psi2)
    
    # Clip negative values (shouldn't happen, but numerical errors)
    P_raw = np.maximum(P_raw, 0.0)
    
    # Normalize to PDF using trapezoidal integration
    dx = x_grid[1] - x_grid[0] if len(x_grid) > 1 else 1.0
    integral = np.trapz(P_raw, x_grid)
    
    if integral > 0:
        P_pdf = P_raw / integral
    else:
        # Fallback: uniform distribution
        P_pdf = np.ones_like(P_raw) / (x_grid[-1] - x_grid[0])
    
    return x_grid, P_raw, P_pdf


def compute_visibility(
    x_grid: np.ndarray,
    P_raw: np.ndarray,
    mode: str
) -> float:
    """
    Compute visibility metric V = (Imax - Imin) / (Imax + Imin).
    
    Uses the central portion of the distribution (10% to 90% of x-range)
    to avoid edge effects. Applies smoothing to find robust max/min.
    
    Parameters:
    -----------
    x_grid : np.ndarray
        Screen positions (meters)
    P_raw : np.ndarray
        Intensity distribution (arbitrary units)
    mode : str
        'both', 'A', or 'B' - in single-slit modes, visibility is not meaningful
    
    Returns:
    --------
    V : float
        Visibility (0 to 1, where 1 is maximum contrast)
    """
    if mode != 'both':
        # In single-slit mode, there's no interference, so visibility is not meaningful
        # Return 0 or compute based on envelope variation
        return 0.0
    
    if len(P_raw) < 10:
        return 0.0
    
    # Use central 10% to 90% of range to avoid edge effects
    x_min = x_grid[0]
    x_max = x_grid[-1]
    x_range = x_max - x_min
    x_low = x_min + 0.1 * x_range
    x_high = x_min + 0.9 * x_range
    
    # Find indices in central region
    mask = (x_grid >= x_low) & (x_grid <= x_high)
    
    if np.sum(mask) < 5:
        # If central region too small, use all data
        mask = np.ones_like(x_grid, dtype=bool)
    
    P_central = P_raw[mask]
    
    if len(P_central) == 0:
        return 0.0
    
    # Apply smoothing (moving average) to find robust extrema
    # Use window size of ~5% of data points
    window_size = max(3, len(P_central) // 20)
    if window_size % 2 == 0:
        window_size += 1
    
    # Pad for convolution
    P_padded = np.pad(P_central, window_size // 2, mode='edge')
    kernel = np.ones(window_size) / window_size
    P_smooth = np.convolve(P_padded, kernel, mode='valid')
    
    Imax = np.max(P_smooth)
    Imin = np.min(P_smooth)
    
    if Imax + Imin == 0:
        return 0.0
    
    V = (Imax - Imin) / (Imax + Imin)
    
    # Ensure visibility is in [0, 1]
    V = np.clip(V, 0.0, 1.0)
    
    return V


def compute_distinguishability(gamma: float) -> float:
    """
    Compute distinguishability D = sqrt(max(0, 1 - γ²)).
    
    In the complementarity relation: V² + D² ≤ 1
    
    Parameters:
    -----------
    gamma : float
        Coherence parameter (0 to 1)
    
    Returns:
    --------
    D : float
        Distinguishability (0 to 1)
    """
    D = np.sqrt(np.maximum(0.0, 1.0 - gamma**2))
    return D


def compute_metrics(
    x_grid: np.ndarray,
    P_raw: np.ndarray,
    gamma: float,
    mode: str
) -> Dict[str, float]:
    """
    Compute all metrics for the experiment.
    
    Parameters:
    -----------
    x_grid : np.ndarray
        Screen positions (meters)
    P_raw : np.ndarray
        Intensity distribution (arbitrary units)
    gamma : float
        Coherence parameter (0 to 1)
    mode : str
        'both', 'A', or 'B'
    
    Returns:
    --------
    metrics : dict
        Dictionary containing:
        - visibility: V
        - distinguishability: D
        - complementarity: V² + D²
        - I_max: Maximum intensity in central region
        - I_min: Minimum intensity in central region
    """
    V = compute_visibility(x_grid, P_raw, mode)
    D = compute_distinguishability(gamma)
    complementarity = V**2 + D**2
    
    # Find max and min in central region (10% to 90%)
    x_min = x_grid[0]
    x_max = x_grid[-1]
    x_range = x_max - x_min
    x_low = x_min + 0.1 * x_range
    x_high = x_min + 0.9 * x_range
    
    mask = (x_grid >= x_low) & (x_grid <= x_high)
    if np.sum(mask) > 0:
        P_central = P_raw[mask]
        I_max = np.max(P_central)
        I_min = np.min(P_central)
    else:
        I_max = np.max(P_raw)
        I_min = np.min(P_raw)
    
    return {
        'visibility': V,
        'distinguishability': D,
        'complementarity': complementarity,
        'I_max': I_max,
        'I_min': I_min,
    }


def sample_hits(
    x_grid: np.ndarray,
    P_pdf: np.ndarray,
    n: int,
    rng: np.random.Generator,
    sigma_det: float = 0.0,
    dark_fraction: float = 0.0
) -> np.ndarray:
    """
    Sample n hit positions from the probability distribution.
    
    Includes detector effects:
    - Resolution blur: Gaussian blur with standard deviation sigma_det
    - Dark counts: Fraction of hits uniformly distributed across x-range
    
    Parameters:
    -----------
    x_grid : np.ndarray
        Grid of x positions (meters)
    P_pdf : np.ndarray
        Probability density function (normalized)
    n : int
        Number of samples to generate
    rng : np.random.Generator
        Random number generator
    sigma_det : float
        Detector resolution blur (meters). Default 0 (no blur)
    dark_fraction : float
        Fraction of hits that are dark counts (uniform background).
        Default 0 (no dark counts)
    
    Returns:
    --------
    hits : np.ndarray
        Array of sampled hit positions (meters)
    """
    if len(x_grid) < 2:
        return np.array([])
    
    # Compute number of signal hits vs dark counts
    n_signal = int(n * (1 - dark_fraction))
    n_dark = n - n_signal
    
    # Sample signal hits from PDF using inverse CDF
    signal_hits = np.array([])
    if n_signal > 0:
        # Compute CDF using trapezoidal integration
        dx = x_grid[1] - x_grid[0]
        cdf = np.cumsum(P_pdf) * dx
        
        # Normalize CDF to [0, 1]
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        else:
            # Uniform distribution fallback
            signal_hits = rng.uniform(x_grid[0], x_grid[-1], size=n_signal)
        
        if len(signal_hits) == 0:
            # Ensure CDF is monotonic and ends at 1
            cdf = np.clip(cdf, 0.0, 1.0)
            cdf[-1] = 1.0
            
            # Sample uniform random numbers
            u = rng.uniform(0, 1, size=n_signal)
            
            # Inverse CDF: find x values corresponding to u values
            signal_hits = np.interp(u, cdf, x_grid)
    
    # Sample dark counts (uniform across x-range)
    dark_hits = np.array([])
    if n_dark > 0:
        dark_hits = rng.uniform(x_grid[0], x_grid[-1], size=n_dark)
    
    # Combine hits
    all_hits = np.concatenate([signal_hits, dark_hits]) if len(signal_hits) > 0 and len(dark_hits) > 0 else (
        signal_hits if len(signal_hits) > 0 else dark_hits
    )
    
    # Apply detector resolution blur (Gaussian)
    if sigma_det > 0 and len(all_hits) > 0:
        blur = rng.normal(0, sigma_det, size=len(all_hits))
        all_hits = all_hits + blur
    
    return all_hits

