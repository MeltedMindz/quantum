"""
Quantum Double-Slit Paradox Simulator

A Streamlit application demonstrating quantum interference and decoherence
in the double-slit experiment using Fraunhofer diffraction theory.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from experiments import DoubleSlitParams, compute_distribution, sample_hits, compute_metrics


# Page configuration
st.set_page_config(
    page_title="Quantum Double-Slit Paradox Simulator",
    page_icon="⚛️",
    layout="wide"
)

st.title("⚛️ Quantum Double-Slit Paradox Simulator")
st.markdown("""
Explore how quantum interference emerges and how which-path information (decoherence) 
erases the interference pattern. Watch the pattern build up from discrete particle hits.

**Physics Model**: Fraunhofer diffraction with sinc² envelope and two-slit interference.
""")

# Initialize session state
if 'hits' not in st.session_state:
    st.session_state.hits = np.array([])
if 'rng' not in st.session_state:
    st.session_state.rng = np.random.default_rng(42)

# Sidebar controls
st.sidebar.header("Experiment Parameters")

# Mode selection
mode = st.sidebar.selectbox(
    "Slit Configuration",
    options=['both', 'A', 'B'],
    format_func=lambda x: {'both': 'Both Slits Open', 'A': 'Slit A Only', 'B': 'Slit B Only'}[x],
    index=0
)

# Physical parameters
st.sidebar.subheader("Physical Parameters")

wavelength_nm = st.sidebar.slider(
    "Wavelength λ (nm)",
    min_value=100.0,
    max_value=1000.0,
    value=500.0,
    step=10.0,
    help="Wavelength of the quantum particles (e.g., photons, electrons)"
)
wavelength = wavelength_nm * 1e-9  # Convert to meters

slit_separation_um = st.sidebar.slider(
    "Slit Separation d (μm)",
    min_value=0.1,
    max_value=100.0,
    value=10.0,
    step=0.1,
    help="Distance between the centers of the two slits"
)
slit_separation = slit_separation_um * 1e-6  # Convert to meters

slit_width_um = st.sidebar.slider(
    "Slit Width a (μm)",
    min_value=0.01,
    max_value=50.0,
    value=1.0,
    step=0.01,
    help="Width of each slit"
)
slit_width = slit_width_um * 1e-6  # Convert to meters

screen_distance_m = st.sidebar.slider(
    "Screen Distance L (m)",
    min_value=0.01,
    max_value=10.0,
    value=0.5,
    step=0.01,
    help="Distance from the slits to the detection screen"
)
screen_distance = screen_distance_m  # Already in meters

# Display parameters
st.sidebar.subheader("Display Parameters")

x_range_mm = st.sidebar.slider(
    "Screen Half-Width (mm)",
    min_value=0.1,
    max_value=100.0,
    value=5.0,
    step=0.1,
    help="Half-width of the screen region to display"
)
x_range = x_range_mm * 1e-3  # Convert to meters

n_points = st.sidebar.slider(
    "Number of Grid Points",
    min_value=100,
    max_value=5000,
    value=2000,
    step=100,
    help="Number of points for computing the probability distribution"
)

# Decoherence parameter
st.sidebar.subheader("Coherence / Which-Path Information")

gamma = st.sidebar.slider(
    "Coherence Parameter γ",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.01,
    help="γ=1: Full interference (no which-path info)\nγ=0: No interference (which-path info available)"
)

if mode != 'both':
    st.sidebar.info("⚠️ Note: γ has no effect in single-slit mode (no interference possible).")

st.sidebar.markdown(f"""
**Current γ = {gamma:.2f}**

- **γ = 1.0**: Fully coherent, maximum interference
- **γ = 0.0**: Fully decohered, no interference (classical sum)
""")

# Detector parameters
st.sidebar.subheader("Detector Realism")

sigma_det_mm = st.sidebar.slider(
    "Detector Resolution σ_det (mm)",
    min_value=0.0,
    max_value=2.0,
    value=0.0,
    step=0.01,
    help="Gaussian blur standard deviation representing finite detector resolution"
)
sigma_det = sigma_det_mm * 1e-3  # Convert to meters

dark_fraction = st.sidebar.slider(
    "Dark Count Fraction",
    min_value=0.0,
    max_value=0.3,
    value=0.0,
    step=0.01,
    help="Fraction of hits that are background/dark counts (uniformly distributed)"
)

# Monte Carlo parameters
st.sidebar.subheader("Monte Carlo Simulation")

hits_per_step = st.sidebar.slider(
    "Hits per Step",
    min_value=10,
    max_value=5000,
    value=500,
    step=50,
    help="Number of particle hits to add in each simulation step"
)

# Random seed
seed = st.sidebar.number_input(
    "Random Seed",
    min_value=0,
    max_value=2**31 - 1,
    value=42,
    step=1,
    help="Random seed for reproducibility"
)

# Update RNG if seed changed
if st.sidebar.button("Reset Random Seed"):
    st.session_state.rng = np.random.default_rng(int(seed))

# Simulation controls
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)

with col1:
    simulate_button = st.button("🎲 Simulate Step", use_container_width=True)

with col2:
    reset_button = st.button("🔄 Reset Hits", use_container_width=True)

# Handle reset
if reset_button:
    st.session_state.hits = np.array([])

# Create parameters object
try:
    params = DoubleSlitParams(
        wavelength=wavelength,
        slit_separation=slit_separation,
        slit_width=slit_width,
        screen_distance=screen_distance,
        x_range=x_range,
        n_points=n_points
    )
except ValueError as e:
    st.error(f"Invalid parameters: {e}")
    st.stop()

# Compute distribution (with caching for performance)
@st.cache_data
def cached_compute_distribution(wavelength, slit_separation, slit_width, screen_distance, x_range, n_points, gamma, mode):
    """Cache distribution computation for performance."""
    temp_params = DoubleSlitParams(
        wavelength=wavelength,
        slit_separation=slit_separation,
        slit_width=slit_width,
        screen_distance=screen_distance,
        x_range=x_range,
        n_points=n_points
    )
    return compute_distribution(temp_params, gamma, mode)

# Compute distribution
x_grid, P_raw, P_pdf = cached_compute_distribution(
    params.wavelength,
    params.slit_separation,
    params.slit_width,
    params.screen_distance,
    params.x_range,
    params.n_points,
    gamma,
    mode
)

# Compute metrics
metrics = compute_metrics(x_grid, P_raw, gamma, mode)

# Handle simulation step
if simulate_button:
    new_hits = sample_hits(
        x_grid, P_pdf, hits_per_step, st.session_state.rng,
        sigma_det=sigma_det, dark_fraction=dark_fraction
    )
    if len(st.session_state.hits) == 0:
        st.session_state.hits = new_hits
    else:
        st.session_state.hits = np.concatenate([st.session_state.hits, new_hits])

# Convert x to mm for display
x_mm = x_grid * 1e3

# Main display area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Intensity Distribution I(x)")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(x_mm, P_raw, 'b-', linewidth=2, label='I(x) (arb. units)')
    ax1.set_xlabel('Position on Screen (mm)', fontsize=12)
    ax1.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax1.set_title('Fraunhofer Diffraction Pattern', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    st.pyplot(fig1)
    plt.close(fig1)

with col2:
    st.subheader("Metrics")
    
    st.metric("Visibility V", f"{metrics['visibility']:.3f}")
    st.metric("Distinguishability D", f"{metrics['distinguishability']:.3f}")
    st.metric("V² + D²", f"{metrics['complementarity']:.3f}")
    st.metric("Max Intensity", f"{metrics['I_max']:.2e}")
    st.metric("Min Intensity", f"{metrics['I_min']:.2e}")
    st.metric("Total Hits", len(st.session_state.hits))
    st.metric("Coherence γ", f"{gamma:.2f}")
    
    mode_display = {'both': 'Both Slits', 'A': 'Slit A Only', 'B': 'Slit B Only'}[mode]
    st.metric("Mode", mode_display)
    
    # Show complementarity relation status
    if mode == 'both':
        if metrics['complementarity'] <= 1.0:
            st.success(f"✓ Complementarity: V² + D² = {metrics['complementarity']:.3f} ≤ 1")
        else:
            st.warning(f"⚠ Complementarity: V² + D² = {metrics['complementarity']:.3f} > 1")

# Histogram of hits
st.subheader("Detected Hits Histogram")

if len(st.session_state.hits) > 0:
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    
    # Convert hits to mm
    hits_mm = st.session_state.hits * 1e3
    
    # Create histogram
    n_bins = min(100, len(st.session_state.hits) // 10) if len(st.session_state.hits) > 0 else 50
    n_bins = max(20, n_bins)
    
    counts, bins, patches = ax2.hist(
        hits_mm, 
        bins=n_bins, 
        density=True, 
        alpha=0.6, 
        color='orange',
        edgecolor='black',
        label='Detected Hits (normalized)'
    )
    
    # Overlay probability distribution (scaled to match histogram normalization)
    # The histogram is normalized (density=True), so we need to scale P_pdf accordingly
    # P_pdf is in 1/m, we need to convert to 1/mm for display
    P_pdf_mm = P_pdf * 1e3  # Convert from 1/m to 1/mm
    
    # Scale P_pdf to match the histogram scale (approximate)
    # Find the peak of histogram and scale P_pdf to similar height
    if len(counts) > 0:
        hist_peak = np.max(counts)
        pdf_peak = np.max(P_pdf_mm)
        if pdf_peak > 0:
            scale_factor = hist_peak / pdf_peak
            P_pdf_scaled = P_pdf_mm * scale_factor
            ax2.plot(x_mm, P_pdf_scaled, 'b-', linewidth=2, alpha=0.7, label='Theoretical P(x) (scaled)')
    
    ax2.set_xlabel('Position on Screen (mm)', fontsize=12)
    ax2.set_ylabel('Probability Density (1/mm)', fontsize=12)
    ax2.set_title('Monte Carlo Simulation: Accumulated Hits', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.info("Click 'Simulate Step' to start accumulating particle hits. The histogram will show how the pattern emerges from discrete detections.")

# Footer
st.markdown("---")
st.markdown("""
**About this simulator:**

This educational tool demonstrates:
- **Fraunhofer Diffraction**: Single-slit envelope with sinc² profile
- **Quantum Interference**: When both slits are open and γ=1, waves from both slits interfere, creating fringes
- **Decoherence**: As γ decreases, which-path information becomes available, erasing interference
- **Complementarity**: Visibility V and Distinguishability D satisfy V² + D² ≤ 1
- **Detector Realism**: Finite resolution blur and dark counts simulate real detectors
- **Particle-Wave Duality**: Individual hits appear random, but accumulate into the interference pattern

**Note**: This is an educational simulation using Fraunhofer diffraction theory, not a full quantum mechanical wave optics solver.
""")
