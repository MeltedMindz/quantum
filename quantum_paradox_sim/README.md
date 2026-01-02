# Quantum Double-Slit Paradox Simulator

An interactive Streamlit application that simulates the quantum double-slit experiment using **Fraunhofer diffraction theory**, demonstrating how interference patterns emerge and how which-path information (modeled as decoherence) erases the fringes.

## What This Simulator Demonstrates

### Fraunhofer Diffraction
The simulator uses physically grounded Fraunhofer diffraction theory:
- **Single-slit envelope**: sinc²(β) where β = (π·a·sin(θ))/λ
- **Two-slit interference**: Phase difference Δφ = (2π/λ)·d·sin(θ)
- **Small-angle approximation**: sin(θ) ≈ x/L for computational efficiency

### Quantum Interference
When both slits are open and the system is fully coherent (γ=1), the intensity distribution shows a characteristic interference pattern with bright and dark fringes modulated by the sinc² envelope. This emerges from the wave nature of quantum particles.

### Decoherence and Which-Path Information
The coherence parameter γ controls the strength of interference:
- **γ = 1.0**: Fully coherent - maximum interference, no which-path information
- **γ = 0.0**: Fully decohered - no interference, which-path information is available

As γ decreases from 1 to 0, the interference fringes smoothly disappear, and the pattern becomes the classical sum of two single-slit patterns.

### Information-Theoretic Complementarity
The simulator demonstrates the complementarity relation:
- **Visibility V**: Measures interference contrast, V = (Imax - Imin)/(Imax + Imin)
- **Distinguishability D**: Measures which-path information, D = √(1 - γ²)
- **Complementarity**: V² + D² ≤ 1

This shows that gaining which-path information (increasing D) necessarily reduces interference visibility (decreasing V).

### Detector Realism
The simulator includes realistic detector effects:
- **Resolution blur**: Gaussian blur with configurable standard deviation σ_det
- **Dark counts**: Uniform background noise representing detector dark counts

### Particle-Wave Duality
The Monte Carlo simulation shows how individual particle detections (discrete hits) accumulate over time to form the continuous interference pattern. Each hit appears random, but the statistical distribution converges to the quantum probability distribution P(x).

## Features

- **Interactive Controls**: Adjust wavelength, slit separation, slit width, screen distance, and coherence parameter
- **Three Modes**: Both slits open, only slit A, or only slit B
- **Detector Realism**: Configurable resolution blur and dark count fraction
- **Real-time Visualization**: 
  - Continuous intensity distribution I(x) vs position
  - Histogram of accumulated particle hits with theoretical overlay
  - Visibility, distinguishability, and complementarity metrics
- **Monte Carlo Simulation**: Step-by-step accumulation of particle detections with detector effects
- **Extensible Architecture**: Clean separation of physics models for easy addition of new experiments

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment**:
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

From the `quantum_paradox_sim` directory, run:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage Tips

1. **Start with defaults**: The default parameters are set to show a clear interference pattern with visible fringes
2. **Adjust coherence**: Move the γ slider to see how interference fades as which-path information becomes available. Watch V decrease and D increase.
3. **Observe complementarity**: Check that V² + D² ≤ 1 is satisfied
4. **Accumulate hits**: Click "Simulate Step" multiple times to watch the histogram converge to the theoretical distribution
5. **Try different modes**: Switch between "Both slits", "Slit A only", and "Slit B only" to see how interference requires both paths
6. **Experiment with parameters**: Adjust wavelength, slit separation, and screen distance to see how they affect the pattern
7. **Add detector effects**: Increase σ_det to see resolution blur, or add dark counts to see background noise

## Physics Model

This simulator uses **Fraunhofer diffraction theory** with physically interpretable parameters:

### Intensity Model
- **Single-slit envelope**: I_envelope(x) = sinc²(β) where β = (π·a·x)/(λ·L)
- **Two-slit interference**: Complex amplitudes with phase difference Δφ = (2π/λ)·d·x/L
- **With decoherence**: I(x) = |ψ₁|² + |ψ₂|² + 2·γ·Re(ψ₁*·ψ₂)
  - ψ₁ = √(envelope) · exp(+i·Δφ/2)
  - ψ₂ = √(envelope) · exp(-i·Δφ/2)

### Parameters
- **λ (wavelength)**: Determines fringe spacing and envelope width
- **d (slit separation)**: Controls fringe spacing (larger d → more fringes)
- **a (slit width)**: Controls envelope width (larger a → narrower envelope)
- **L (screen distance)**: Affects overall scale of pattern
- **γ (coherence)**: Controls interference strength (0 = no interference, 1 = full interference)

### Detector Effects
- **Resolution blur**: Each hit is blurred by Gaussian with σ = σ_det
- **Dark counts**: Fraction of hits are uniformly distributed background

The model is **educational** and demonstrates key quantum concepts using standard diffraction theory, but is not a full quantum mechanical wave optics solver suitable for laboratory-grade calculations.

## File Structure

```
quantum_paradox_sim/
├── app.py                    # Streamlit UI and main application
├── experiments/
│   ├── __init__.py          # Experiments registry
│   └── double_slit.py       # Double-slit experiment implementation
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Extensibility

The codebase is structured to support additional experiments:
- **Mach-Zehnder interferometer**: Can be added as `experiments/mach_zehnder.py`
- **Quantum eraser**: Can be added as `experiments/quantum_eraser.py`
- Each experiment follows a consistent interface with `compute_distribution()`, `sample_hits()`, and `compute_metrics()`

## License

This is an educational tool provided as-is for learning and demonstration purposes.
