module VelocityDistributionFunctions

using Tullio: @tullio
using Bumper
using LinearAlgebra
using StaticArrays

include("utils.jl")
include("spectra.jl")
include("pad.jl")
include("flux.jl")
include("distributions/Distributions.jl")

# New particle analysis modules
include("particles.jl")
include("spectrograms.jl")
include("moments.jl")

# Legacy exports (pitch angle distributions)
export pitch_angle_distribution, tpitch_angle_distribution
export directional_energy_spectra

# Distribution types and functions
export VelocityDistribution, KappaDistribution, BiMaxwellian, BiKappa, Maxwellian, Kappa
export AbstractVelocityPDF, AbstractVelocityDistribution
export MaxwellianPDF, BiMaxwellianPDF, KappaPDF, BiKappaPDF, ShiftedPDF
export pdf
export kappa_thermal_speed
export VPar, VPerp

# Particle data structures
export AbstractParticleData, ParticleData

# Coordinate utilities
export solid_angle, velocity_grid, look_directions
export velocity_from_energy
export pitch_angles, gyrophase_angles

# Spectrogram functions
export energy_spectrogram, theta_spectrogram, phi_spectrogram
export pitch_angle_spectrogram, gyrophase_spectrogram

# Moment functions
export density, bulk_velocity
export pressure_tensor, pressure_scalar
export temperature_tensor, temperature_scalar
export temperature_parallel, temperature_perpendicular
export heat_flux, heat_flux_parallel
export entropy_density

end
