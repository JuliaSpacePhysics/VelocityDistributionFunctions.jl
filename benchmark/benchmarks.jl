using BenchmarkTools
using VelocityDistributionFunctions
using StaticArrays: SA

const SUITE = BenchmarkGroup()

const v3 = SA[0.5, 0.3, 0.8]

for (name, d) in [
    ("Maxwellian",  Maxwellian(1.0)),
    ("BiMaxwellian", BiMaxwellian(1.0, 2.0)),
    ("Kappa",       Kappa(1.0, 4.0)),
    ("BiKappa",     BiKappa(1.0, 2.0, 4.0)),
]
    g = SUITE[name] = BenchmarkGroup()
    g["pdf"]      = @benchmarkable pdf($d, $v3)
    g["rand"]     = @benchmarkable rand($d)
    g["rand_1k"]  = @benchmarkable rand($d, 1000)
end

# tmoments benchmark with synthetic MMS-like data
let nt = 100, nphi = 32, ntheta = 16, nenergy = 32
    data = rand(Float32, nt, nphi, ntheta, nenergy)
    theta = collect(range(-90.0, 90.0, length = ntheta))
    phi = collect(range(0.0, 360.0, length = nphi + 1)[1:nphi])
    energy = 10 .^ collect(range(log10(10.0), log10(30000.0), length = nenergy))
    scpot = zeros(nt)
    magf = randn(nt, 3)

    g = SUITE["tmoments"] = BenchmarkGroup()
    g["no_magf"]   = @benchmarkable tmoments($data, $theta, $phi, $energy, $scpot; edim = 4, tdim = 1, units = :df_cm)
    g["with_magf"] = @benchmarkable tmoments($data, $theta, $phi, $energy, $scpot, $magf; edim = 4, tdim = 1, units = :df_cm)
end
