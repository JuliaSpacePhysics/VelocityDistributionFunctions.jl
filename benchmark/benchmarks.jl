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
