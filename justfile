default:
    just --list

perf:
    #!/usr/bin/env -S julia --threads=auto --project=. --startup-file=no
    @time Base.@time_imports using VelocityDistributionFunctions