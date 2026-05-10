default:
    just --list

perf:
    #!/usr/bin/env -S julia --threads=auto --project=. --startup-file=no
    using TOML
    project_name = Symbol(TOML.parsefile(Base.active_project())["name"])
    @time Base.@time_imports @eval using $project_name
