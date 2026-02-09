function download_test_data(url, filename = basename(url))
    cache_dir = joinpath(pkgdir(VelocityDistributionFunctions), "data")
    mkpath(cache_dir)
    filepath = joinpath(cache_dir, filename)
    if !isfile(filepath)
        @info "Downloading test data: $filename"
        Downloads.download(url, filepath)
    end
    return filepath
end

_angle(input) = selectdim(input, 1, 1)

# --- Compute moments in Julia for each timestep ---
_dict2nt(dist) = (
    data = dist["data"], energy = unique(dist["energy"]),
    theta = _angle(dist["theta"]), phi = _angle(dist["phi"]),
    dtheta = _angle(dist["dtheta"]), dphi = _angle(dist["dphi"]),
)
