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


# --- Compute moments in Julia for each timestep ---
_dict2nt(dist) = (
    data = dist["data"], energy = unique(dist["energy"]),
    theta = dist["theta"], dtheta = dist["dtheta"],
    phi = dist["phi"], dphi = dist["dphi"],
    bins = dist["bins"], mass = dist["mass"], charge = dist["charge"],
)
