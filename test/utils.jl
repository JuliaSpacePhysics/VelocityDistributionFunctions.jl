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
