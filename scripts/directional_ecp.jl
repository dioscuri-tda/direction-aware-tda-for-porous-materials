using LinearAlgebra
using MultivariateStats
import NPZ
import Ecp
using Glob
using Base.Threads

function get_ecp(input_file, output_file, grid_res)
    data = NPZ.npzread(input_file)
    contr = Ecp.compute_contributions_3d(data)
    try
        out_vec = Ecp.vectorize_ecp(contr, grid_res)
        NPZ.npzwrite(output_file, out_vec)
    catch e
        println("Error: ", e)
        println("Error in computing ECP for file: $input_file")
        out_vec = fill(0.0, grid_res .+ 1)
        NPZ.npzwrite(output_file, out_vec)
    end
end

function process_directory(directory, grid_res)
    println("Processing directory: $directory")
    files = glob("*", directory)

    @threads for i in eachindex(files)
        file = files[i]
        output_file = joinpath(OUTPUT_DIR, split(file, "/")[end])
        get_ecp(file, output_file, grid_res)
    end
end

function parse_args()
    if length(ARGS) < 2
        println("Usage: julia directional_ecp.jl <input_dir> <output_dir> [--grid-res N] [--grid-res-x Nx --grid-res-y Ny --grid-res-z Nz]")
        println("  --grid-res N          Uniform grid resolution (default: 6)")
        println("  --grid-res-x/y/z N    Per-axis resolution (overrides --grid-res)")
        exit(1)
    end

    input_dir  = ARGS[1]
    output_dir = ARGS[2]

    gx = gy = gz = 6  # defaults

    i = 3
    while i <= length(ARGS)
        if ARGS[i] == "--grid-res" && i + 1 <= length(ARGS)
            v = parse(Int, ARGS[i+1])
            gx = gy = gz = v
            i += 2
        elseif ARGS[i] == "--grid-res-x" && i + 1 <= length(ARGS)
            gx = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--grid-res-y" && i + 1 <= length(ARGS)
            gy = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--grid-res-z" && i + 1 <= length(ARGS)
            gz = parse(Int, ARGS[i+1]); i += 2
        else
            println("Unknown argument: $(ARGS[i])"); exit(1)
        end
    end

    return input_dir, output_dir, (gx, gy, gz)
end

INPUT_DIR, OUTPUT_DIR, GRID_RES = parse_args()

println("ECP grid resolution: $GRID_RES  →  output shape: $(GRID_RES .+ 1)")

mkpath(OUTPUT_DIR)

process_directory(INPUT_DIR, GRID_RES)
