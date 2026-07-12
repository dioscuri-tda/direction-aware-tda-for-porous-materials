using LinearAlgebra
import NPZ

struct WeightedConeFiltratorZ
    radius::Float64
    height::Float64
    max_X::Int
    max_Y::Int
    max_Z::Int
    r_multiplier::Float64
    h_multiplier::Float64

    function WeightedConeFiltratorZ(radius, height, r_multiplier, h_multiplier)
        max_X = ceil(Int, radius)
        max_Y = ceil(Int, radius)
        max_Z = abs(height)
        new(radius, height, max_X, max_Y, max_Z, r_multiplier, h_multiplier)
    end
end

function get_weighted_impact(wc::WeightedConeFiltratorZ, x, y, z, x_tip, y_tip, z_tip)
    z_diff_abs = abs(z_tip - z)
    if z_diff_abs > wc.height
        return 0.0
    end
    orth_distance = sqrt((x_tip - x)^2 + (y_tip - y)^2)
    if orth_distance <= wc.radius * z_diff_abs / wc.height
        return (wc.r_multiplier^orth_distance) * (wc.h_multiplier^z_diff_abs)
    else
        return 0.0
    end
end

function (wc::WeightedConeFiltratorZ)(x, y, z, input_grid)
    if !input_grid[x, y, z]
        return 0.0
    end
    weights_full_sum = 0.0
    weights_empty_sum = 0.0
    GRID_SHAPE = size(input_grid)
    START_J = max(1, y - wc.max_Y)
    STOP_J = min(y + wc.max_Y, GRID_SHAPE[2])
    START_K = max(1, z - wc.max_Z)
    STOP_K = min(z + wc.max_Z, GRID_SHAPE[3])
    for i in max(1, x - wc.max_X):min(x + wc.max_X, GRID_SHAPE[1])
        for j in START_J:STOP_J
            for k in START_K:STOP_K
                weight = get_weighted_impact(wc, i, j, k, x, y, z)
                if input_grid[i, j, k]
                    weights_full_sum += weight
                else
                    weights_empty_sum += weight
                end
            end
        end
    end
    return weights_full_sum / (weights_full_sum + weights_empty_sum)
end

function get_weighted_cone_filtration(data, h_multiplier=0.75, r_multiplier=0.25, radius=5.0, height=10.0)
    GRID_SIZE = size(data)
    filtration = Array{Float64}(undef, GRID_SIZE)
    wc = WeightedConeFiltratorZ(radius, height, r_multiplier, h_multiplier)
    for x in 1:GRID_SIZE[1]
        for y in 1:GRID_SIZE[2]
            for z in 1:GRID_SIZE[3]
                if data[x, y, z]
                    filtration[x, y, z] = wc(x, y, z, data)
                else
                    filtration[x, y, z] = 0.0
                end
            end
        end
    end
    return filtration
end

if abspath(PROGRAM_FILE) == @__FILE__
    data = NPZ.npzread(ARGS[1])
    filtration = get_weighted_cone_filtration(data)
    NPZ.npzwrite(ARGS[2], filtration)
end
