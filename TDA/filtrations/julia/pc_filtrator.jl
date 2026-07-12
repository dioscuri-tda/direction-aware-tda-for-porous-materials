using LinearAlgebra
using MultivariateStats
import NPZ

struct ConeFiltratorZ
    radius::Float64
    height::Float64
    max_X::Int
    max_Y::Int
    max_Z::Int

    function ConeFiltratorZ(radius, height)
        max_X = ceil(Int, radius)
        max_Y = ceil(Int, radius)
        max_Z = abs(height)
        new(radius, height, max_X, max_Y, max_Z)
    end
end

function get_weighted_impact(wc::ConeFiltratorZ, x, y, z, x_tip, y_tip, z_tip)
    z_diff_abs = abs(z_tip - z)
    if z_diff_abs > wc.height
        return 0.0
    end
    orth_distance = sqrt((x_tip - x)^2 + (y_tip - y)^2)
    if orth_distance <= wc.radius * z_diff_abs / wc.height
        return 1.0
    else
        return 0.0
    end
end

function (wc::ConeFiltratorZ)(x, y, z, input_grid)
    if !input_grid[x, y, z]
        return 1.25
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
    return 1.0 - (weights_full_sum / (weights_full_sum + weights_empty_sum))
end


function fit_line(data)
    # First principal component (direction of the line)
    try
        return fit(PCA, data; maxoutdim=1).proj[:, 1]
    catch e
        print("Error: ", e)
        return [0.0, 0.0, 0.0]
    end
end

function get_local_direction(data, x, y, z, squared_radius, radius)
    GRID_SHAPE = size(data)
    START_Y = max(1, y - radius)
    STOP_Y = min(y + radius, GRID_SHAPE[2])
    START_Z = max(1, z - radius)
    STOP_Z = min(z + radius, GRID_SHAPE[3])
    dataset = Array{Tuple{Int64, Int64, Int64}, 1}()
    for x_iter in max(1, x - radius):min(x + radius, GRID_SHAPE[1])
        x_diff = x_iter - x
        x_diff_squared = x_diff^2
        for y_iter in START_Y:STOP_Y
            y_diff = y_iter - y
            y_diff_squared = y_diff^2
            for z_iter in START_Z:STOP_Z
                if data[x_iter, y_iter, z_iter] == 0
                    continue
                end
                z_diff = z_iter - z
                z_diff_squared = z_diff^2
                if x_diff_squared + y_diff_squared + z_diff_squared <= squared_radius
                    push!(dataset, (x_diff, y_diff, z_diff))
                end
            end
        end
    end
    mat_data = Array{Float64}(undef,3,  length(dataset))
    for i in 1:length(dataset)
        mat_data[1,i] = dataset[i][1]
        mat_data[2,i] = dataset[i][2]
        mat_data[3,i] = dataset[i][3]
    end

    normalized_direction = normalize(fit_line(mat_data))[2:3]
    normalized_direction[1] = 1.0 - abs(normalized_direction[1])
    normalized_direction[2] = 1.0 - abs(normalized_direction[2])
    return  normalized_direction
end

function get_direction_filtration(data, radius)
    GRID_SIZE = size(data)
    SQUARED_RADIUS = radius^2
    filtration = Array{Float64}(undef, (GRID_SIZE[1], GRID_SIZE[2], GRID_SIZE[3], 2))
    for x in 1:GRID_SIZE[1]
        for y in 1:GRID_SIZE[2]
            for z in 1:GRID_SIZE[3]
                if data[x, y, z]
                    filtration[x, y, z,:] = get_local_direction(data, x, y, z, SQUARED_RADIUS, radius)
                else
                    filtration[x, y, z, 1] = 1.25
                    filtration[x, y, z, 2] = 1.25
                end
            end
        end
    end
    return filtration
end

function get_cone_pca_filtration(data, radius, radius_cone=5.0, height_cone=10.0)
    GRID_SIZE = size(data)
    SQUARED_RADIUS = radius^2
    wc = ConeFiltratorZ(radius_cone, height_cone)
    filtration = Array{Float64}(undef, (GRID_SIZE[1], GRID_SIZE[2], GRID_SIZE[3], 3))
    for x in 1:GRID_SIZE[1]
        for y in 1:GRID_SIZE[2]
            for z in 1:GRID_SIZE[3]
                filtration[x, y, z, 1] = wc(x, y, z, data)
                if data[x, y, z]
                    filtration[x, y, z,2:3] = get_local_direction(data, x, y, z, SQUARED_RADIUS, radius)
                else
                    filtration[x, y, z, 2] = 1.25
                    filtration[x, y, z, 3] = 1.25
                end
            end
        end
    end
    return filtration
end

if abspath(PROGRAM_FILE) == @__FILE__
    data = NPZ.npzread(ARGS[1])
    filtration = get_direction_filtration(data, 3)
    NPZ.npzwrite(ARGS[2], filtration)
end
