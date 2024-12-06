
# Import necessary packages
using JSON           # For reading JSON files
using Flux           # For defining neural networks
using Statistics     # For statistical operations like mean and std
using LinearAlgebra  # For matrix operations
using Random         # For adding small noise
using JuMP           # For TSP optimization model
using GLPK           # For the optimization solver
using InferOpt       # For differentiable oracle
import InferOpt: PerturbedAdditive
using Flux.Optimise  # To ensure access to ADAM optimizer and update function
using Optimisers     # For advanced optimizer functionality
using ProgressMeter
import InferOpt: PerturbedMultiplicative
using Plots
using ChainRules
using BSON
using JLD2
using PrettyTables
import JuMP: objective_value
using JSON
using Graphs
using DataFrames
using CSV



########################################################


########################################################



# Check if the graph is fully connected
function is_fully_connected(edges, num_nodes)
   g = SimpleGraph(num_nodes)
   for edge in edges
       add_edge!(g, edge["from"] + 1, edge["to"] + 1)  # Convert to 1-based indexing
   end
   return is_connected(g)
end


# Function to inspect dataset structure
function print_dataset_structure(dataset)
   println("Dataset Type: ", typeof(dataset))
   println("Number of entries: ", length(dataset))
  
   if length(dataset) > 0
       println("\n--- Inspecting First Entry ---")
       #println("Keys in the first entry: ", keys(dataset[1]))
       #println("First entry example: ", dataset[1])
      
       # Check datatypes
       println("\n--- Datatypes ---")
       for key in keys(dataset[1])
           #println("$key => ", typeof(dataset[1][key]))
       end


       # Inspect "cost_matrix" if present
       if haskey(dataset[1], "cost_matrix")
           cost_matrix = dataset[1]["cost_matrix"]
           println("\n--- Cost Matrix ---")
           println("Cost Matrix Shape: ", size(cost_matrix))
           println("First 5 rows of Cost Matrix:")
           for row in 1:min(5, size(cost_matrix, 1))
               println(cost_matrix[row, :])
           end
       end


       # Inspect "edges" if present
       if haskey(dataset[1], "edges")
           println("\n--- Inspecting Edges ---")
           num_nodes = length(dataset[1]["nodes"])
           edges = dataset[1]["edges"]
           println("Number of nodes: ", num_nodes)
           println("Number of edges: ", length(edges))
          
           # Check if fully connected
           connected = is_fully_connected(edges, num_nodes)
           println("Is the graph fully connected? ", connected)
       end


       # Inspect "optimal_tour" if present
       if haskey(dataset[1], "optimal_tour")
           println("\n--- Optimal Tour ---")
           println("Optimal tour example: ", dataset[1]["optimal_tour"])
       end
      
       # Inspect "total_cost" if present
       if haskey(dataset[1], "total_cost")
           println("\n--- Optimal Cost ---")
           println("Optimal cost example: ", dataset[1]["total_cost"])
       end
   else
       println("Dataset is empty!")
   end
end




########################################################

#data loading function
########################################################



# Function to load and preprocess dataset
function load_data(filename::String)
   println("Loading dataset from $filename...")
   file = open(filename, "r")
   raw_data = JSON.parse(file)
   close(file)
  
   println("Dataset loaded successfully with $(length(raw_data)) entries.")
  
   # Preprocess each graph in the dataset
   dataset = map(graph -> begin
       # Convert "nodes" keys from strings to integers
       if haskey(graph, "nodes")
           graph["nodes"] = Dict(parse(Int, k) => v for (k, v) in graph["nodes"])
       end
      
       # Convert "edges" keys from strings to integers (if any edges are present)
       if haskey(graph, "edges")
           graph["edges"] = map(edge -> begin
               edge["from"] = parse(Int, string(edge["from"]))
               edge["to"] = parse(Int, string(edge["to"]))
               edge
           end, graph["edges"])
       end


       # Convert the cost matrix to a proper Julia Matrix if needed
       if haskey(graph, "cost_matrix") && typeof(graph["cost_matrix"]) != Matrix
           graph["cost_matrix"] = hcat(map(x -> collect(x), graph["cost_matrix"])...)
       end
      
       # Replace Inf with 0.0 in the cost matrix
       graph["cost_matrix"] = replace(graph["cost_matrix"], Inf => 0.0)
      
       return graph
   end, raw_data)
  
   return dataset
end

########################################################
# encoder function
########################################################

function encoder_tsp_flattened(graph_data, n_nodes)
   edges = graph_data["edges"]
   edge_features = []  # Store edge features row-wise


   # Loop through all edges to create a feature matrix
   for edge in edges
       from_node = Int(edge["from"]) + 1  # Julia's 1-based indexing
       to_node = Int(edge["to"]) + 1


       # Extract the 5 features and append them to the edge features list
       features = [
           Float32(edge["feature1"]),
           Float32(edge["feature2"]),
           Float32(edge["feature3"]),
           Float32(edge["feature4"]),
           Float32(edge["feature5"])
       ]
       push!(edge_features, features)
   end


   # Add self-referencing edges with all features set to 0
   for node in 1:n_nodes
       features = [0.0f0, 0.0f0, 0.0f0, 0.0f0, 0.0f0]  # Features for self-loops
       push!(edge_features, features)
   end


   # Convert the list of features into a matrix (number of edges × number of features)
   edge_features_matrix = hcat(edge_features...)'


   # Transpose the matrix to match model expectations (features as the first dimension)
   return edge_features_matrix'  # Shape: (5, #edges + n_nodes)
end



########################################################
#data split
########################################################

function classic_split_80_10_10(dataset)
    # Anzahl der Graphen
    num_graphs = length(dataset)

    # Indizes für die Splits berechnen
    train_end = floor(Int, 0.8 * num_graphs)
    val_end = train_end + floor(Int, 0.1 * num_graphs)

    # Datensätze aufteilen
    train_set = dataset[1:train_end]
    val_set = dataset[train_end+1:val_end]
    test_set = dataset[val_end+1:end]

    return train_set, val_set, test_set
end





########################################################
#vector to cost matrix functio 

########################################################




function vector_to_cost_matrix(predicted_cost_vector, n_nodes)
   # Reshape the vector into a square matrix
   cost_matrix = reshape(predicted_cost_vector, n_nodes, n_nodes)
  
   # Create a mask with zeros on the diagonal and ones elsewhere
   mask = Float32.(1 .- I(n_nodes))
  
   # Apply the mask to the cost matrix
   zero_diag_matrix = cost_matrix .* mask
   return zero_diag_matrix
end





########################################################

#tsp oracle

########################################################


function tsp_oracle(cost_matrix; only_cost::Bool=false)
   # Ensure cost_matrix is square
   if size(cost_matrix, 1) != size(cost_matrix, 2)
       error("Cost matrix must be square.")
   end


   n = size(cost_matrix, 1)  # Number of nodes
   model = Model(GLPK.Optimizer)


   # Decision variables: x[i, j] = 1 if edge (i, j) is in the tour
   @variable(model, x[1:n, 1:n], Bin)


   # Subtour elimination variables (MTZ formulation)
   @variable(model, u[2:n], Int)


   # Objective function: Minimize total edge cost
   @objective(model, Min, sum(cost_matrix[i, j] * x[i, j] for i in 1:n, j in 1:n))


   # Degree constraints: one edge in and one edge out per node
   for i in 1:n
       @constraint(model, sum(x[i, j] for j in 1:n if i != j) == 1)
       @constraint(model, sum(x[j, i] for j in 1:n if i != j) == 1)
   end


   # Subtour elimination constraints
   for i in 2:n, j in 2:n
       if i != j
           @constraint(model, u[i] - u[j] + n * x[i, j] <= n - 1)
       end
   end


   # Solve the optimization problem
   optimize!(model)


   # Handle infeasible solutions
   if termination_status(model) != MOI.OPTIMAL
       println("Warning: No feasible solution found for TSP instance.")
       return only_cost ? 1e6 : ([], 1e6)
   end


   total_cost = objective_value(model)


   if only_cost
       return total_cost
   end


   # Extract the tour from the solution
   tour = [(i, j) for i in 1:n, j in 1:n if value(x[i, j]) > 0.5]
   return tour, total_cost
end


########################################################
#hyperparameter tuning function
########################################################
function hyperparameter_tuning!(train_set, val_set; param_grid, epochs=10)
    best_params = nothing
    best_val_loss = Inf
    results = []

    for lr in param_grid["learning_rate"],
        hidden_units in param_grid["hidden_units"],
        regularization_factor in param_grid["regularization_factor"]

        println("\nTesting combination: learning_rate=$lr, hidden_units=$hidden_units, regularization_factor=$regularization_factor")

        # Create a new model for each configuration
        model = Chain(
            Dense(5, hidden_units, relu),
            Dense(hidden_units, hidden_units, relu),
            Dense(hidden_units, 1)
        )

        # Train the model with train_set and val_set
        trained_model = train_model!(
            model, 
            train_set, 
            val_set;  # Include the validation set
            epochs=epochs, 
            learning_rate=lr
        )

        # Evaluate on validation set
        val_loss = 0.0
        for graph in val_set
            n_nodes = length(graph["nodes"])
            encoded_features = encoder_tsp_flattened(graph, n_nodes)
            y = Float32.(graph["cost_matrix"])
            pred_cost_vector = trained_model(encoded_features)
            pred_cost_matrix = vector_to_cost_matrix(pred_cost_vector, n_nodes)
            val_loss += sum((pred_cost_matrix .- y).^2)
        end

        avg_val_loss = val_loss / length(val_set)

        println("Validation Loss: $avg_val_loss")

        # Save the best parameters
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            best_params = (
                learning_rate=lr,
                hidden_units=hidden_units,
                regularization_factor=regularization_factor
            )
        end

        # Log the results
        push!(results, (lr, hidden_units, regularization_factor, avg_val_loss))
    end

    println("\nBest Parameters: $best_params with Validation Loss: $best_val_loss")
    return best_params, best_val_loss
end



########################################################
#loss function
########################################################




# Loss function
function loss_function(model, encoded_features, true_cost_matrix)
   predicted_cost_vector = model(encoded_features)
   predicted_cost_matrix = vector_to_cost_matrix(predicted_cost_vector, size(true_cost_matrix, 1))
   return sum((predicted_cost_matrix .- true_cost_matrix) .^ 2)  # MSE loss
end


########################################################
#training function
########################################################


# Simplified Training Loop with Early Stopping
function train_model!(model, train_set, val_set; epochs=10, learning_rate=0.001, patience=3)
    opt = Flux.Adam(learning_rate)  # Optimizer
    best_val_loss = Inf             # Initialize best validation loss
    no_improve_count = 0            # Counter for no improvement in validation loss
  
    for epoch in 1:epochs
        train_loss = 0.0
        val_loss = 0.0
      
        # --- Training ---
        for graph in train_set
            n_nodes = length(graph["nodes"])
          
            # Encode features
            encoded_features = encoder_tsp_flattened(graph, n_nodes)
          
            # Get ground truth cost matrix
            y = Float32.(graph["cost_matrix"])
          
            # Compute loss and gradients
            grads = Flux.gradient(() -> begin
                pred_cost_vector = model(encoded_features)
                pred_cost_matrix = vector_to_cost_matrix(pred_cost_vector, n_nodes)
                loss = sum((pred_cost_matrix .- y).^2)  # MSE Loss
                train_loss += loss
                return loss
            end, Flux.params(model))
          
            # Update model parameters
            Flux.Optimise.update!(opt, Flux.params(model), grads)
        end

        # --- Validation ---
        for graph in val_set
            n_nodes = length(graph["nodes"])

            # Encode features
            encoded_features = encoder_tsp_flattened(graph, n_nodes)

            # Get ground truth cost matrix
            y = Float32.(graph["cost_matrix"])

            # Calculate validation loss
            pred_cost_vector = model(encoded_features)
            pred_cost_matrix = vector_to_cost_matrix(pred_cost_vector, n_nodes)
            loss = sum((pred_cost_matrix .- y).^2)  # MSE Loss
            val_loss += loss
        end

        # Average losses
        avg_train_loss = train_loss / length(train_set)
        avg_val_loss = val_loss / length(val_set)

        # Log epoch losses
        println("Epoch $epoch: Train Loss = $avg_train_loss, Validation Loss = $avg_val_loss")

        # Check for improvement in validation loss
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            no_improve_count = 0  # Reset the counter
            println("Validation Loss Improved!")
        else
            no_improve_count += 1
            println("No improvement in Validation Loss for $no_improve_count epoch(s).")
        end

        # Early stopping if validation loss doesn't improve for `patience` epochs
        if no_improve_count >= patience
            println("Early stopping triggered due to no improvement in validation loss.")
            break
        end
    end
  
    return model  # Ensure the trained model is returned
end






########################################################
#evaluation function
########################################################



function evaluate_and_save_model!(model, dataset, results::Dict{Int, Dict{String, Any}})
    total_predicted_cost = 0.0
    total_optimal_cost = 0.0
    cost_differences = Float64[]  # Store differences between predicted and optimal costs

    for (i, graph) in enumerate(dataset)
        println("Processing graph $i...")

        n_nodes = size(graph["cost_matrix"], 1)
        encoded_features = encoder_tsp_flattened(graph, n_nodes)
        predicted_cost_vector = model(encoded_features)
        predicted_cost_matrix = vector_to_cost_matrix(predicted_cost_vector, n_nodes)
        real_cost_matrix = Float32.(graph["cost_matrix"])

        # Calculate predicted tour cost
        predicted_tour, _ = tsp_oracle(predicted_cost_matrix)
        predicted_cost = sum(real_cost_matrix[edge[1], edge[2]] for edge in predicted_tour)
        total_predicted_cost += predicted_cost

        # Calculate optimal tour cost
        optimal_tour, optimal_cost = tsp_oracle(real_cost_matrix)
        total_optimal_cost += optimal_cost

        # Compute and store cost difference
        cost_difference = predicted_cost - optimal_cost
        push!(cost_differences, cost_difference)

        # Save results in the dictionary
        results[i] = Dict(
            "TrueCost" => optimal_cost,
            "PredictedCost" => predicted_cost,
            "CostDifference" => cost_difference,
            "PredictedTour" => join(map(x -> "($(x[1]), $(x[2]))", predicted_tour), " -> "),
            "OptimalTour" => join(map(x -> "($(x[1]), $(x[2]))", optimal_tour), " -> ")
        )
    end

    # Calculate metrics
    avg_predicted_cost = total_predicted_cost / length(dataset)
    avg_optimal_cost = total_optimal_cost / length(dataset)
    avg_difference = mean(cost_differences)
    median_difference = median(cost_differences)
    std_difference = std(cost_differences)

    # Display metrics
    println("\nEvaluation Results:")
    println("Average Predicted Cost: $avg_predicted_cost")
    println("Average Optimal Cost: $avg_optimal_cost")
    println("Average Difference: $avg_difference")
    println("Median Difference: $median_difference")
    println("Standard Deviation of Differences: $std_difference")

    return avg_difference, median_difference, std_difference
end

function save_results_to_csv(results::Dict{Int, Dict{String, Any}}, output_path::String)
    # Convert the dictionary to a DataFrame for easier saving
    data = DataFrame(
        Datapoint = Int[],
        TrueCost = Float64[],
        PredictedCost = Float64[],
        CostDifference = Float64[],
        PredictedTour = String[],
        OptimalTour = String[]
    )

    for (i, result) in pairs(results)
        push!(data, (
            i,
            result["TrueCost"],
            result["PredictedCost"],
            result["CostDifference"],
            result["PredictedTour"],
            result["OptimalTour"]
        ))
    end

    # Save the DataFrame as a CSV
    CSV.write(output_path, data)
    println("Results saved to $output_path")
end








########################################################
#here the model gets started and the variables get defined
########################################################



# Load the dataset
dataset = load_data("data/final/tsp_dataset_with_optimal8.json")


function preprocess_cost_matrix(graph)
    # Ersetze Infinity-Werte in der cost_matrix durch 0
    graph["cost_matrix"] = replace(graph["cost_matrix"], Inf => 0)
    return graph
end


dataset = map(preprocess_cost_matrix, dataset)


# Inspect the dataset
print_dataset_structure(dataset)


# Access the first cost matrix
cost_matrix = dataset[1]["cost_matrix"]
println("First Cost Matrix Shape: ", size(cost_matrix))
println("First Cost Matrix:")
#println(cost_matrix)


train_set, val_set, test_set = classic_split_80_10_10(dataset)

println("Trainingsdatensatz: $(length(train_set)) Einträge")
println("Validierungsdatensatz: $(length(val_set)) Einträge")
println("Testdatensatz: $(length(test_set)) Einträge")







# Define the initial parameter grid after a few iterations of testing these wehre the chosen ones regularization factor is not needed
param_grid = Dict(
    # Learning rates around 2.5 with a +50% range
    "learning_rate" => [0.63, 1.25, 1.88, 2.5, 3.13, 3.75, 4.38],

    # Hidden units around 45 with a +50% range
    "hidden_units" => [11, 23, 34, 45, 56, 68, 79],

    # Regularization factor around 135 with a +50% range
    "regularization_factor" => [34, 68, 101, 135, 169, 203, 236]
)







#1
#Best Parameters: (learning_rate = 3.0, hidden_units = 51, regularization_factor = 105) with Validation Loss: 765.3240966796875

#2
#Best Parameters: (learning_rate = 2.5, hidden_units = 60, regularization_factor = 105) with Validation Loss: 762.26416015625

#3
#Best Parameters: (learning_rate = 2.5, hidden_units = 45, regularization_factor = 135) with Validation Loss: 762.2608642578125


##################################
#find hyperparameters, create model and start training
##################################
best_params, best_val_loss = hyperparameter_tuning!(
    train_set, val_set; 
    param_grid=param_grid, 
    epochs=10
)


final_model = Chain(
    Dense(5, best_params.hidden_units, relu),
    Dense(best_params.hidden_units, best_params.hidden_units, relu),
    Dense(best_params.hidden_units, 1)
)


predict_then_optimize_model = train_model!(
    final_model, 
    train_set, 
    val_set;  # Include validation set for proper early stopping
    epochs=300, 
    learning_rate=best_params.learning_rate, 
    patience=5
)




########################################################
#evaluate model this function has sometimes to be started a few times to work reliable, it loads really long for some instances. if that happpens just restart the code until it works
########################################################




println("Model and parameters have been saved successfully!")






output_csv_path = "/Users/jakobmayer/Desktop/VSCode/ML_pipeline_evaluation.csv"
model_to_evaluate = predict_then_optimize_model  # Example model
dataset_to_evaluate = load_data("data/final/tsp_dataset_with_optimal9.json")  # Example dataset

# Initialize a shared dictionary to store results
results = Dict{Int, Dict{String, Any}}()

# Evaluate the model and save results to the dictionary
evaluate_and_save_model!(model_to_evaluate, dataset_to_evaluate, results)

save_results_to_csv(results, output_csv_path)

