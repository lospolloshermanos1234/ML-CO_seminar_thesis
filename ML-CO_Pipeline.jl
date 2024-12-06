







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
using StatsBase
using BSON
using JLD2
using DataFrames
using CSV
import JuMP: objective_value
using JSON
using Graphs

#####################################
# functions to load data and inspect the dataset
#####################################


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


#####################################
# function to encode features

#####################################
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


#####################################
#train test validation split
#####################################
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



#####################################

#function to create a model
#####################################



function create_simple_model(hidden_units)
    return Chain(
        Dense(5, hidden_units, relu),
        Dense(hidden_units, hidden_units, relu),
        Dense(hidden_units, 1)
    )
end




#####################################
# transform model output into cost matrix
#####################################

function vector_to_cost_matrix(predicted_cost_vector, n_nodes)
  # Reshape the vector into a square matrix
  cost_matrix = reshape(predicted_cost_vector, n_nodes, n_nodes)
   # Create a mask with zeros on the diagonal and ones elsewhere
  mask = Float32.(1 .- I(n_nodes))
   # Apply the mask to the cost matrix
  zero_diag_matrix = cost_matrix .* mask
  return zero_diag_matrix
end


#####################################
#function for tsp solver
#####################################

function tsp_oracle(cost_matrix; n_nodes=nothing, only_cost::Bool=false)
   if n_nodes === nothing
       n_nodes = Int(sqrt(length(cost_matrix)))  # Infer n_nodes from the cost matrix or flat vector
       cost_matrix = reshape(cost_matrix, n_nodes, n_nodes)  # Reshape if it's flat
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












#########################################
#wrap the oracle into a layer
#########################################




function create_perturbed_tsp_oracle(epsilon, nb_samples)
    return PerturbedAdditive(
        θ -> tsp_oracle(θ; only_cost=true),
        ε=epsilon,
        nb_samples=nb_samples
    )
end







function differentiable_tsp_oracle(flat_cost_vector, perturbed_tsp_oracle)
    return perturbed_tsp_oracle(flat_cost_vector)
end




###########################################
# function for fenchel young loss
##########################################



# Regularization function (L2 norm)
function create_regularization_function(regularization_factor)
    return y -> sum(y .^ 2) / regularization_factor
end

# Fenchel conjugate of L2 regularization
function create_fenchel_conjugate_function(conjugate_factor)
    return θ -> sum(θ .^ 2) / conjugate_factor
end

# Fenchel-Young loss function
function fenchel_young_loss(θ, y_bar, regularization, fenchel_conjugate)
    disc = dot(θ, y_bar)
    fy_loss = regularization(y_bar) + fenchel_conjugate(θ) - disc
    return fy_loss
end





#####################################
# hyperparameter tuning function
#####################################


function hyperparameter_tuning_extended!(train_set, val_set; 
    param_grid, 
    initial_epochs=3, 
    extended_epochs=10, 
    threshold_improvement=0.05, 
    patience=2)

    best_params = nothing
    best_val_loss = Inf
    results = []

    for lr in param_grid["learning_rate"],
        epsilon in param_grid["epsilon"],
        nb_samples in param_grid["nb_samples"],
        reg_factor in param_grid["regularization_factor"],
        conjugate_factor in param_grid["conjugate_factor"],
        hidden_units in param_grid["hidden_units"]

        println("\nTesting combination: lr=$lr, epsilon=$epsilon, nb_samples=$nb_samples, reg_factor=$reg_factor, conjugate_factor=$conjugate_factor, hidden_units=$hidden_units")

        # Initialisiere Modell und Optimizer
        model = create_simple_model(hidden_units)
        opt = Flux.Adam(lr)
        regularization = create_regularization_function(reg_factor)
        fenchel_conjugate = create_fenchel_conjugate_function(conjugate_factor)
        perturbed_tsp_oracle = create_perturbed_tsp_oracle(epsilon, nb_samples)

        # Schnelles Training
        val_losses, train_losses = [], []
        for epoch in 1:initial_epochs
            train_loss, val_loss = 0.0, 0.0

            # --- Training ---
            for graph in train_set
                n_nodes = size(graph["cost_matrix"], 1)
                encoded_features = encoder_tsp_flattened(graph, n_nodes)
                y = Float32.(graph["cost_matrix"])

                grads = Flux.gradient(() -> begin
                    pred_cost_vector = model(encoded_features)
                    loss = fenchel_young_loss(pred_cost_vector, vec(y), regularization, fenchel_conjugate)
                    train_loss += loss
                    return loss
                end, Flux.params(model))
                Flux.Optimise.update!(opt, Flux.params(model), grads)
            end

            # --- Validierung ---
            for graph in val_set
                n_nodes = size(graph["cost_matrix"], 1)
                encoded_features = encoder_tsp_flattened(graph, n_nodes)
                y = Float32.(graph["cost_matrix"])

                pred_cost_vector = model(encoded_features)
                val_loss += fenchel_young_loss(pred_cost_vector, vec(y), regularization, fenchel_conjugate)
            end

            # Ergebnisse speichern
            push!(train_losses, train_loss / length(train_set))
            push!(val_losses, val_loss / length(val_set))
        end

        # Metriken berechnen
        avg_val_loss = mean(val_losses)
        std_val_loss = std(val_losses)
        avg_train_loss = mean(train_losses)
        overfitting_indicator = avg_val_loss - avg_train_loss

        println("Initial Metrics: avg_val_loss=$avg_val_loss, std_val_loss=$std_val_loss, overfitting_indicator=$overfitting_indicator")

        # Erweiterung prüfen
        if avg_val_loss < best_val_loss * (1 - threshold_improvement)
            println("Promising combination found! Extending training epochs.")

            for epoch in 1:extended_epochs
                train_loss, val_loss = 0.0, 0.0

                # --- Training ---
                for graph in train_set
                    n_nodes = size(graph["cost_matrix"], 1)
                    encoded_features = encoder_tsp_flattened(graph, n_nodes)
                    y = Float32.(graph["cost_matrix"])

                    grads = Flux.gradient(() -> begin
                        pred_cost_vector = model(encoded_features)
                        loss = fenchel_young_loss(pred_cost_vector, vec(y), regularization, fenchel_conjugate)
                        train_loss += loss
                        return loss
                    end, Flux.params(model))
                    Flux.Optimise.update!(opt, Flux.params(model), grads)
                end

                # --- Validierung ---
                for graph in val_set
                    n_nodes = size(graph["cost_matrix"], 1)
                    encoded_features = encoder_tsp_flattened(graph, n_nodes)
                    y = Float32.(graph["cost_matrix"])

                    pred_cost_vector = model(encoded_features)
                    val_loss += fenchel_young_loss(pred_cost_vector, vec(y), regularization, fenchel_conjugate)
                end

                # Ergebnisse speichern
                push!(train_losses, train_loss / length(train_set))
                push!(val_losses, val_loss / length(val_set))
            end

            avg_val_loss = mean(val_losses)
            std_val_loss = std(val_losses)
            avg_train_loss = mean(train_losses)
            overfitting_indicator = avg_val_loss - avg_train_loss
        end

        # Frühes Abbrechen prüfen
        if avg_val_loss >= best_val_loss
            patience -= 1
            if patience <= 0
                println("Stopping evaluation of this combination due to lack of improvement.")
                continue
            end
        else
            patience = 2  # Reset
        end

        # Beste Parameter speichern
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            best_params = (
                learning_rate=lr, 
                epsilon=epsilon, 
                nb_samples=nb_samples, 
                reg_factor=reg_factor, 
                conjugate_factor=conjugate_factor, 
                hidden_units=hidden_units
            )
        end

        # Ergebnisse loggen
        push!(results, (lr, epsilon, nb_samples, reg_factor, conjugate_factor, hidden_units, avg_val_loss, std_val_loss, overfitting_indicator))
    end

    println("\nBest Hyperparameters: $best_params with Validation Loss: $best_val_loss")
    return best_params, best_val_loss
end






####################### hyperparameter







println("\nStarte Trainingsprozess...\n")












function train_model_with_early_stopping!(train_set, val_set; 
    epochs=50, 
    best_params, 
    patience=5)
    # Extract hyperparameters using dictionary-style access
    learning_rate = best_params.learning_rate
    epsilon = best_params.epsilon
    nb_samples = best_params.nb_samples
    regularization_factor = best_params.reg_factor
    conjugate_factor = best_params.conjugate_factor
    hidden_units = best_params.hidden_units


    println("Starting training with:")
    println("Learning Rate = $learning_rate, Epsilon = $epsilon, Nb Samples = $nb_samples")
    println("Regularization Factor = $regularization_factor, Conjugate Factor = $conjugate_factor")
    println("Hidden Units = $hidden_units")
    println("Patience = $patience")

    # Create the model with the optimal number of hidden units
    simple_model = create_simple_model(hidden_units)

    # Optimizer
    opt = Flux.Adam(learning_rate)  
    best_val_loss = Inf             # Initialize best validation loss
    no_improve_count = 0            # Counter for early stopping

    # Create regularization and Fenchel conjugate functions
    regularization = create_regularization_function(regularization_factor)
    fenchel_conjugate = create_fenchel_conjugate_function(conjugate_factor)

    # Create the perturbed TSP oracle
    perturbed_tsp_oracle = create_perturbed_tsp_oracle(epsilon, nb_samples)

    for epoch in 1:epochs
        train_loss = 0.0
        val_loss = 0.0

        # --- Training ---
        for graph in train_set
            n_nodes = size(graph["cost_matrix"], 1)

            # Encode features
            encoded_features = encoder_tsp_flattened(graph, n_nodes)

            # Ground truth
            y = Float32.(graph["cost_matrix"])

            # Compute loss and gradients
            grads = Flux.gradient(() -> begin
                pred_cost_vector = simple_model(encoded_features)
                loss = fenchel_young_loss(
                    pred_cost_vector, 
                    vec(y), 
                    regularization, 
                    fenchel_conjugate
                )
                train_loss += loss
                return loss
            end, Flux.params(simple_model))

            # Update parameters
            Flux.Optimise.update!(opt, Flux.params(simple_model), grads)
        end

        # --- Validation ---
        for graph in val_set
            n_nodes = size(graph["cost_matrix"], 1)
            encoded_features = encoder_tsp_flattened(graph, n_nodes)
            y = Float32.(graph["cost_matrix"])
            pred_cost_vector = simple_model(encoded_features)
            val_loss += fenchel_young_loss(
                pred_cost_vector, 
                vec(y), 
                regularization, 
                fenchel_conjugate
            )
        end

        # Compute average losses
        avg_train_loss = train_loss / length(train_set)
        avg_val_loss = val_loss / length(val_set)

        println("Epoch $epoch: Train Loss = $avg_train_loss, Validation Loss = $avg_val_loss")

        # --- Early Stopping ---
        if avg_val_loss < best_val_loss
            best_val_loss = avg_val_loss
            no_improve_count = 0  # Reset counter
            println("Validation Loss Improved!")
        else
            no_improve_count += 1
            println("No improvement in Validation Loss for $no_improve_count epoch(s).")
        end

        # Stop training if no improvement
        if no_improve_count >= patience
            println("Early stopping triggered. Best Validation Loss: $best_val_loss")
            break
        end
    end

    println("Training completed with Best Validation Loss: $best_val_loss")

    # Return the trained model and the best validation loss
    return simple_model, best_val_loss
end




#####################################
#evaluate model
#####################################

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




#####################################

#setting the variables and initializing the pipeline
#####################################

 


# Load the dataset for training
dataset = load_data("data/final/tsp_dataset_with_optimal8.json")




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


######################################################
#function to easier iteratively override parameters
#####################################################

function override_best_params!(best_params, custom_params)
    # Überprüfen, ob alle erforderlichen Schlüssel in custom_params vorhanden sind
    required_keys = [:learning_rate, :epsilon, :nb_samples, :regularization_factor, :conjugate_factor, :hidden_units]
    
    for key in required_keys
        if !haskey(custom_params, key)
            error("Fehlender Parameter: $key in custom_params")
        end
    end

    # Parameter überschreiben
    for key in keys(custom_params)
        best_params[key] = custom_params[key]
    end

    println("Best parameters wurden überschrieben mit:")
    println(best_params)
end


# notes while hyperparameters where selected


#Die besten Hyperparameter sind:


#Starting training with:
#Learning Rate = 0.01, 
#Epsilon = 40, 
#Nb Samples = 30
#Regularization Factor = 150, 
#Conjugate Factor = 250
#Hidden Units = 60
#Patience = 5



#Best Hyperparameters: (lr = 0.015, epsilon = 46, nb_samples = 35, reg_factor = 175, conjugate_factor = 275, hidden_units = 55) with Validation Loss: -1.1424524269230769e6
#Starting training with:
#Learning Rate = 0.015, 
#Epsilon = 46, Nb Samples = 35
#Regularization Factor = 175, 
#Conjugate Factor = 275
#Hidden Units = 55
#Patience = 5


#version 3



#Best Hyperparameters: (lr = 5.0, epsilon = 46, nb_samples = 33, reg_factor = 190, conjugate_factor = 300, hidden_units = 50) with Validation Loss: -1.2463860634615384e6
#Starting training with:
#Learning Rate = 5.0, Epsilon = 46, Nb Samples = 33
#Regularization Factor = 190, Conjugate Factor = 300
#Hidden Units = 50
#Patience = 5


param_grid = Dict(
    # Learning Rate ±50%
    "learning_rate" => [0.63, 1.25, 1.88, 2.5, 3.13, 3.75, 4.38],

    # Epsilon ±50%
    "epsilon" => [17, 35, 52, 69, 86, 104, 121],

    # Number of Samples ±50%
    "nb_samples" => [4, 8, 13, 17, 21, 26, 30],

    # Regularization Factor ±50%
    "regularization_factor" => [71, 143, 214, 285, 356, 428, 499],

    # Conjugate Factor ±50%
    "conjugate_factor" => [113, 225, 338, 450, 563, 675, 788],

    # Hidden Units ±50%
    "hidden_units" => [6, 13, 19, 25, 31, 38, 44]
)




#Best Hyperparameters: (lr = 5.0, epsilon = 46, nb_samples = 33, reg_factor = 190, conjugate_factor = 300, hidden_units = 50) with Validation Loss: -1.2463860634615384e6




#####################################
# this function is used insted of hyperparameter tuning to save ressources
#####################################
# Aktuelle best_params
best_params = Dict(
    :learning_rate => 2.5,
    :epsilon => 69,
    :nb_samples => 50,
    :regularization_factor => 285,
    :conjugate_factor => 450,
    :hidden_units => 75
)

# Neue Werte für die besten Parameter
new_params = Dict(
    :learning_rate => 5,
    :epsilon => 46,
    :nb_samples => 33,
    :regularization_factor => 190,
    :conjugate_factor => 300,
    :hidden_units => 50
)







# Parameter überschreiben
#override_best_params!(best_params, new_params)

# Ausgabe der aktualisierten Parameter
#println("Aktualisierte best_params: $best_params")


# Hyperparameter-Tuning



println("Starting Hyperparameter Tuning...")


#####################################
#here hyperparameter tuning can be initialized again
#####################################

best_params, best_val_loss = hyperparameter_tuning_extended!(train_set, val_set; 
    param_grid=param_grid, 
    initial_epochs=10,  # Anzahl der Epochen für schnelles Training
    extended_epochs=20,  # Anzahl der Epochen für vielversprechende Kombinationen
    threshold_improvement=0.05,  # Schwellenwert für Erweiterung
    patience=5)  # Toleranz für Stagnation bei Validierungsverlust






#####################################
#here the training starts

#####################################


# Final Training
# Training with fixed `patience`
predict_and_optimize_model, best_val_loss = train_model_with_early_stopping!(train_set, val_set; 
                                                                epochs=300, 
                                                                best_params=best_params, 
                                                                patience=5)




#####################################
#here evaluating starts
#####################################

# Paths and models
output_csv_path = "/Users/jakobmayer/Desktop/VSCode/CO_noerror_evaluation.csv" #define path
model_to_evaluate = predict_and_optimize_model  # Example model
dataset_to_evaluate = load_data("data/final/tsp_dataset_with_optimal9.json")  # Example dataset


# Initialize a shared dictionary to store results
results = Dict{Int, Dict{String, Any}}()

# Evaluate the model and save results to the dictionary
evaluate_and_save_model!(model_to_evaluate, dataset_to_evaluate, results)







save_results_to_csv(results, output_csv_path)


