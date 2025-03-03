using Flux
using DataFrames
using CSV
using StatsBase
using Lathe.preprocess: TrainTestSplit
using Plots


# Loading data and creating a DataFrame

data = DataFrame(CSV.File("data.csv"))
println("Data before preprocessing: ")
println()
display(describe(data))
println()


# Data preprocessing

# Replacing missing values with the average
data[ismissing.(data[!, :atemp]), :atemp] .= trunc(Int64, mean(skipmissing(data[!, :atemp])))
data[ismissing.(data[!, :registered]), :registered] .= trunc(Int64, mean(skipmissing(data[!, :registered])))

# Removing columns with constant values (since the data is from January 1st to January 31st)
select!(data, Not(:season))
select!(data, Not(:yr))
select!(data, Not(:mnth))

# Removing columns that are irrelevant for analysis (numerical ID and date)
select!(data, Not([:instant, :dteday]))

println("Data after preprocessing: ")
println()
display(describe(data))
println()


# Creating the model

data_train, data_test = TrainTestSplit(data, 0.8)

x_train = convert(Array{Float64}, select(data_train, Not([:cnt])))'
y_train = convert(Array{Float64}, select(data_train, :cnt))'

model = Dense(size(x_train, 1), 1)     # or: model = Dense(11, 1) 
loss(x, y) = Flux.mse(model(x), y)
ps = Flux.params(model)
opt = Flux.Descent(0.0001)

dataset = [(x_train, y_train)]

for epoch in 1:500
   Flux.train!(loss, ps, dataset, opt)  
end   


# Displaying the model

println("Model: ")
println(ps)    
println()


# Calculating errors

y_model = model(x_train)
errors_train= y_train - y_model  

mse = mean(abs.(errors_train.*errors_train))    
rmse = sqrt(mse) 
mae = mean(abs.(errors_train))                                            
mape = mean(abs.(y_model-y_train)./y_train)*100  

println("Greske: ")
println("RMSE = $rmse")
println("MSE = $mse")  
println("MAE = $mae")
println("MAPE = $mape")
println()

# Printing predicted cnt values

println("Predicted cnt values:")
println(y_model)


# Plot actual vs. predicted values
p = plot(y_train', label="Actual Values", lw=2, xlabel="Sample Index", ylabel="cnt", title="Actual vs. Predicted Values", legend=:topright)
plot!(p, y_model', label="Predicted Values", lw=2, linestyle=:dash)

display(p)