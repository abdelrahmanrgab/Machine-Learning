import matplotlib.pyplot as plt

# Linear Regression Algorithm

# Initial Parameters
theta = [1, 1,1]
learning_rate = 0.002
iterations = 2

# Input Data
x = [0, 1, 2, 3, 4,5]
x2= [0,1,4,9,16,25]
y = [2.1,7.7,13.6,27.2,40.9,61.1]

# Lists to store cost values for each iteration
cost_values = []


# Gradient Descent
for iter in range(1, iterations + 1):
    total_error = 0
    total_errorX1 = 0
    total_errorX2 = 0
    total_squared_error = 0

    print(f'Iteration {iter}:')
    table = []

    for i in range(len(x)):
        # Hypothesis function
        h = theta[0] + theta[1] * x[i] + theta[2] * x2[i] 

        # Error
        error = h - y[i]
        total_error += error

        # Errors for x1 
        errorX1 = error * x[i]
        total_errorX1 += errorX1

        # Errors for x2 
        errorX2 = error * x2[i]
        total_errorX2 += errorX2

        # for cost function
        total_squared_error += error ** 2
        cost = 1/(2 *len(x))*total_squared_error

        # Append data to the table
        table.append([x[i], y[i], h, error, errorX1])

    # Output Results
    print(tabulate(table, headers=["x", "y", "h", "error", "errorX1","errorX2"], tablefmt="grid"))
    print('Total Error:', total_error)
    print('Total Error x1:', total_errorX1)
    print('Total Error x2:', total_errorX2)
    print('cost:', cost)
    cost_values.append(cost)


    # Update theta[0] and theta[1]
    theta[0] -= (learning_rate / len(x)) * total_error
    theta[1] -= (learning_rate / len(x)) * total_errorX1
    theta[2] -= (learning_rate / len(x)) * total_errorX2

    print('Theta:', theta)  # Print the parameters for each iteration
    print('---')

# Final Results
print('Final Parameters:', theta)
# Plotting the cost function
plt.plot(range(1, iterations + 1), cost_values, marker='o')
plt.title('Cost Function')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.grid(True)
plt.show()
