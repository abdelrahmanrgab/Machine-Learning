import matplotlib.pyplot as plt
import math
from tabulate import tabulate  # Import tabulate for pretty printing

# Logistic Regression Algorithm

# Initial Parameters
theta = [0, 0, 0]
learning_rate = 1  # Adjust the learning rate appropriately
iterations = 2  # Let's set a higher number of iterations for convergence

# Input Data
x = [0, 1, 2, 2]
x2 = [0, 0, 2, 3]
y = [0, 0, 1, 1]

# Lists to store cost values for each iteration
cost_values = []


# Hypothesis function
def hypothesis(theta, x, x2):
    z = theta[0] + theta[1] * x + theta[2] * x2
    return 1 / (1 + math.exp(-z))


# Gradient Descent
for iter in range(iterations):
    total_error = 0
    total_errorX1 = 0
    total_errorX2 = 0
    costi = 0
    sum_costi = 0;

    table = []

    for i in range(len(x)):
        h = hypothesis(theta, x[i], x2[i])

        error = h - y[i]
        total_error += error

        errorX1 = error * x[i]
        total_errorX1 += errorX1

        errorX2 = error * x2[i]
        total_errorX2 += errorX2

        costi = y[i] * math.log(h) + (1 - y[i]) * math.log(1 - h)
        sum_costi +=costi

        table.append([x[i], x2[i], y[i], h,error, errorX1, errorX2, costi])

    cost = -1 / len(x) * sum_costi
    cost_values.append(cost)

    theta[0] -= (learning_rate/len(x))* total_error
    theta[1] -= (learning_rate/len(x))* total_errorX1
    theta[2] -= (learning_rate/len(x)) * total_errorX2

    # Output Results
    print(f'Iteration {iter + 1}:')
    print(tabulate(table, headers=["x1", "x2", "y", "z", "h", "error","total_errorX1", "total_errorX2", "costi"], tablefmt="grid"))
    print('Total Error:', total_error)
    print('Cost:', cost)
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


# Plotting the points and the hypothesis function
plt.scatter(x, y, c='green', label='x')
plt.scatter(x2, y, c='red', label='x2')

# Plotting the hypothesis function
h_values = [hypothesis(theta, xi, xi2) for xi, xi2 in zip(x, x2)]
plt.plot(x, h_values, label='h', color='blue')

plt.title('Plot of Points and Hypothesis Function')
plt.xlabel('x / x2')
plt.ylabel('y / h')
plt.legend()
plt.grid(True)
plt.show()
