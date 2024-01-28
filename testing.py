from dataprocess import X_test, y_test;
import numpy as np;
from Neural_Network import runNetwork;
from Neural_Network import loss_function;

print("\nHere in my NN for predicting the grade of students on an exam given other parameters")

output_layer_array_test = np.array([])
for i in range(X_test.shape[0]):
    output_layer = runNetwork(X_test.iloc[i])
    output_layer_array_test = np.append(output_layer_array_test, output_layer) 
print(f'\nTesting Loss: {loss_function(y_test, output_layer_array_test)}')


print("\nTo illustrate the network at work, let's compare a few predicted testing values to their actual values:")

print("\n\nFirst is row 2:")
print("Predicted: " + str(output_layer_array_test[2]))
print("Actual: " + str(y_test.iloc[2]))

print("\nNext is row 15:")
print("Predicted: " + str(output_layer_array_test[15]))
print("Actual: " + str(y_test.iloc[15]))

print("\nNext is row 20:")
print("Predicted: " + str(output_layer_array_test[20]))
print("Actual: " + str(y_test.iloc[20]))

print("\nNext is row 30:")
print("Predicted: " + str(output_layer_array_test[30]))
print("Actual: " + str(y_test.iloc[30]))
