import numpy as np
from dataprocess import X_train, y_train




# TODO:
# Add Stochastic
# Add Momentum




Size_Of_First_Hidden_Layer = 25
Size_Of_Second_Hidden_Layer = 44
Size_Of_Third_Hidden_Layer = 60
Size_Of_Output_Layer = 1 


input_length = len(X_train.T)


weights_0 = np.random.uniform(-1, 1, (input_length, Size_Of_First_Hidden_Layer))* 0.01
weights_1 = np.random.uniform(-1, 1, (Size_Of_First_Hidden_Layer, Size_Of_Second_Hidden_Layer))* 0.01
weights_2 = np.random.uniform(-1, 1, (Size_Of_Second_Hidden_Layer, Size_Of_Third_Hidden_Layer))* 0.01
weights_3 = np.random.uniform(-1, 1, (Size_Of_Third_Hidden_Layer, Size_Of_Output_Layer))* 0.01





# Assuming you have defined a loss function
def loss_function(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))

# Initialize biases
bias_0 = np.zeros(Size_Of_First_Hidden_Layer)
bias_1 = np.zeros(Size_Of_Second_Hidden_Layer)
bias_2 = np.zeros(Size_Of_Third_Hidden_Layer)
bias_3 = np.zeros(Size_Of_Output_Layer)

learning_rate = 0.001
epochs = 10

def runNetworkForTrain(X):

    layer_0 = X
    layer_1 = np.maximum(0, np.dot(layer_0,weights_0) + bias_0)
    layer_2 = np.maximum(0, np.dot(layer_1,weights_1) + bias_1)
    layer_3 = np.maximum(0, np.dot(layer_2, weights_2) + bias_2)

        
    output_layer = np.maximum(0, np.dot(layer_3,weights_3) + bias_3)

    return output_layer, layer_3, layer_2, layer_1, layer_0


def runNetwork(X):

    layer_0 = X
    layer_1 = np.maximum(0, np.dot(layer_0,weights_0) + bias_0)
    layer_2 = np.maximum(0, np.dot(layer_1,weights_1) + bias_1)
    layer_3 = np.maximum(0, np.dot(layer_2, weights_2) + bias_2)

        
    output_layer = np.maximum(0, np.dot(layer_3,weights_3) + bias_3)

    return output_layer


#Training

output_layer_array = np.array([])
for epoch in range(epochs): 
    for i in range(X_train.shape[0]):
     
        input_layer = X_train.iloc[i]
        
        output_layer, layer_3, layer_2, layer_1, layer_0 = runNetworkForTrain(input_layer)
    
        output_error = 2 * (output_layer - y_train.iloc[i])
        output_error = output_error.astype(np.float64)
        layer_3_error = weights_3.dot(output_error)
        layer_2_error = weights_2.dot(layer_3_error)
        layer_1_error = weights_1.dot(layer_2_error)
        output_layer_array = np.append(output_layer_array, output_layer) 
        
        
        weights_3 -= learning_rate * np.outer(layer_3, output_error).astype(np.float64)
        bias_3 -= learning_rate * output_error
        weights_2 -= learning_rate * np.outer(layer_2, layer_3_error ).astype(np.float64)
        bias_2 -= learning_rate * layer_3_error
        weights_1 -= learning_rate * np.outer(layer_1, layer_2_error).astype(np.float64)
        bias_1 -= learning_rate * layer_2_error
        weights_0 -= learning_rate * np.outer(layer_0, layer_1_error).astype(np.float64)
        bias_0 -= learning_rate * layer_1_error

  
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_function(y_train, output_layer_array)}')
    output_layer_array = np.array([])


