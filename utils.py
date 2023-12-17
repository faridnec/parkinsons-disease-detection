import csv
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_INDEX = 17
PARKINSON_CSV = './data/PARKINSONS.csv'

def load_data():
    X = []
    y = []
    
    # names = []
    
    with open(PARKINSON_CSV) as csvfile:
        
        reader = csv.reader(csvfile, delimiter=',')
        
        #skip the header
        next(reader)
        
        for row in reader:
            # Extract input variables from the row (exclude the output variable)
            #skip the name column as we begin at index 1
            input_data = [float(value) for value in row[1:OUTPUT_INDEX] + row[OUTPUT_INDEX+1:]]
            output_data = int(row[OUTPUT_INDEX])
            
            X.append(input_data)
            y.append(output_data)
            
            # name = row[0]
            # names.append(name)
    
    # Convert X and y to NumPy arrays
    X = np.array(X)
    y = np.array(y)

    return X,y

def sig(z):
 
    return 1/(1+np.exp(-z))

# def map_feature(X1, X2):
#     """
#     Feature mapping function to polynomial features    
#     """
#     X1 = np.atleast_1d(X1)
#     X2 = np.atleast_1d(X2)
#     degree = 6
#     out = []
#     for i in range(1, degree+1):
#         for j in range(i + 1):
#             out.append((X1**(i-j) * (X2**j)))
#     return np.stack(out, axis=1)

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)

