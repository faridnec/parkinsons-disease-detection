import csv
import numpy as np


OUTPUT_INDEX = 17
PARKINSON_CSV = './data/PARKINSONS.csv'

# def load_data():
#     data = np.loadtxt(PARKINSON_CSV, delimiter=',', skiprows=1)
#     #separate the input variable from output
#     X = data[:,:OUTPUT_INDEX] + data[:,OUTPUT_INDEX+1:]
#     y = data[:,OUTPUT_INDEX]
#     return X, y

def parse_data():
    X = []
    y = []
    
    # names = []
    
    with open(PARKINSON_CSV) as csvfile:
        
        reader = csv.reader(csvfile, delimiter=',')
        
        #skip the header
        next(reader)
        
        for row in reader:
            # Extract input variables from the row (exclude the output variable)
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

