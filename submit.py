import numpy as np
import sklearn
from scipy.linalg import khatri_rao
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# You are allowed to import any submodules of sklearn that learn linear models e.g. sklearn.svm etc
# You are not allowed to use other libraries such as keras, tensorflow etc
# You are not allowed to use any scipy routine other than khatri_rao

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_fit, my_map etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def my_fit(X_train, y_train):
################################
#  Non Editable Region Ending  #
################################
  
    X = []
    
    Y = np.where(y_train == 0.0, -1.0, 1.0)
    
    X = my_map(X_train)

    
    

    # Create a linear SVM model

    # model = LinearSVC(C=1, tol=2)
    model = LogisticRegression(C=1.0)


    # Train the model 
    
  
    model.fit(X, Y)

    # Get the coefficients (w) and intercept (b) 

    w = model.coef_.T.flatten()  # Flatten the coefficient array to 1D
    b = model.intercept_[0]  # Extract the scalar intercept

   

    # Use this method to train your model using training CRPs
    # X_train has 32 columns containing the challenge bits
    # y_train contains the responses
    
    # THE RETURNED MODEL SHOULD BE A SINGLE VECTOR AND A BIAS TERM
    # If you do not wish to use a bias term, set it to 0
    return w, b


def my_map(X):
    X = np.where(X == 0.0, -1.0, X)  # Efficiently vectorized
    n_samples, n_features = X.shape
    X_final = []

    for features in X:

        # Reverse cumulative product for preprocessed_features
        preprocessed_features = np.cumprod(features[::-1])[::-1]

        # Calculate unique pairwise products
        # Exploit symmetry and use broadcasting
        # Generate a matrix of all products

        prod_matrix = np.triu(np.outer(preprocessed_features, preprocessed_features), 1)
        unique_pairwise_products = 2 * prod_matrix[prod_matrix != 0]  # Extract upper triangular without diagonal

        # Handle individual values
        individual_values = 2 * preprocessed_features

        # Concatenate results
        result = np.concatenate((unique_pairwise_products, individual_values))
        X_final.append(result)

    return np.array(X_final)    
   
    

