import numpy as np

## MS2

class PCA(object):
    """
    PCA dimensionality reduction class.
    
    Feel free to add more functions to this class if you need,
    but make sure that __init__(), find_principal_components(), and reduce_dimension() work correctly.
    """

    def __init__(self, d):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            d (int): dimensionality of the reduced space
        """
        self.d = d
        
        # the mean of the training data (will be computed from the training data and saved to this variable)
        self.mean = None 
        # the principal components (will be computed from the training data and saved to this variable)
        self.W = None

    def find_principal_components(self, training_data):
        """
        Finds the principal components of the training data and returns the explained variance in percentage.

        IMPORTANT: 
            This function should save the mean of the training data and the kept principal components as
            self.mean and self.W, respectively.

        Arguments:
            training_data (array): training data of shape (N,D)
        Returns:
            exvar (float): explained variance of the kept dimensions (in percentage, i.e., in [0,100])
        """


        self.mean = np.mean(training_data, axis = 0)
        le_data_centre_autour_de_origine = training_data - self.mean
        matrice_cov  = np.cov(le_data_centre_autour_de_origine, rowvar = False)
        e_values, e_component_vectors = np.linalg.eigh(matrice_cov)
        i_decroissant_ordre = np.argsort(e_values)[::-1]
        self.W = e_component_vectors[:, i_decroissant_ordre[: self.d]]


        variance_totale = np.sum(e_values)

        exvar = (np.sum(e_values[i_decroissant_ordre[:self.d]]) / variance_totale * 100)

        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return exvar

    def reduce_dimension(self, data):
        """
        Reduce the dimensionality of the data using the previously computed components.

        Arguments:
            data (array): data of shape (N,D)
        Returns:
            data_reduced (array): reduced data of shape (N,d)
        """

        le_data_centre_autour_de_origine = data - self.mean
        data_reduced = np.dot(le_data_centre_autour_de_origine, self.W) #to reduce to d dimension

        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        return data_reduced
        

