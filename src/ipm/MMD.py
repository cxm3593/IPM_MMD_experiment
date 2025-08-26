"""
Maximum Mean Discrepancy (MMD) implementation for point clouds
Author: Chengyi Ma
"""

import numpy as np
from typing import Optional, Callable


class MMD:
    """
    A minimal class for computing Maximum Mean Discrepancy between point clouds.
    """
    
    def __init__(self, kernel: str = 'rbf', gamma: float = 1.0):
        """
        Initialize MMD calculator.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            gamma: Kernel parameter (bandwidth for RBF)
        """
        self.kernel = kernel
        self.gamma = gamma
        self.kernel_func = self._get_kernel_function()
    
    def _get_kernel_function(self) -> Callable:
        """Get the kernel function based on kernel type."""
        if self.kernel == 'rbf':
            return self._rbf_kernel
        elif self.kernel == 'linear':
            return self._linear_kernel
        elif self.kernel == 'polynomial':
            return self._polynomial_kernel
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel}")
    
    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        RBF (Gaussian) kernel: k(x,y) = exp(-gamma * ||x-y||^2)
        
        Args:
            X: Array of shape (n, d)
            Y: Array of shape (m, d)
            
        Returns:
            Kernel matrix of shape (n, m)
        """
        # Compute pairwise squared distances
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        distances_sq = X_norm + Y_norm.T - 2 * np.dot(X, Y.T)
        
        return np.exp(-self.gamma * distances_sq)
    
    def _linear_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Linear kernel: k(x,y) = x^T * y
        
        Args:
            X: Array of shape (n, d)
            Y: Array of shape (m, d)
            
        Returns:
            Kernel matrix of shape (n, m)
        """
        return np.dot(X, Y.T)
    
    def _polynomial_kernel(self, X: np.ndarray, Y: np.ndarray, degree: int = 3) -> np.ndarray:
        """
        Polynomial kernel: k(x,y) = (gamma * x^T * y + 1)^degree
        
        Args:
            X: Array of shape (n, d)
            Y: Array of shape (m, d)
            degree: Polynomial degree
            
        Returns:
            Kernel matrix of shape (n, m)
        """
        return (self.gamma * np.dot(X, Y.T) + 1) ** degree
    
    def compute(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute MMD between two point sets.
        
        MMD^2 = (1/n^2) * sum_i,j k(x_i, x_j) + (1/m^2) * sum_i,j k(y_i, y_j) 
                - (2/nm) * sum_i,j k(x_i, y_j)
        
        Args:
            X: First point set of shape (n, d)
            Y: Second point set of shape (m, d)
            
        Returns:
            MMD value (squared)
        """
        n, m = X.shape[0], Y.shape[0]
        
        # Compute kernel matrices
        K_XX = self.kernel_func(X, X)
        K_YY = self.kernel_func(Y, Y)
        K_XY = self.kernel_func(X, Y)
        
        # Compute MMD^2
        term1 = np.sum(K_XX) / (n * n)
        term2 = np.sum(K_YY) / (m * m)
        term3 = 2 * np.sum(K_XY) / (n * m)
        
        mmd_squared = term1 + term2 - term3
        
        return mmd_squared
    
    def compare_point_clouds(self, pc1, pc2) -> float:
        """
        Compute MMD between two PointCloud objects.
        
        Args:
            pc1: First PointCloud object
            pc2: Second PointCloud object
            
        Returns:
            MMD value (squared)
        """
        if pc1.n_points == 0 or pc2.n_points == 0:
            raise ValueError("Cannot compute MMD with empty point clouds")
            
        return self.compute(pc1.points, pc2.points)
    
    def __repr__(self) -> str:
        """String representation of the MMD calculator."""
        return f"MMD(kernel='{self.kernel}', gamma={self.gamma})"
