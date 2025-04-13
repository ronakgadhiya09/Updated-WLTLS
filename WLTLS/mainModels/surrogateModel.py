"""Surrogate model for computing gradients in adversarial training"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

class AdversarialTrainer:
    def __init__(self, wltls_model, epsilon=0.1, alpha=0.01):
        """
        Initialize adversarial trainer
        Args:
            wltls_model: The main W-LTLS model
            epsilon: Maximum perturbation for FGSM
            alpha: Step size for FGSM
        """
        self.wltls_model = wltls_model
        self.epsilon = epsilon
        self.alpha = alpha
        
        # Initialize surrogate model
        self.surrogate = SurrogateModel(wltls_model.DIM, wltls_model.LABELS)
        self.optimizer = torch.optim.Adam(self.surrogate.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def _get_wltls_predictions(self, X):
        """Get predictions from W-LTLS model"""
        predictions = []
        for x in X:
            pred = self.wltls_model._predict(x)[0]
            predictions.append(pred)
        return np.array(predictions)
    
    def train_surrogate(self, X, y, epochs=5):
        """Train surrogate model to match W-LTLS predictions"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        for _ in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.surrogate(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            loss.backward()
            self.optimizer.step()
    
    def generate_adversarial_examples(self, X, y):
        """Generate adversarial examples using FGSM"""
        X_tensor = torch.FloatTensor(X)
        X_tensor.requires_grad = True
        y_tensor = torch.LongTensor(y)
        
        # Forward pass
        outputs = self.surrogate(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        
        # Compute gradients
        loss.backward()
        
        # FGSM step
        X_adv = X_tensor + self.epsilon * torch.sign(X_tensor.grad)
        X_adv = torch.clamp(X_adv, X_tensor.min(), X_tensor.max())
        
        return X_adv.detach().numpy()
    
    def train_with_adversarial(self, X, y, batch_size=32):
        """Train W-LTLS model with adversarial examples"""
        # First get W-LTLS predictions for surrogate training
        wltls_preds = self._get_wltls_predictions(X)
        
        # Train surrogate model
        self.train_surrogate(X, wltls_preds)
        
        # Generate adversarial examples
        X_adv = self.generate_adversarial_examples(X, y)
        
        # Combine original and adversarial examples
        X_combined = np.vstack([X, X_adv])
        y_combined = np.hstack([y, y])
        
        # Train W-LTLS on combined dataset
        return self.wltls_model.train(X_combined, y_combined)