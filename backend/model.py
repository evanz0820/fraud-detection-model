"""
Credit Card Fraud Detection Model using PyTorch
Dataset: Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
import joblib
import json
import os
from datetime import datetime


class FraudDetectionNN(nn.Module):
    """
    Neural Network for Credit Card Fraud Detection
    Architecture: Deep feedforward network with dropout and batch normalization
    """
    
    def __init__(self, input_dim=30, hidden_dims=[128, 64, 32], dropout_rate=0.3):
        super(FraudDetectionNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class FraudDetectionAutoencoder(nn.Module):
    """
    Autoencoder for anomaly detection in credit card transactions
    Useful for detecting fraudulent transactions as anomalies
    """
    
    def __init__(self, input_dim=30, encoding_dim=14):
        super(FraudDetectionAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 18),
            nn.ReLU(),
            nn.Linear(18, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 18),
            nn.ReLU(),
            nn.Linear(18, 24),
            nn.ReLU(),
            nn.Linear(24, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x):
        """Calculate reconstruction error for anomaly detection"""
        reconstructed = self.forward(x)
        mse = torch.mean((x - reconstructed) ** 2, dim=1)
        return mse


class FraudDetectionTrainer:
    """
    Complete training pipeline for fraud detection models
    """
    
    def __init__(self, model_type='nn', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.training_history = []
        self.metrics = {}
        
    def load_data(self, filepath):
        """Load and preprocess the credit card fraud dataset"""
        print(f"Loading data from {filepath}...")
        
        # Load dataset
        df = pd.read_csv(filepath)
        
        # Store dataset info
        self.dataset_info = {
            'total_transactions': len(df),
            'fraud_count': int(df['Class'].sum()),
            'normal_count': int(len(df) - df['Class'].sum()),
            'fraud_percentage': float(df['Class'].mean() * 100),
            'features': list(df.columns[:-1]),
            'time_range': [float(df['Time'].min()), float(df['Time'].max())],
            'amount_range': [float(df['Amount'].min()), float(df['Amount'].max())]
        }
        
        # Separate features and target
        X = df.drop('Class', axis=1).values
        y = df['Class'].values
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, use_smote=True):
        """Preprocess data with scaling and optional SMOTE for imbalanced classes"""
        print("Preprocessing data...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Apply SMOTE for handling class imbalance
        if use_smote and self.model_type == 'nn':
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
            print(f"After SMOTE: {sum(y_train == 0)} normal, {sum(y_train == 1)} fraud")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_scaled).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test).to(self.device)
        
        self.X_test = X_test_tensor
        self.y_test = y_test_tensor
        self.X_test_raw = X_test
        self.y_test_raw = y_test
        
        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def create_model(self, input_dim=30):
        """Create the appropriate model based on model_type"""
        if self.model_type == 'nn':
            self.model = FraudDetectionNN(input_dim=input_dim).to(self.device)
        elif self.model_type == 'autoencoder':
            self.model = FraudDetectionAutoencoder(input_dim=input_dim).to(self.device)
        
        print(f"Created {self.model_type} model on {self.device}")
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=256, learning_rate=0.001):
        """Train the model"""
        print(f"\nTraining {self.model_type} model for {epochs} epochs...")
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        if self.model_type == 'nn':
            # Use weighted BCE loss for imbalanced data
            criterion = nn.BCELoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        self.training_history = []
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if self.model_type == 'nn':
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                else:
                    # For autoencoder, only train on normal transactions
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_X)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Evaluate on test set every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    if self.model_type == 'nn':
                        test_pred = self.model(self.X_test).squeeze()
                        test_loss = criterion(test_pred, self.y_test).item()
                        auc = roc_auc_score(self.y_test.cpu().numpy(), test_pred.cpu().numpy())
                        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, AUC: {auc:.4f}")
                    else:
                        test_pred = self.model(self.X_test)
                        test_loss = criterion(test_pred, self.X_test).item()
                        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': optimizer.param_groups[0]['lr']
            })
        
        print("Training complete!")
        return self.training_history
    
    def evaluate(self, threshold=0.5):
        """Evaluate model and compute comprehensive metrics"""
        print("\nEvaluating model...")
        
        self.model.eval()
        with torch.no_grad():
            if self.model_type == 'nn':
                y_pred_proba = self.model(self.X_test).squeeze().cpu().numpy()
            else:
                # For autoencoder, use reconstruction error as anomaly score
                recon_error = self.model.get_reconstruction_error(self.X_test).cpu().numpy()
                # Normalize to [0, 1]
                y_pred_proba = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min())
        
        y_test_np = self.y_test.cpu().numpy()
        
        # Find optimal threshold using precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test_np, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else threshold
        
        # Make predictions with optimal threshold
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Compute metrics
        self.metrics = {
            'accuracy': float(accuracy_score(y_test_np, y_pred)),
            'precision': float(precision_score(y_test_np, y_pred)),
            'recall': float(recall_score(y_test_np, y_pred)),
            'f1_score': float(f1_score(y_test_np, y_pred)),
            'roc_auc': float(roc_auc_score(y_test_np, y_pred_proba)),
            'average_precision': float(average_precision_score(y_test_np, y_pred_proba)),
            'optimal_threshold': float(optimal_threshold),
            'confusion_matrix': confusion_matrix(y_test_np, y_pred).tolist(),
            'classification_report': classification_report(y_test_np, y_pred, output_dict=True)
        }
        
        # Add precision-recall curve data (sampled for JSON)
        sample_indices = np.linspace(0, len(precision)-1, 100, dtype=int)
        self.metrics['precision_recall_curve'] = {
            'precision': precision[sample_indices].tolist(),
            'recall': recall[sample_indices].tolist()
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:           {self.metrics['accuracy']:.4f}")
        print(f"Precision:          {self.metrics['precision']:.4f}")
        print(f"Recall:             {self.metrics['recall']:.4f}")
        print(f"F1 Score:           {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:            {self.metrics['roc_auc']:.4f}")
        print(f"Average Precision:  {self.metrics['average_precision']:.4f}")
        print(f"Optimal Threshold:  {self.metrics['optimal_threshold']:.4f}")
        print("\nConfusion Matrix:")
        cm = self.metrics['confusion_matrix']
        print(f"  TN: {cm[0][0]:6d}  FP: {cm[0][1]:6d}")
        print(f"  FN: {cm[1][0]:6d}  TP: {cm[1][1]:6d}")
        print("="*50)
        
        return self.metrics
    
    def save_model(self, model_dir='models'):
        """Save the trained model and preprocessing artifacts"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save PyTorch model
        model_path = os.path.join(model_dir, f'{self.model_type}_model.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'input_dim': 30
        }, model_path)
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Save metrics and dataset info
        info_path = os.path.join(model_dir, 'model_info.json')
        model_info = {
            'model_type': self.model_type,
            'metrics': self.metrics,
            'dataset_info': self.dataset_info,
            'training_history': self.training_history,
            'trained_at': datetime.now().isoformat()
        }
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"\nModel saved to {model_dir}/")
        return model_path
    
    def predict(self, X):
        """Make predictions on new data"""
        self.model.eval()
        
        # Scale input
        if isinstance(X, np.ndarray):
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        else:
            X_tensor = X
        
        with torch.no_grad():
            if self.model_type == 'nn':
                proba = self.model(X_tensor).squeeze().cpu().numpy()
            else:
                recon_error = self.model.get_reconstruction_error(X_tensor).cpu().numpy()
                proba = (recon_error - recon_error.min()) / (recon_error.max() - recon_error.min() + 1e-8)
        
        threshold = self.metrics.get('optimal_threshold', 0.5)
        predictions = (proba >= threshold).astype(int)
        
        return predictions, proba


def train_fraud_detection_model(data_path, model_type='nn', epochs=50):
    """
    Main function to train the fraud detection model
    """
    trainer = FraudDetectionTrainer(model_type=model_type)
    
    # Load and preprocess data
    X, y = trainer.load_data(data_path)
    X_train, y_train, X_test, y_test = trainer.preprocess_data(X, y, use_smote=True)
    
    # Create and train model
    trainer.create_model(input_dim=X.shape[1])
    trainer.train(X_train, y_train, epochs=epochs)
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Save model
    trainer.save_model()
    
    return trainer, metrics


if __name__ == "__main__":
    import sys
    
    # Default data path
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/creditcard.csv"
    
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("\nOr use: python model.py <path_to_creditcard.csv>")
        sys.exit(1)
    
    trainer, metrics = train_fraud_detection_model(data_path, model_type='nn', epochs=50)
