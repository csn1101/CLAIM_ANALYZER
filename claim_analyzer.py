import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import re
from datetime import datetime, timedelta
import logging

class ClaimAnalyzer:
    def __init__(self):
        self.fraud_model = None
        self.risk_model = None
        self.text_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load models if they exist
        self.load_models()
        
        # If no models exist, create and train them
        if not self.is_trained:
            self.create_sample_data_and_train()
    
    def create_sample_data_and_train(self):
        """Create sample training data and train the models"""
        self.logger.info("Creating sample data and training models...")
        
        # Generate synthetic training data
        sample_data = self.generate_sample_data()
        
        # Train the models
        self.train_models(sample_data)
        
        # Save the trained models
        self.save_models()
    
    def generate_sample_data(self, n_samples=1000):
        """Generate synthetic insurance claim data for training"""
        np.random.seed(42)
        
        data = []
        
        # Fraud indicators in text
        fraud_keywords = [
            'emergency', 'urgent', 'immediate', 'rush', 'cash only',
            'no receipt', 'lost paperwork', 'estimate only', 'total loss',
            'previous damage', 'undisclosed', 'midnight', 'weekend',
            'remote location', 'no witnesses'
        ]
        
        legitimate_keywords = [
            'police report', 'witness statement', 'medical records',
            'official receipt', 'repair estimate', 'insurance adjuster',
            'documented', 'verified', 'standard procedure', 'routine claim'
        ]
        
        for i in range(n_samples):
            # Generate basic claim features
            claim_amount = np.random.lognormal(7, 1.5)  # Log-normal distribution for claim amounts
            claim_age_days = np.random.randint(0, 365)
            claimant_age = np.random.randint(18, 80)
            policy_age_days = np.random.randint(30, 3650)
            previous_claims = np.random.poisson(1.2)
            
            # Determine if this should be a fraudulent claim (20% fraud rate)
            is_fraud = np.random.random() < 0.2
            
            # Generate text based on fraud status
            if is_fraud:
                # Add more fraud indicators
                text_keywords = np.random.choice(fraud_keywords, size=np.random.randint(2, 5), replace=False)
                claim_amount *= np.random.uniform(1.5, 3.0)  # Fraudulent claims tend to be higher
                claim_text = f"Claim description: {' '.join(text_keywords)}. Incident occurred during unusual circumstances."
            else:
                text_keywords = np.random.choice(legitimate_keywords, size=np.random.randint(1, 3), replace=False)
                claim_text = f"Claim description: {' '.join(text_keywords)}. Standard incident with proper documentation."
            
            # Calculate risk score (0-100)
            risk_factors = [
                min(claim_amount / 10000, 1) * 30,  # Amount risk
                min(previous_claims / 5, 1) * 25,   # Previous claims risk
                (1 - min(policy_age_days / 365, 1)) * 20,  # New policy risk
                np.random.uniform(0, 25)  # Random risk factors
            ]
            risk_score = sum(risk_factors)
            
            # High-risk threshold
            is_high_risk = risk_score > 60
            
            data.append({
                'claim_id': f'CLM{i+1:06d}',
                'claim_text': claim_text,
                'claim_amount': round(claim_amount, 2),
                'claimant_age': claimant_age,
                'policy_age_days': policy_age_days,
                'previous_claims': previous_claims,
                'claim_age_days': claim_age_days,
                'risk_score': round(risk_score, 2),
                'is_fraud': 1 if is_fraud else 0,
                'is_high_risk': 1 if is_high_risk else 0,
                'should_approve': 1 if not is_fraud and not is_high_risk else 0
            })
        
        return pd.DataFrame(data)
    
    def extract_features(self, df):
        """Extract features for ML models"""
        # Text features
        text_features = self.text_vectorizer.fit_transform(df['claim_text'])
        
        # Numerical features
        numerical_features = df[['claim_amount', 'claimant_age', 'policy_age_days', 
                               'previous_claims', 'claim_age_days']].values
        
        # Normalize numerical features
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine features
        from scipy.sparse import hstack
        features = hstack([text_features, numerical_features])
        
        return features
    
    def train_models(self, data):
        """Train fraud detection and risk assessment models"""
        self.logger.info("Training models...")
        
        # Extract features
        X = self.extract_features(data)
        
        # Train fraud detection model
        y_fraud = data['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y_fraud, test_size=0.2, random_state=42)
        
        self.fraud_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.fraud_model.fit(X_train, y_train)
        
        # Evaluate fraud model
        y_pred = self.fraud_model.predict(X_test)
        fraud_accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Fraud detection accuracy: {fraud_accuracy:.3f}")
        
        # Train risk assessment model (using isolation forest for anomaly detection)
        self.risk_model = IsolationForest(contamination=0.1, random_state=42)
        self.risk_model.fit(X_train)
        
        self.is_trained = True
    
    def analyze_claim(self, claim_data):
        """Analyze a single claim for fraud, risk, and approval recommendation"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        # Create DataFrame for processing
        df = pd.DataFrame([claim_data])
        
        # Extract features
        text_features = self.text_vectorizer.transform(df['claim_text'])
        numerical_features = df[['claim_amount', 'claimant_age', 'policy_age_days', 
                               'previous_claims', 'claim_age_days']].values
        numerical_features = self.scaler.transform(numerical_features)
        
        from scipy.sparse import hstack
        features = hstack([text_features, numerical_features])
        
        # Fraud prediction
        fraud_probability = self.fraud_model.predict_proba(features)[0][1]
        is_likely_fraud = fraud_probability > 0.5
        
        # Risk assessment
        anomaly_score = self.risk_model.decision_function(features)[0]
        risk_score = max(0, min(100, (0.5 - anomaly_score) * 100))
        
        # Business logic for approval
        approval_recommendation = self.get_approval_recommendation(
            fraud_probability, risk_score, claim_data['claim_amount']
        )
        
        return {
            'fraud_probability': round(fraud_probability * 100, 2),
            'is_likely_fraud': is_likely_fraud,
            'risk_score': round(risk_score, 2),
            'risk_level': self.get_risk_level(risk_score),
            'approval_recommendation': approval_recommendation,
            'confidence_score': round((1 - abs(fraud_probability - 0.5) * 2) * 100, 2)
        }
    
    def get_risk_level(self, risk_score):
        """Convert risk score to categorical level"""
        if risk_score < 30:
            return "Low"
        elif risk_score < 60:
            return "Medium"
        else:
            return "High"
    
    def get_approval_recommendation(self, fraud_prob, risk_score, claim_amount):
        """Business logic for claim approval"""
        # High fraud probability
        if fraud_prob > 0.7:
            return {
                'decision': 'REJECT',
                'reason': 'High fraud probability detected',
                'requires_review': True
            }
        
        # Medium fraud probability with high amount
        if fraud_prob > 0.4 and claim_amount > 10000:
            return {
                'decision': 'REVIEW',
                'reason': 'Medium fraud risk with high claim amount',
                'requires_review': True
            }
        
        # High risk score
        if risk_score > 70:
            return {
                'decision': 'REVIEW',
                'reason': 'High risk score detected',
                'requires_review': True
            }
        
        # Low risk, auto-approve
        if fraud_prob < 0.3 and risk_score < 40:
            return {
                'decision': 'APPROVE',
                'reason': 'Low risk claim',
                'requires_review': False
            }
        
        # Default to review for moderate cases
        return {
            'decision': 'REVIEW',
            'reason': 'Moderate risk - requires human review',
            'requires_review': True
        }
    
    def save_models(self):
        """Save trained models to disk"""
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        with open(os.path.join(models_dir, 'fraud_model.pkl'), 'wb') as f:
            pickle.dump(self.fraud_model, f)
        
        with open(os.path.join(models_dir, 'risk_model.pkl'), 'wb') as f:
            pickle.dump(self.risk_model, f)
        
        with open(os.path.join(models_dir, 'text_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.text_vectorizer, f)
        
        with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info("Models saved successfully")
    
    def load_models(self):
        """Load trained models from disk"""
        models_dir = 'models'
        
        try:
            with open(os.path.join(models_dir, 'fraud_model.pkl'), 'rb') as f:
                self.fraud_model = pickle.load(f)
            
            with open(os.path.join(models_dir, 'risk_model.pkl'), 'rb') as f:
                self.risk_model = pickle.load(f)
            
            with open(os.path.join(models_dir, 'text_vectorizer.pkl'), 'rb') as f:
                self.text_vectorizer = pickle.load(f)
            
            with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            
            self.is_trained = True
            self.logger.info("Models loaded successfully")
            
        except FileNotFoundError:
            self.logger.info("No saved models found, will train new ones")
            self.is_trained = False
