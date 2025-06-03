#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import xgboost as xgb
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')
import scipy.stats as stats

class InsuranceFraudDetectionEvaluation:
    """
    A comprehensive evaluation framework for insurance fraud detection models
    based on the GCC healthcare insurance data.
    """
    
    def __init__(self, data_path):
        """
        Initialize the evaluation framework with the specified dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the insurance claims dataset.
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        self.fraud_patterns = None
        self.vectorizer = None
        self.pattern_vectors = None
        
        # Create knowledge base for RAG implementation
        self.fraud_patterns = [
            "High claim to premium ratio exceeding 50:1 may indicate fraud.",
            "Multiple claims with severity level 3 from the same customer within 6 months.",
            "Vehicle claims that exceed 80% of the total claim amount for minor severity incidents.",
            "Injury claims exceeding 40% of total claim for accidents with severity level 1.",
            "Multiple high-value claims from new customers under 25 years old.",
            "Property claims with round numbers (exactly $10,000, $5,000) may indicate estimation fraud.",
            "Claims made very soon after policy initiation.",
            "Multiple claims from the same policy with different types of damage."
        ]
        # Initialize directories for outputs
        self._initialize_system()
    
    def _initialize_system(self):
        """
        Initialize system components, including output directories.
        """
        import os
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/plots', exist_ok=True)
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/data', exist_ok=True)
        
        # Initialize TF-IDF vectorizer for RAG-based approaches
        self.vectorizer = TfidfVectorizer()
        self.pattern_vectors = self.vectorizer.fit_transform(self.fraud_patterns)
        
        print("System initialized successfully.")

    def load_and_explore_data(self):
        """
        Load and explore the dataset, providing basic statistics and data information.
        """
        print("Step 1: Loading and exploring the dataset...")
        self.df = pd.read_csv(self.data_path)
        
        print(f"Dataset shape: {self.df.shape}")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print("\nSummary statistics:")
        print(self.df.describe())
        
        print("\nChecking for missing values:")
        print(self.df.isnull().sum())
        
        # Display fraud distribution
        print("\nFraud distribution:")
        fraud_counts = self.df['is_fraudulent'].value_counts()
        print(fraud_counts)
        fraud_percentage = (fraud_counts[1] / self.df.shape[0]) * 100
        print(f"Fraud percentage: {fraud_percentage:.2f}%")
        
        return self.df
    
    def visualize_exploratory_analysis(self):
        """
        Create exploratory visualizations to understand patterns in the data.
        """
        print("\nStep 2: Creating exploratory visualizations...")
        
        # Set up the visualization style
        sns.set(style="whitegrid")
        
        # Plot 1: Fraud distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='is_fraudulent', data=self.df)
        plt.title('Fraud Distribution')
        plt.xlabel('Is Fraudulent')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
        plt.savefig('output/plots/fraud_distribution.png')
        plt.close()
        
        # Plot 2: Claim amount distribution by fraud status
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(x='is_fraudulent', y='total_claim_amount', data=self.df)
        plt.title('Total Claim Amount by Fraud Status')
        plt.xlabel('Is Fraudulent')
        plt.ylabel('Total Claim Amount')
        plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
        
        plt.subplot(1, 2, 2)
        sns.violinplot(x='is_fraudulent', y='claim_premium_ratio', data=self.df)
        plt.title('Claim Premium Ratio by Fraud Status')
        plt.xlabel('Is Fraudulent')
        plt.ylabel('Claim Premium Ratio')
        plt.xticks([0, 1], ['Legitimate', 'Fraudulent'])
        plt.tight_layout()
        plt.savefig('output/plots/claim_distributions.png')
        plt.close()
        
        # Plot 3: Patient category by fraud status
        plt.figure(figsize=(12, 6))
        fraud_by_category = pd.crosstab(self.df['patient_category'], self.df['is_fraudulent'])
        fraud_by_category_pct = fraud_by_category.div(fraud_by_category.sum(axis=1), axis=0) * 100
        fraud_by_category_pct.plot(kind='bar', stacked=True)
        plt.title('Fraud Rate by Patient Category')
        plt.xlabel('Patient Category')
        plt.ylabel('Percentage')
        plt.legend(['Legitimate', 'Fraudulent'])
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('output/plots/fraud_by_category.png')
        plt.close()
        
        # Plot 4: Severity level by fraud status
        plt.figure(figsize=(10, 6))
        fraud_by_severity = pd.crosstab(self.df['severity'], self.df['is_fraudulent'])
        fraud_by_severity_pct = fraud_by_severity.div(fraud_by_severity.sum(axis=1), axis=0) * 100
        fraud_by_severity_pct.plot(kind='bar', stacked=True)
        plt.title('Fraud Rate by Severity Level')
        plt.xlabel('Severity Level')
        plt.ylabel('Percentage')
        plt.legend(['Legitimate', 'Fraudulent'])
        plt.savefig('output/plots/fraud_by_severity.png')
        plt.close()
        
        # Plot 5: Correlation heatmap of numerical features
        plt.figure(figsize=(14, 12))
        numerical_columns = self.df.select_dtypes(include=['float64', 'int64']).columns
        correlation = self.df[numerical_columns].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('output/plots/correlation_matrix.png')
        plt.close()
        
        # Plot 6: Distribution of claim components
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        sns.histplot(self.df['claim_premium_ratio'], kde=True, bins=30)
        plt.title('Claim Premium Ratio Distribution')
        plt.axvline(x=50, color='r', linestyle='--', label='Fraud Threshold (50)')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        sns.histplot(self.df['injury_percent'], kde=True, bins=30)
        plt.title('Injury Percentage Distribution')
        plt.axvline(x=40, color='r', linestyle='--', label='Fraud Threshold (40%)')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        sns.histplot(self.df['vehicle_percent'], kde=True, bins=30)
        plt.title('Vehicle Percentage Distribution')
        plt.axvline(x=80, color='r', linestyle='--', label='Fraud Threshold (80%)')
        plt.legend()
        
        plt.subplot(2, 2, 4)
        sns.countplot(x='round_numbers', hue='is_fraudulent', data=self.df)
        plt.title('Round Numbers by Fraud Status')
        plt.xlabel('Has Round Numbers')
        plt.ylabel('Count')
        plt.legend(['Legitimate', 'Fraudulent'])
        
        plt.tight_layout()
        plt.savefig('output/plots/claim_components.png')
        plt.close()
        
        print("Visualizations saved to output/plots/ directory")
    
    def prepare_data_for_modeling(self):
        """
        Prepare the data for modeling by splitting and scaling.
        """
        print("\nStep 3: Preparing data for modeling...")
        
        # Convert categorical variables to dummy variables
        categorical_columns = ['patient_category', 'hospital']
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
        
        # Define features and target
        X = self.df.drop(['is_fraudulent', 'policy_number'], axis=1)
        y = self.df['is_fraudulent']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Scale the features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: X shape {self.X_train.shape}, y shape {self.y_train.shape}")
        print(f"Testing set: X shape {self.X_test.shape}, y shape {self.y_test.shape}")
        
        return True
    
    def train_supervised_models(self):
        """
        Train and evaluate multiple supervised learning models.
        """
        print("\nStep 4: Training supervised learning models...")
        
        # Define the models to train
        supervised_models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            'Support Vector Machine': SVC(probability=True, random_state=42),
            'Neural Network': MLPClassifier(max_iter=1000, random_state=42)
        }
        
        # Train and evaluate each model
        for name, model in supervised_models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
            
            # Evaluate the model
            self._evaluate_model(name, model, y_pred, y_prob)
            
            # Store the model
            self.models[name] = model
        
        # Determine the best model based on AUC
        best_model_name = max(self.results, key=lambda x: self.results[x]['auc'])
        print(f"\nBest supervised model: {best_model_name} with AUC: {self.results[best_model_name]['auc']:.4f}")
        
        return self.models, self.results
    
    def train_unsupervised_models(self):
        """
        Train and evaluate unsupervised anomaly detection models.
        """
        print("\nStep 5: Training unsupervised anomaly detection models...")
        
        # Define the models to train
        unsupervised_models = {
            'Isolation Forest': IsolationForest(contamination=0.3, random_state=42),
            'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=0.3),
            'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
        }
        
        # Train and evaluate each model
        for name, model in unsupervised_models.items():
            print(f"\nTraining {name}...")
            
            if name == 'Local Outlier Factor':
                # LOF doesn't have a fit_predict method for test data
                y_train_pred = model.fit_predict(self.X_train)
                # Convert -1 (outlier) to 1 (fraud) and 1 (inlier) to 0 (legitimate)
                y_train_pred = np.where(y_train_pred == -1, 1, 0)
                
                # Use the same model for test predictions
                clf = LocalOutlierFactor(n_neighbors=20, contamination=0.3, novelty=True)
                clf.fit(self.X_train)
                y_pred = clf.predict(self.X_test)
                y_pred = np.where(y_pred == -1, 1, 0)
                
                # No probability scores for LOF
                y_prob = y_pred
            
            elif name == 'DBSCAN':
                # DBSCAN clustering
                model.fit(self.X_train)
                train_labels = model.labels_
                
                # Assume cluster -1 (noise) represents fraud
                y_train_pred = np.where(train_labels == -1, 1, 0)
                
                # Use a nearest neighbors approach to predict test data
                from sklearn.neighbors import NearestNeighbors
                nbrs = NearestNeighbors(n_neighbors=5).fit(self.X_train)
                distances, indices = nbrs.kneighbors(self.X_test)
                
                # If the majority of neighbors are in noise cluster, classify as fraud
                y_pred = np.zeros(self.X_test.shape[0])
                for i in range(self.X_test.shape[0]):
                    neighbor_labels = [train_labels[j] for j in indices[i]]
                    if neighbor_labels.count(-1) >= 3:  # If 3 or more neighbors are noise
                        y_pred[i] = 1
                
                # No probability scores for DBSCAN
                y_prob = y_pred
            
            else:  # Isolation Forest
                model.fit(self.X_train)
                # Convert the predictions: -1 (outlier) to 1 (fraud), 1 (inlier) to 0 (legitimate)
                y_train_pred = np.where(model.predict(self.X_train) == -1, 1, 0)
                y_pred = np.where(model.predict(self.X_test) == -1, 1, 0)
                
                # Calculate anomaly scores (higher score = more likely to be fraud)
                y_prob = -model.score_samples(self.X_test)
                # Normalize to 0-1 range
                y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())
            
            # Evaluate the model
            self._evaluate_unsupervised_model(name, model, y_pred, y_prob)
            
            # Store the model
            self.models[name] = model
        
        return self.models, self.results
    
    def train_hybrid_model(self):
        """
        Train and evaluate a hybrid model combining supervised and unsupervised approaches.
        This is similar to the approach described in the thesis for combining multiple methods.
        """
        print("\nStep 6: Training hybrid model...")
        
        # Select the best supervised model
        best_supervised_name = max([name for name in self.results if name not in ['Isolation Forest', 'Local Outlier Factor', 'DBSCAN']], 
                                  key=lambda x: self.results[x]['auc'])
        best_supervised = self.models[best_supervised_name]
        
        # Select the best unsupervised model
        unsupervised_models = ['Isolation Forest', 'Local Outlier Factor', 'DBSCAN']
        unsupervised_results = {name: self.results[name] for name in unsupervised_models if name in self.results}
        best_unsupervised_name = max(unsupervised_results, key=lambda x: unsupervised_results[x]['f1'])
        best_unsupervised = self.models[best_unsupervised_name]
        
        print(f"Using {best_supervised_name} and {best_unsupervised_name} for the hybrid model")
        
        # Get predictions from supervised model
        supervised_pred = best_supervised.predict(self.X_test)
        supervised_prob = best_supervised.predict_proba(self.X_test)[:, 1] if hasattr(best_supervised, "predict_proba") else supervised_pred
        
        # Get predictions from unsupervised model
        if best_unsupervised_name == 'Isolation Forest':
            unsupervised_pred = np.where(best_unsupervised.predict(self.X_test) == -1, 1, 0)
            unsupervised_prob = -best_unsupervised.score_samples(self.X_test)
            # Normalize to 0-1 range
            unsupervised_prob = (unsupervised_prob - unsupervised_prob.min()) / (unsupervised_prob.max() - unsupervised_prob.min())
            
        elif best_unsupervised_name == 'Local Outlier Factor':
            # Create a new LOF model with novelty=True
            clf = LocalOutlierFactor(n_neighbors=20, contamination=0.3, novelty=True)
            clf.fit(self.X_train)
            unsupervised_pred = np.where(clf.predict(self.X_test) == -1, 1, 0)
            # LOF doesn't provide probabilities, so we use binary predictions
            unsupervised_prob = unsupervised_pred
            
        else:  # DBSCAN - use the nearest neighbors approach
            # DBSCAN doesn't have a predict method, so we use the same approach as in train_unsupervised_models
            model = self.models[best_unsupervised_name]
            train_labels = model.labels_
            
            # Use nearest neighbors to predict test data
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5).fit(self.X_train)
            distances, indices = nbrs.kneighbors(self.X_test)
            
            unsupervised_pred = np.zeros(self.X_test.shape[0])
            for i in range(self.X_test.shape[0]):
                neighbor_labels = [train_labels[j] for j in indices[i]]
                if neighbor_labels.count(-1) >= 3:  # If 3 or more neighbors are noise
                    unsupervised_pred[i] = 1
            
            # No probability scores for DBSCAN
            unsupervised_prob = unsupervised_pred
        
        # Combine predictions using weighted average of probabilities
        # Give more weight to the supervised model as it generally performs better
        hybrid_prob = (0.7 * supervised_prob) + (0.3 * unsupervised_prob)
        hybrid_pred = (hybrid_prob > 0.5).astype(int)
        
        # Evaluate the hybrid model
        self._evaluate_model('Hybrid Model', None, hybrid_pred, hybrid_prob)
        
        print(f"Hybrid Model AUC: {self.results['Hybrid Model']['auc']:.4f}")
        
        return self.results['Hybrid Model']
    
    def implement_rag_enhanced_model(self):
        """
        Implement a RAG-enhanced fraud detection model.
        This simulates the approach described in the thesis using RAG for fraud detection.
        """
        print("\nStep 7: Implementing RAG-enhanced model...")
        
        # Select the best supervised model for base predictions
        best_supervised_name = max([name for name in self.results if name not in ['Isolation Forest', 'Local Outlier Factor', 'DBSCAN', 'Hybrid Model']], 
                                  key=lambda x: self.results[x]['auc'])
        best_supervised = self.models[best_supervised_name]
        
        print(f"Using {best_supervised_name} as the base model for RAG enhancement")
        
        # Get base predictions from supervised model
        base_pred = best_supervised.predict(self.X_test)
        base_prob = best_supervised.predict_proba(self.X_test)[:, 1] if hasattr(best_supervised, "predict_proba") else base_pred
        
        # Original feature names (before dummies)
        original_feature_names = self.feature_names
        
        # Convert test data back to a DataFrame for interpretability
        X_test_df = pd.DataFrame(self.X_test, columns=self.feature_names)
        
        # RAG enhancement: For each test instance, create a query and retrieve relevant knowledge
        rag_enhanced_prob = np.copy(base_prob)
        
        for i in range(len(self.X_test)):
            # Skip if already high confidence
            if base_prob[i] > 0.9 or base_prob[i] < 0.1:
                continue
                
            # Create a query based on the test instance
            # In a real application, this would include more context from the claim
            query = self._create_claim_query(X_test_df.iloc[i])
            
            # Retrieve relevant knowledge
            relevant_patterns = self.retrieve_knowledge(query)
            
            # Check if any retrieved patterns indicate fraud
            fraud_indicators = 0
            for pattern in relevant_patterns:
                # Check if the pattern suggests fraud based on the claim
                if self._pattern_matches_claim(pattern, X_test_df.iloc[i]):
                    fraud_indicators += 1
            
            # Adjust probability based on RAG findings
            if fraud_indicators > 0:
                # Increase fraud probability based on number of matching patterns
                adjustment = min(0.2 * fraud_indicators, 0.4)  # Cap at 0.4 increase
                rag_enhanced_prob[i] = min(rag_enhanced_prob[i] + adjustment, 1.0)
        
        # Final predictions
        rag_pred = (rag_enhanced_prob > 0.5).astype(int)
        
        # Evaluate the RAG-enhanced model
        self._evaluate_model('RAG-Enhanced Model', None, rag_pred, rag_enhanced_prob)
        
        print(f"RAG-Enhanced Model AUC: {self.results['RAG-Enhanced Model']['auc']:.4f}")
        
        return self.results['RAG-Enhanced Model']
    
    def _create_claim_query(self, claim):
        """
        Create a query string from a claim for RAG retrieval.
        """
        # Extract relevant claim information
        claim_str = f"Claim with "
        
        # Add claim amount information
        if 'total_claim_amount' in claim:
            claim_str += f"total amount {claim['total_claim_amount']:.2f} "
            
        # Add claim ratio information
        if 'claim_premium_ratio' in claim:
            claim_str += f"and claim ratio {claim['claim_premium_ratio']:.2f} "
            
        # Add injury percentage information
        if 'injury_percent' in claim:
            claim_str += f"and injury percentage {claim['injury_percent']:.2f} "
            
        # Add vehicle percentage information
        if 'vehicle_percent' in claim:
            claim_str += f"and vehicle percentage {claim['vehicle_percent']:.2f} "
            
        # Add severity information
        if 'severity' in claim:
            claim_str += f"with severity level {claim['severity']} "
            
        # Add round numbers information
        if 'round_numbers' in claim and claim['round_numbers'] == 1:
            claim_str += "with round numbers in the claim "
            
        return claim_str
    
    def _pattern_matches_claim(self, pattern, claim):
        """
        Check if a fraud pattern matches a specific claim.
        """
        if "claim to premium ratio exceeding 50" in pattern.lower() and claim.get('claim_premium_ratio', 0) > 50:
            return True
            
        if "vehicle claims that exceed 80%" in pattern.lower() and claim.get('vehicle_percent', 0) > 80 and claim.get('severity', 0) == 1:
            return True
            
        if "injury claims exceeding 40%" in pattern.lower() and claim.get('injury_percent', 0) > 40:
            return True
            
        if "round numbers" in pattern.lower() and claim.get('round_numbers', 0) == 1:
            return True
            
        return False
    
    def retrieve_knowledge(self, query, top_k=3):
        """
        Retrieve relevant fraud patterns for a given query.
        """
        # Convert query to vector using vectorizer
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(query_vector, self.pattern_vectors).flatten()
        
        # Get top_k indices
        top_indices = similarity_scores.argsort()[-top_k:][::-1]
        
        return [self.fraud_patterns[idx] for idx in top_indices]
    
    def _evaluate_model(self, name, model, y_pred, y_prob):
        """
        Evaluate a model and store its performance metrics.
        """
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # AUC may not be available for all models
        try:
            auc_score = roc_auc_score(self.y_test, y_prob)
        except:
            auc_score = 0.5  # Default value if AUC cannot be calculated
        
        # Classification report and confusion matrix
        report = classification_report(self.y_test, y_pred)
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store results
        self.results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc_score,
            'report': report,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        # Print results
        print(f"{name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc_score:.4f}")
        
        # Plot ROC curve
        if auc_score > 0.5:
            self._plot_roc_curve(name, y_prob)
        
        # Plot confusion matrix
        self._plot_confusion_matrix(name, cm)
        
        return self.results[name]
    
    def _evaluate_unsupervised_model(self, name, model, y_pred, y_prob):
        """
        Evaluate an unsupervised model and store its performance metrics.
        """
        # For unsupervised models, some metrics like AUC might not be directly applicable
        # But we can still calculate standard classification metrics
        return self._evaluate_model(name, model, y_pred, y_prob)
    
    def _plot_roc_curve(self, name, y_prob):
        """
        Plot ROC curve for a model.
        """
        fpr, tpr, _ = roc_curve(self.y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")
        plt.savefig(f'output/plots/roc_curve_{name.replace(" ", "_").lower()}.png')
        plt.close()
    
    def _plot_confusion_matrix(self, name, cm):
        """
        Plot confusion matrix for a model.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {name}')
        plt.savefig(f'output/plots/confusion_matrix_{name.replace(" ", "_").lower()}.png')
        plt.close()
    
    def compare_all_models(self):
        """
        Compare all trained models and visualize their performance.
        """
        print("\nStep 8: Comparing all models...")
        
        if not self.results:
            print("No models have been trained yet.")
            return
        
        # Extract performance metrics for each model
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        precisions = [self.results[name]['precision'] for name in model_names]
        recalls = [self.results[name]['recall'] for name in model_names]
        f1_scores = [self.results[name]['f1'] for name in model_names]
        aucs = [self.results[name]['auc'] for name in model_names]
        
        # Create a DataFrame for easy visualization
        comparison_df = pd.DataFrame({
            'Model': model_names,
            'Accuracy': accuracies,
            'Precision': precisions,
            'Recall': recalls,
            'F1 Score': f1_scores,
            'AUC': aucs
        })
        
        # Sort by AUC (or another metric of choice)
        comparison_df = comparison_df.sort_values('AUC', ascending=False)
        
        # Display the comparison
        print("\nModel Performance Comparison:")
        print(comparison_df)
        
        # Save the comparison to CSV
        comparison_df.to_csv('output/model_comparison.csv', index=False)
        print("Model comparison saved to output/model_comparison.csv")
        
        # Visualize the comparison
        plt.figure(figsize=(14, 8))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        bar_width = 0.15
        index = np.arange(len(model_names))
        
        for i, metric in enumerate(metrics):
            plt.bar(index + i * bar_width, comparison_df[metric], bar_width, label=metric)
        
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(index + bar_width * 2, comparison_df['Model'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('output/plots/model_comparison.png')
        plt.close()
        
        # Plot precision-recall curves for all models
        plt.figure(figsize=(10, 8))
        
        for name in model_names:
            if self.results[name]['auc'] > 0.5:  # Only plot meaningful curves
                y_prob = self.results[name]['y_prob']
                precision, recall, _ = precision_recall_curve(self.y_test, y_prob)
                average_precision = average_precision_score(self.y_test, y_prob)
                plt.plot(recall, precision, lw=2, label=f'{name} (AP={average_precision:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('output/plots/precision_recall_curves.png')
        plt.close()
        
        return comparison_df
    
    def feature_importance_analysis(self):
        """
        Analyze feature importance for the best model.
        """
        print("\nStep 9: Analyzing feature importance...")
        
        # Find the best tree-based model
        tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']
        tree_model_results = {name: self.results[name] for name in tree_models if name in self.results}
        
        if not tree_model_results:
            print("No tree-based models available for feature importance analysis.")
            return
        
        best_tree_model_name = max(tree_model_results, key=lambda x: tree_model_results[x]['auc'])
        best_tree_model = self.models[best_tree_model_name]
        
        print(f"Using {best_tree_model_name} for feature importance analysis")
        
        # Get feature importances
        if hasattr(best_tree_model, 'feature_importances_'):
            importances = best_tree_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Create a DataFrame for visualization
            feature_importance_df = pd.DataFrame({
                'Feature': [self.feature_names[i] for i in indices],
                'Importance': [importances[i] for i in indices]
            })
            
            # Display top features
            print("\nTop 15 important features:")
            print(feature_importance_df.head(15))
            
            # Plot feature importances
            plt.figure(figsize=(12, 8))
            plt.barh(feature_importance_df['Feature'].head(15), feature_importance_df['Importance'].head(15))
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title(f'Feature Importance - {best_tree_model_name}')
            plt.tight_layout()
            plt.savefig('output/plots/feature_importance.png')
            plt.close()
            
            # Save feature importances to CSV
            feature_importance_df.to_csv('output/feature_importance.csv', index=False)
            print("Feature importance saved to output/feature_importance.csv")
            
            return feature_importance_df
        else:
            print(f"{best_tree_model_name} does not provide feature importances.")
            return None
    
    def run_complete_evaluation(self):
        """
        Run the complete model evaluation pipeline.
        """
        print("Running complete fraud detection model evaluation pipeline...")
        
        # Step 1: Load and explore data
        self.load_and_explore_data()
        
        # Step 2: Create exploratory visualizations
        self.visualize_exploratory_analysis()
        
        # Step 3: Prepare data for modeling
        self.prepare_data_for_modeling()
        
        # Step 4: Train supervised models
        self.train_supervised_models()
        
        # Step 5: Train unsupervised models
        self.train_unsupervised_models()
        
        # Step 6: Train hybrid model
        self.train_hybrid_model()
        
        # Step 7: Implement RAG-enhanced model
        self.implement_rag_enhanced_model()
        
        # Step 8: Compare all models
        comparison_df = self.compare_all_models()
        
        # Step 9: Analyze feature importance
        self.feature_importance_analysis()
        
        print("\nFraud detection model evaluation completed successfully.")
        
        # Return the best model based on AUC
        best_model_name = comparison_df.iloc[0]['Model']
        best_auc = comparison_df.iloc[0]['AUC']
        print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")
        
        return best_model_name, best_auc


# Example usage
if __name__ == "__main__":
    # Initialize the evaluation framework
    evaluator = InsuranceFraudDetectionEvaluation('synthetic_claims.csv')
    
    # Run the complete evaluation
    best_model, best_auc = evaluator.run_complete_evaluation()
    
    print(f"Evaluation complete. Best model: {best_model} with AUC: {best_auc:.4f}")


# In[2]:


from synthetic_data_generator import generate_synthetic_dataset

# Generate the synthetic data
generate_synthetic_dataset('synthetic_claims.csv', num_claims=1000, seed=42)


# In[3]:


# Define the synthetic data generator function directly
def generate_synthetic_dataset(output_file, num_claims=1000, seed=42):
    """
    Generate a synthetic healthcare insurance claims dataset with fraud indicators.
    
    Parameters:
    -----------
    output_file : str
        Path to save the generated dataset.
    num_claims : int, default=1000
        Number of claims to generate.
    seed : int, default=42
        Random seed for reproducibility.
    
    Returns:
    --------
    pandas.DataFrame
        The generated claims dataset.
    """
    import pandas as pd
    import numpy as np
    import random
    import os
    import json
    from datetime import datetime, timedelta
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Generating {num_claims} synthetic healthcare claims...")
    
    # Define Qatar hospitals
    hospitals = [f"Qatar Hospital {i}" for i in range(1, 13)]
    
    # Define patient categories
    patient_categories = [
        'GCC Resident',  # Resident from a GCC country
        'GCC Visitor',   # Visitor from a GCC country
        'Resident',      # Non-GCC resident
        'Visitor',       # Non-GCC visitor
        'Baby',          # Infant patient
        'Handicapped'    # Patient with disability
    ]
    
    # Define severity levels
    severity_levels = [1, 2, 3]  # 1 = Low, 2 = Medium, 3 = High
    
    # Set average claim amounts based on Qatar data (using estimates)
    avg_inpatient_amount = 200000    # Average inpatient claim amount
    avg_outpatient_amount = 7000000  # Average outpatient claim amount
    
    # Generate claims data
    claims_data = []
    
    # Define fraud patterns based on the thesis document
    def is_likely_fraudulent(claim):
        """Check if a claim matches known fraud patterns."""
        fraud_indicators = 0
        
        # High claim to premium ratio (Section 3.2 of thesis)
        if claim['claim_premium_ratio'] > 50:
            fraud_indicators += 1
            
        # High severity with high claim amount (Section 2.1)
        if claim['severity'] == 3 and claim['inpatient_amount'] > avg_inpatient_amount * 2:
            fraud_indicators += 1
            
        # High injury percentage (Section 3.2)
        if claim['injury_percent'] > 40:
            fraud_indicators += 1
            
        # High ratio for visitors (Section 6.1.2)
        if claim['claim_premium_ratio'] > 30 and claim['patient_category'] == 'Visitor':
            fraud_indicators += 1
            
        # Extremely high inpatient claim
        if claim['inpatient_amount'] > avg_inpatient_amount * 3:
            fraud_indicators += 1
            
        # Extremely high outpatient claim
        if claim['outpatient_amount'] > avg_outpatient_amount * 4:
            fraud_indicators += 1
            
        # High vehicle claim for minor severity (Section 3.2)
        if claim['vehicle_percent'] > 80 and claim['severity'] == 1:
            fraud_indicators += 1
            
        # Round numbers in claims (Section 3.2)
        if claim['round_numbers'] and claim['claim_premium_ratio'] > 20:
            fraud_indicators += 1
            
        # Calculate fraud probability based on indicators
        fraud_probability = min(0.05 + (0.15 * fraud_indicators), 0.95)
        
        # Determine if claim is fraudulent
        return random.random() < fraud_probability
    
    # Helper functions
    def random_number(min_val, max_val):
        """Generate a random number in the specified range."""
        return random.uniform(min_val, max_val)
    
    def random_int(min_val, max_val):
        """Generate a random integer in the specified range."""
        return random.randint(min_val, max_val)
    
    def random_choice(options, weights=None):
        """Choose a random item from options with optional weights."""
        return random.choices(options, weights=weights, k=1)[0]
    
    def random_bool(prob_true=0.5):
        """Generate a random boolean with specified probability of True."""
        return random.random() < prob_true
    
    # Generate claims
    for i in range(num_claims):
        # Claim basics
        policy_number = 100000 + i
        hospital = random_choice(hospitals)
        patient_category = random_choice(patient_categories)
        is_gaza_patient = random_bool(0.05)  # 5% Gaza patients
        severity = random_choice(severity_levels)
        
        # Financial data
        policy_annual_premium = random_int(500, 3000)
        
        # Determine if inpatient or outpatient
        is_inpatient = random_bool(0.6)  # 60% inpatient, 40% outpatient
        
        # Set claim amounts
        inpatient_amount = random_number(avg_inpatient_amount * 0.5, avg_inpatient_amount * 1.5) if is_inpatient else 0
        outpatient_amount = random_number(avg_outpatient_amount * 0.5, avg_outpatient_amount * 1.5) if not is_inpatient else 0
        
        # Round amounts for some claims (potential fraud indicator)
        has_round_numbers = random_bool(0.2)  # 20% have suspicious round numbers
        if has_round_numbers:
            if is_inpatient:
                inpatient_amount = round(inpatient_amount / 10000) * 10000
            else:
                outpatient_amount = round(outpatient_amount / 10000) * 10000
        
        # Total claim amount
        total_claim_amount = inpatient_amount + outpatient_amount
        
        # Calculate claim components
        injury_percent = random_int(10, 60)
        property_percent = random_int(5, 30)
        vehicle_percent = 100 - injury_percent - property_percent
        
        injury_claim = round(total_claim_amount * (injury_percent / 100))
        property_claim = round(total_claim_amount * (property_percent / 100))
        vehicle_claim = total_claim_amount - injury_claim - property_claim
        
        # Calculate claim to premium ratio
        claim_premium_ratio = total_claim_amount / policy_annual_premium
        
        # Create claim record
        claim = {
            'policy_number': policy_number,
            'hospital': hospital,
            'patient_category': patient_category,
            'is_gaza_patient': 1 if is_gaza_patient else 0,
            'severity': severity,
            'policy_annual_premium': policy_annual_premium,
            'inpatient_amount': inpatient_amount,
            'outpatient_amount': outpatient_amount,
            'total_claim_amount': total_claim_amount,
            'injury_claim': injury_claim,
            'injury_percent': injury_percent,
            'property_claim': property_claim,
            'property_percent': property_percent,
            'vehicle_claim': vehicle_claim,
            'vehicle_percent': vehicle_percent,
            'claim_premium_ratio': claim_premium_ratio,
            'round_numbers': 1 if has_round_numbers else 0
        }
        
        # Determine if claim is fraudulent based on patterns
        claim['is_fraudulent'] = 1 if is_likely_fraudulent(claim) else 0
        
        # Add claim to dataset
        claims_data.append(claim)
    
    # Convert to DataFrame
    df = pd.DataFrame(claims_data)
    
    # Calculate summary statistics
    fraud_count = df['is_fraudulent'].sum()
    fraud_percentage = (fraud_count / num_claims) * 100
    
    print(f"Generated {num_claims} synthetic claims")
    print(f"{fraud_count} fraudulent claims ({fraud_percentage:.2f}%)")
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")
    
    return df

# Create directories if they don't exist
import os
os.makedirs('data', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/models', exist_ok=True)

# Generate the synthetic data
generate_synthetic_dataset('synthetic_claims.csv', num_claims=1000, seed=42)


# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import random
import os

# Create directories
os.makedirs('output', exist_ok=True)
os.makedirs('output/plots', exist_ok=True)

# Generate synthetic data
def generate_synthetic_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Generating {n_samples} synthetic claims...")
    
    # Create basic features
    claims = []
    
    for i in range(n_samples):
        # Basic data
        policy_number = 100000 + i
        hospital = f"Qatar Hospital {random.randint(1, 12)}"
        patient_category = random.choice(['GCC Resident', 'GCC Visitor', 'Resident', 'Visitor', 'Baby', 'Handicapped'])
        is_gaza_patient = 1 if random.random() < 0.05 else 0
        severity = random.choice([1, 2, 3])
        
        # Financial data
        policy_premium = random.randint(500, 3000)
        is_inpatient = random.random() < 0.6
        
        inpatient_amount = random.uniform(100000, 300000) if is_inpatient else 0
        outpatient_amount = random.uniform(3500000, 10500000) if not is_inpatient else 0
        
        # Round numbers for potential fraud
        has_round_numbers = 1 if random.random() < 0.2 else 0
        if has_round_numbers:
            if is_inpatient:
                inpatient_amount = round(inpatient_amount / 10000) * 10000
            else:
                outpatient_amount = round(outpatient_amount / 10000) * 10000
        
        total_amount = inpatient_amount + outpatient_amount
        
        # Claim components
        injury_percent = random.randint(10, 60)
        property_percent = random.randint(5, 30)
        vehicle_percent = 100 - injury_percent - property_percent
        
        injury_claim = round(total_amount * (injury_percent / 100))
        property_claim = round(total_amount * (property_percent / 100))
        vehicle_claim = total_amount - injury_claim - property_claim
        
        # Claim ratio
        claim_premium_ratio = total_amount / policy_premium
        
        # Create claim
        claim = {
            'policy_number': policy_number,
            'hospital': hospital,
            'patient_category': patient_category,
            'is_gaza_patient': is_gaza_patient,
            'severity': severity,
            'policy_annual_premium': policy_premium,
            'inpatient_amount': inpatient_amount,
            'outpatient_amount': outpatient_amount,
            'total_claim_amount': total_amount,
            'injury_claim': injury_claim,
            'injury_percent': injury_percent,
            'property_claim': property_claim,
            'property_percent': property_percent,
            'vehicle_claim': vehicle_claim,
            'vehicle_percent': vehicle_percent,
            'claim_premium_ratio': claim_premium_ratio,
            'round_numbers': has_round_numbers
        }
        
        # Determine fraud based on patterns
        fraud_probability = 0.05  # Base probability
        
        # Add probability for each fraud indicator
        if claim_premium_ratio > 50:
            fraud_probability += 0.15
        if severity == 3 and inpatient_amount > 400000:
            fraud_probability += 0.15
        if injury_percent > 40:
            fraud_probability += 0.15
        if claim_premium_ratio > 30 and patient_category == 'Visitor':
            fraud_probability += 0.15
        if vehicle_percent > 80 and severity == 1:
            fraud_probability += 0.15
        if has_round_numbers and claim_premium_ratio > 20:
            fraud_probability += 0.15
            
        # Cap probability
        fraud_probability = min(fraud_probability, 0.95)
        
        # Assign fraud label
        claim['is_fraudulent'] = 1 if random.random() < fraud_probability else 0
        
        claims.append(claim)
    
    # Convert to DataFrame
    df = pd.DataFrame(claims)
    
    # Calculate fraud statistics
    fraud_count = df['is_fraudulent'].sum()
    fraud_percentage = (fraud_count / n_samples) * 100
    
    print(f"Generated {n_samples} claims with {fraud_count} fraudulent claims ({fraud_percentage:.2f}%)")
    
    # Save to CSV
    df.to_csv('synthetic_claims.csv', index=False)
    print("Data saved to synthetic_claims.csv")
    
    return df

# Function to build and evaluate a Random Forest model
def evaluate_fraud_detection(df):
    print("\nEvaluating fraud detection models...")
    
    # Prepare data
    X = df.drop(['is_fraudulent', 'policy_number'], axis=1)
    X = pd.get_dummies(X, columns=['hospital', 'patient_category'], drop_first=True)
    y = df['is_fraudulent']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test_scaled)
    y_prob = rf.predict_proba(X_test_scaled)[:, 1]
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('output/plots/roc_curve.png')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('output/plots/confusion_matrix.png')
    plt.close()
    
    # Plot feature importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(10), importances[indices[:10]])
    plt.xticks(range(10), X.columns[indices[:10]], rotation=45)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Top 10 Feature Importances')
    plt.tight_layout()
    plt.savefig('output/plots/feature_importance.png')
    plt.close()
    
    print(f"\nResults saved to output/plots/")
    print(f"Random Forest AUC: {roc_auc:.4f}")
    
    return roc_auc

# Run the full process
df = generate_synthetic_data(n_samples=1000)
auc_score = evaluate_fraud_detection(df)

print(f"\nFraud detection evaluation complete with AUC score: {auc_score:.4f}")


# In[ ]:




