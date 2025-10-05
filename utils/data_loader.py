mport pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import requests
import urllib.parse
import io

class ExoplanetDataLoader:
    """Utility class for loading and preprocessing exoplanet data"""
    
    def __init__(self, mission='kepler'):
        self.mission = mission
        self.tap_urls = {
            'kepler': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync',
            'k2': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync',
            'tess': 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync'
        }
        
        self.table_names = {
            'kepler': 'cumulative',
            'k2': 'k2pandc',
            'tess': 'toi'
        }
    
    def load_data(self, n_features=20):
        """Load data for the selected mission"""
        print(f"Loading {self.mission.upper()} data...")
        
        # Get data from TAP
        query = f"SELECT * FROM {self.table_names[self.mission]}"
        encoded_query = urllib.parse.quote(query)
        tap_url = f"{self.tap_urls[self.mission]}?query={encoded_query}&format=csv"
        
        response = requests.get(tap_url)
        data = pd.read_csv(io.StringIO(response.text))
        
        print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns")
        return data
      
    def preprocess_k2_data(self, data, n_features=20, correlation_threshold=0.8, missing_threshold=50):
        """Preprocess data for machine learning analysis"""
        print(f"Preprocessing {self.mission.upper()} data...")
        
        # STEP 1: Remove REFUTED [PLANET]
        data_clean = data[data['disposition'] != 'REFUTED [PLANET]'].copy()
        print(f"After removing REFUTED [PLANET]: {data_clean.shape[0]} rows")
        
        # STEP 2: Create target variable FIRST
        y_ternary = data_clean['disposition'].map({
            'CANDIDATE': 0,           
            'FALSE POSITIVE [CANDIDATE]': 1, 
            'CONFIRMED': 2            
        })
        
        # Add target back to dataframe for analysis
      
        data_with_target = data_clean.copy()
        data_with_target['target'] = y_ternary
        
        # STEP 3: Drop columns with high missing values
      
        missing_percentages = (data_with_target.isnull().sum() / len(data_with_target)) * 100
        high_missing = missing_percentages[missing_percentages > missing_threshold]
        columns_to_drop = missing_percentages[missing_percentages > missing_threshold].index.tolist()
        
        print(f"Columns with >{missing_threshold}% missing values: {len(columns_to_drop)}")
      
        if columns_to_drop:
            print(f"Dropping columns: {columns_to_drop}")
        
        data_after_missing = data_with_target.drop(columns=columns_to_drop)
      
        print(f"After dropping high missing columns: {data_after_missing.shape[1]} features")
        
        # STEP 4: Remove highly correlated features
      
        data_final, removed_features = self._remove_highly_correlated_features(
            data_after_missing, threshold=correlation_threshold
        )
        print(f"After removing correlated features: {data_final.shape[1]} features")
        print(f"Removed {len(removed_features)} correlated features")
        
        # STEP 5: Get top features based on correlation with target
      
        target_corr = self._calculate_feature_importance('target', 'disposition', data_final)
        top_features = target_corr.head(n_features).index.tolist()
        
        # STEP 6: Prepare features (exclude target column)
        feature_columns = [col for col in top_features if col != 'target']
        numeric_features = data_final[feature_columns].select_dtypes(include=[np.number])
        
        # STEP 7: Impute missing values
        imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            random_state=42
        )
        
        X_imputed = imputer.fit_transform(numeric_features)
        X_imputed_df = pd.DataFrame(X_imputed, columns=numeric_features.columns)
        
        # Get final target (remove any rows that might have been dropped)
        y_final = data_final['target']
        
        return X_imputed_df, y_final, top_features, imputer, target_corr, removed_features, columns_to_drop

    def _calculate_feature_importance(self, disposition_key,data):
        """Calculate feature importance using correlation with target"""
        # Convert target to numeric for correlation
        y_numeric = pd.get_dummies(data[disposition_key]).iloc[:, 0]
        
        # Calculate correlation
        numeric_data = data.select_dtypes(include=[np.number])
        target_corr = numeric_data.corrwith(y_numeric).abs().sort_values(ascending=False)
        
        return target_corr

    def _remove_highly_correlated_features(self, df, threshold=0.8):
        """Remove highly correlated features"""
        # Compute correlation matrix
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        
        # Decide which features to remove
        features_to_remove = set()
        for feat1, feat2 in high_corr_pairs:
            if df[feat1].isnull().sum() <= df[feat2].isnull().sum():
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
        
        return df.drop(columns=list(features_to_remove)), list(features_to_remove)
