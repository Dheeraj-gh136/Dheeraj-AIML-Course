import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

print("🚀 WEEK 2: BOSTON HOUSING PRICE PREDICTION")

class HousingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_data(self):
        """Auto-downloads Boston dataset from OpenML.org"""
        print("📥 Downloading from OpenML.org...")
        housing = fetch_openml(name='boston', version=1, as_frame=True, parser='auto')
        df = housing.frame
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna()
        df = df.rename(columns={'target': 'MEDV'})
        print(f"✅ LOADED: {df.shape[0]} houses, {df.shape[1]-1} features")
        return df
    
    def explore_data(self, df):
        """Week 2 EDA - 4 beautiful charts"""
        plt.figure(figsize=(15, 12))
        
        # Chart 1: Price distribution
        plt.subplot(2,2,1)
        plt.hist(df['MEDV'], bins=30, color='skyblue', edgecolor='black')
        plt.title('🏠 House Price Distribution ($1000s)')
        plt.xlabel('Price')
        
        # Chart 2: Top correlations
        plt.subplot(2,2,2)
        corr = df.corr()['MEDV'].drop('MEDV').sort_values(ascending=False)[:5]
        plt.barh(corr.index, corr.values, color='coral')
        plt.title('📈 Top 5 Features vs Price')
        plt.xlabel('Correlation')
        
        # Chart 3: Rooms vs Price
        plt.subplot(2,2,3)
        plt.scatter(df['RM'], df['MEDV'], alpha=0.6, color='green')
        plt.title('🛏️ Average Rooms vs Price')
        plt.xlabel('Rooms')
        
        # Chart 4: Lower status vs Price
        plt.subplot(2,2,4)
        plt.scatter(df['LSTAT'], df['MEDV'], alpha=0.6, color='red')
        plt.title('👥 % Lower Status Population vs Price')
        plt.xlabel('% Lower Status')
        
        plt.tight_layout()
        plt.savefig('week2_eda_charts.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Charts saved: week2_eda_charts.png")
    
    def preprocess(self, df):
        """Week 2: Feature selection & preparation"""
        features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                   'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        X = df[features]
        y = df['MEDV']
        self.feature_names = features
        print(f"✅ Features ready: {X.shape}")
        return X, y
    
    def train(self, X, y):
        """Week 2: Train Random Forest model"""
        print("\n🤖 TRAINING MODEL...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\n🎯 WEEK 2 RESULTS:")
        print(f"   R² Score:   {r2:.3f}")
        print(f"   RMSE:       ${np.sqrt(mse)*1000:,.0f}")
        print(f"   MSE:        {mse:.3f}")
        
        # Feature importance
        importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🏆 TOP 5 FEATURES:")
        print(importance.head().to_string(index=False))
        
        return X_test_scaled, y_test, y_pred
    
    def predict_price(self, features):
        """Week 2: Predict new house price"""
        if self.model is None:
            raise ValueError("❌ Train model first!")
        features_scaled = self.scaler.transform([features])
        return self.model.predict(features_scaled)[0]

# COMPLETE WEEK 2 DEMO
if __name__ == "__main__":
    predictor = HousingPredictor()
    df = predictor.load_data()
    predictor.explore_data(df)
    X, y = predictor.preprocess(df)
    predictor.train(X, y)
    
    # Week 2 sample prediction
    sample_house = [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.90, 4.98]
    price = predictor.predict_price(sample_house)
    print(f"\n🔮 Week 2 Prediction: ${price*1000:,.0f}")
