"""Week 2: Train and save model"""
from housing_prediction import HousingPredictor
import joblib

print("🏋️  WEEK 2: Training model...")
predictor = HousingPredictor()
df = predictor.load_data()
X, y = predictor.preprocess(df)
predictor.train(X, y)
joblib.dump(predictor, 'week2_housing_model.pkl')
print("✅ SAVED: week2_housing_model.pkl")
