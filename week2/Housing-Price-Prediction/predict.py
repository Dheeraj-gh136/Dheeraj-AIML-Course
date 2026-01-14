"""Week 2: Load model and predict prices"""
import joblib

predictor = joblib.load('week2_housing_model.pkl')
print("✅ Week 2 model loaded!")

# Feature order: CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT
test_houses = {
    "Budget Home": [0.02731, 0.0, 7.07, 0, 0.469, 6.421, 78.9, 4.9671, 2, 242, 17.8, 396.90, 9.14],
    "Mid Range":   [0.123, 10.0, 2.45, 1, 0.403, 7.890, 32.1, 5.678, 1, 300, 16.2, 391.20, 12.34],
    "Luxury":      [0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296, 15.3, 396.90, 4.98]
}

print("\n🏠 WEEK 2 PREDICTIONS:")
for name, features in test_houses.items():
    price = predictor.predict_price(features)
    print(f"   {name}: ${price*1000:,.0f}")
