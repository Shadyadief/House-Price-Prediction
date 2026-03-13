"""
predict.py — Use trained model to predict house price
Usage: python src/predict.py
"""

import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def predict_price(area_sqft, bedrooms, bathrooms, floors, garage_spaces,
                  year_built, has_pool, has_garden, distance_to_center_km,
                  neighborhood):
    """Predict house price using trained Random Forest model."""

    neighborhoods = ['Countryside', 'Downtown', 'Historic', 'Suburbs', 'Waterfront']
    if neighborhood not in neighborhoods:
        raise ValueError(f"Neighborhood must be one of: {neighborhoods}")

    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    model = joblib.load(os.path.join(MODEL_DIR, 'random_forest.pkl'))

    house_age     = 2024 - year_built
    total_rooms   = bedrooms + bathrooms
    area_per_room = area_sqft / total_rooms if total_rooms > 0 else 0
    is_new_house  = int(house_age <= 10)
    luxury_score  = int(has_pool) + int(has_garden) + int(garage_spaces > 1)
    neigh_encoded = le.transform([neighborhood])[0]

    features = np.array([[
        area_sqft, bedrooms, bathrooms, floors, garage_spaces,
        house_age, has_pool, has_garden, distance_to_center_km,
        neigh_encoded, total_rooms, area_per_room, is_new_house, luxury_score
    ]])

    predicted_price = model.predict(features)[0]
    return predicted_price


if __name__ == "__main__":
    print("=" * 45)
    print("   HOUSE PRICE PREDICTION")
    print("=" * 45)

    sample_house = {
        'area_sqft':             2500,
        'bedrooms':              3,
        'bathrooms':             2,
        'floors':                2,
        'garage_spaces':         1,
        'year_built':            2010,
        'has_pool':              0,
        'has_garden':            1,
        'distance_to_center_km': 8.5,
        'neighborhood':          'Suburbs',
    }

    price = predict_price(**sample_house)

    print("\n🏠 House Features:")
    for k, v in sample_house.items():
        print(f"   {k:<30}: {v}")

    print(f"\n💰 Predicted Price: ${price:,.0f}")
    print("=" * 45)
