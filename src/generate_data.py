import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

neighborhoods = ['Downtown', 'Suburbs', 'Countryside', 'Waterfront', 'Historic']
neighborhood = np.random.choice(neighborhoods, n)

area = np.random.randint(500, 5000, n)
bedrooms = np.random.randint(1, 7, n)
bathrooms = np.random.randint(1, 5, n)
floors = np.random.randint(1, 4, n)
garage = np.random.randint(0, 3, n)
year_built = np.random.randint(1950, 2023, n)
age = 2024 - year_built
has_pool = np.random.randint(0, 2, n)
has_garden = np.random.randint(0, 2, n)
distance_to_center = np.random.uniform(0.5, 30, n)

neighborhood_map = {
    'Downtown': 1.4, 'Waterfront': 1.5, 'Historic': 1.2,
    'Suburbs': 1.0, 'Countryside': 0.8
}
neigh_factor = np.array([neighborhood_map[n_] for n_ in neighborhood])

price = (
    area * 120
    + bedrooms * 8000
    + bathrooms * 6000
    + floors * 5000
    + garage * 10000
    + has_pool * 20000
    + has_garden * 10000
    - age * 500
    - distance_to_center * 3000
) * neigh_factor

noise = np.random.normal(0, 15000, n)
price = np.abs(price + noise)

# Add some missing values
idx_missing = np.random.choice(n, 40, replace=False)
bathrooms = bathrooms.astype(float)
bathrooms[idx_missing[:20]] = np.nan
garage = garage.astype(float)
garage[idx_missing[20:]] = np.nan

df = pd.DataFrame({
    'area_sqft': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'floors': floors,
    'garage_spaces': garage,
    'year_built': year_built,
    'house_age': age,
    'has_pool': has_pool,
    'has_garden': has_garden,
    'distance_to_center_km': distance_to_center,
    'neighborhood': neighborhood,
    'price': price.astype(int)
})

df.to_csv('/home/claude/house_price_prediction/data/house_prices.csv', index=False)
print(f"Dataset created: {df.shape}")
print(df.head())
