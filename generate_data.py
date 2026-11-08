import pandas as pd
import numpy as np

np.random.seed(42)
HOUSES_PER_PLACE = 20

# (state, district): (sample price per sq ft, [3 places])
# The 3 places are in this order: City Center, Suburb, Outskirts
DISTRICTS = {
    ("Karnataka", "Bengaluru"): (9000, ["Indiranagar", "Whitefield", "Electronic City"]),
    ("Karnataka", "Mysuru"): (4500, ["Saraswathipuram", "Vijayanagar", "Hebbal"]),
    ("Karnataka", "Mangaluru"): (5000, ["Hampankatta", "Kankanady", "Surathkal"]),
    ("Tamil Nadu", "Chennai"): (8500, ["T Nagar", "Velachery", "Tambaram"]),
    ("Tamil Nadu", "Coimbatore"): (5000, ["RS Puram", "Saravanampatti", "Singanallur"]),
    ("Kerala", "Kochi"): (6000, ["Ernakulam South", "Edappally", "Kakkanad"]),
    ("Kerala", "Thiruvananthapuram"): (5200, ["Pattom", "Kazhakkoottam", "Neyyattinkara"]),
    ("Maharashtra", "Mumbai"): (18000, ["Bandra", "Thane", "Virar"]),
    ("Maharashtra", "Pune"): (8000, ["Koregaon Park", "Hinjewadi", "Wagholi"]),
    ("Telangana", "Hyderabad"): (8000, ["Banjara Hills", "Gachibowli", "Shamshabad"]),
    ("Andhra Pradesh", "Visakhapatnam"): (4800, ["MVP Colony", "Gajuwaka", "Anandapuram"]),
    ("Delhi", "New Delhi"): (12000, ["Connaught Place", "Dwarka", "Narela"]),
    ("West Bengal", "Kolkata"): (6500, ["Park Street", "Salt Lake", "Rajarhat"]),
    ("Gujarat", "Ahmedabad"): (5500, ["Navrangpura", "Bopal", "Naroda"]),
}

# zone name, price factor
ZONES = [
    ("City Center", 1.00),
    ("Suburb", 0.75),
    ("Outskirts", 0.55),
]

sizes = [(20, 30), (30, 40), (30, 50), (40, 60), (50, 80)]
directions = ["North", "South", "East", "West"]

rows = []
house_id = 1
for (state, district), (rate, places) in DISTRICTS.items():
    for place, (zone, factor) in zip(places, ZONES):
        for _ in range(HOUSES_PER_PLACE):
            w, l = sizes[np.random.randint(len(sizes))]
            sqft = w * l
            beds = int(np.random.randint(1, 6))
            baths = int(min(beds, np.random.randint(1, 4)))

            price = rate * factor * sqft + beds * 150000
            price = price * np.random.uniform(0.95, 1.05)
            price = int(round(price / 10000) * 10000)

            rows.append({
                "house_id": house_id,
                "state": state,
                "district": district,
                "place": place,
                "location": zone,
                "area": directions[np.random.randint(4)],
                "dimensions": f"{w} x {l} ft",
                "square_feet": sqft,
                "bedrooms": beds,
                "bhk": f"{beds} BHK",
                "bathrooms": baths,
                "price": price,
            })
            house_id += 1

df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
print("dataset.csv created with", len(df), "houses,",
      df["state"].nunique(), "states,", df["district"].nunique(), "districts")