# House Price Predictor — KNN

A house price prediction web app using the **K-Nearest Neighbors (KNN)** algorithm, built with Node.js and Express.

🌐 **Live Demo:** [https://house-price-predictor-knn.vercel.app/](https://house-price-predictor-knn.vercel.app/)

---

## Features

- Predicts house price based on location and size
- KNN algorithm with feature standardization (z-score normalization)
- Trained on King County, WA housing dataset (21,000+ records)
- Responsive UI

## Input Features

| Feature | Description |
|---|---|
| Latitude | Geographic latitude of the house |
| Longitude | Geographic longitude of the house |
| Lot Size (sqft) | Total lot area in square feet |
| Living Area (sqft) | Interior living space in square feet |

## Tech Stack

- **Backend:** Node.js, Express
- **Algorithm:** K-Nearest Neighbors (pure JavaScript, K=10)
- **Dataset:** `kc_house_data.csv` (King County House Sales)
- **Deployment:** Vercel

## Run Locally

```bash
npm install
node server.js
```

Then open [http://localhost:3000](http://localhost:3000).
