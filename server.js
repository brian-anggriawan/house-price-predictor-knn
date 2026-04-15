const express = require("express");
const path = require("path");
const loadCSV = require("./load-csv");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

const K = 10;

// Load and prepare training data once at startup
const { features, labels } = loadCSV("kc_house_data.csv", {
  shuffle: true,
  splitTest: 10,
  dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
  labelColumns: ["price"],
});

// Compute per-column mean and std for standardization
const numCols = features[0].length;
const means = new Array(numCols).fill(0);
const stds = new Array(numCols).fill(0);

for (const row of features) {
  for (let i = 0; i < numCols; i++) means[i] += row[i];
}
for (let i = 0; i < numCols; i++) means[i] /= features.length;

for (const row of features) {
  for (let i = 0; i < numCols; i++) stds[i] += Math.pow(row[i] - means[i], 2);
}
for (let i = 0; i < numCols; i++)
  stds[i] = Math.sqrt(stds[i] / features.length);

function knn(features, labels, predPoint, k) {
  const distances = features.map((row, i) => {
    const dist = Math.sqrt(
      row.reduce((sum, val, j) => {
        const normVal = (val - means[j]) / stds[j];
        const normPred = (predPoint[j] - means[j]) / stds[j];
        return sum + Math.pow(normVal - normPred, 2);
      }, 0),
    );
    return { dist, price: labels[i][0] };
  });

  distances.sort((a, b) => a.dist - b.dist);
  const nearest = distances.slice(0, k);
  const prices = nearest.map((d) => d.price);

  return {
    prediction: prices.reduce((a, b) => a + b, 0) / k,
    priceMin: Math.min(...prices),
    priceMax: Math.max(...prices),
  };
}

app.post("/predict", (req, res) => {
  const { lat, long, sqft_lot, sqft_living } = req.body;

  if (
    lat === undefined ||
    long === undefined ||
    sqft_lot === undefined ||
    sqft_living === undefined
  ) {
    return res.status(400).json({ error: "All fields are required." });
  }

  const { prediction, priceMin, priceMax } = knn(
    features,
    labels,
    [lat, long, sqft_lot, sqft_living],
    K,
  );

  res.json({
    prediction: Math.round(prediction),
    priceMin: Math.round(priceMin),
    priceMax: Math.round(priceMax),
    k: K,
    features: 4,
    algorithm: "K-Nearest Neighbors",
  });
});

const PORT = process.env.PORT || 3000;
if (require.main === module) {
  app.listen(PORT, () => {
    console.log(`Server berjalan di http://localhost:${PORT}`);
  });
}

module.exports = app;
