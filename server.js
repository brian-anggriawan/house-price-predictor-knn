const tf = require("@tensorflow/tfjs-node");
const express = require("express");
const path = require("path");
const loadCSV = require("./load-csv");

const app = express();
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Load & prepare training data once at startup
let { features, labels } = loadCSV("kc_house_data.csv", {
  shuffle: true,
  splitTest: 10,
  dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
  labelColumns: ["price"],
});

const featuresTensor = tf.tensor(features);
const labelsTensor = tf.tensor(labels);

const K = 10;

function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);
  const scaledPredictionPoint = predictionPoint
    .sub(mean)
    .div(variance.pow(0.5));

  const sorted = features
    .sub(mean)
    .div(variance.pow(0.5))
    .sub(scaledPredictionPoint)
    .pow(2)
    .sum(1)
    .pow(0.5)
    .expandDims(1)
    .concat(labels, 1)
    .unstack()
    .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
    .slice(0, k);

  const prices = sorted.map((pair) => pair.arraySync()[1]);
  const prediction = prices.reduce((a, b) => a + b, 0) / k;
  const priceMin = Math.min(...prices);
  const priceMax = Math.max(...prices);

  mean.dispose();
  variance.dispose();
  scaledPredictionPoint.dispose();
  sorted.forEach((t) => t.dispose());

  return { prediction, priceMin, priceMax };
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

  const predictionPoint = tf.tensor([lat, long, sqft_lot, sqft_living]);
  const { prediction, priceMin, priceMax } = knn(
    featuresTensor,
    labelsTensor,
    predictionPoint,
    K,
  );
  predictionPoint.dispose();

  res.json({
    prediction: Math.round(prediction),
    priceMin: Math.round(priceMin),
    priceMax: Math.round(priceMax),
    k: K,
    features: 4,
    algorithm: "K-Nearest Neighbors",
  });
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`Server berjalan di http://localhost:${PORT}`);
});
