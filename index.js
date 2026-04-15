const tf = require("@tensorflow/tfjs-node");
const loadCSV = require("./load-csv");

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
  },
);

function knn(features, labels, predictionPoint, k) {
  return tf.tidy(() => {
    const { mean, variance } = tf.moments(features, 0);
    const scaledPredictionPoint = predictionPoint
      .sub(mean)
      .div(variance.pow(0.5));

    const distances = features
      .sub(mean)
      .div(variance.pow(0.5))
      .sub(scaledPredictionPoint)
      .pow(2)
      .sum(1)
      .pow(0.5)
      .expandDims(1)
      .concat(labels, 1);

    const sorted = distances
      .unstack()
      .sort((a, b) => a.arraySync()[0] - b.arraySync()[0])
      .slice(0, k);

    const sum = sorted.reduce((acc, pair) => acc + pair.arraySync()[1], 0);
    return sum / k;
  });
}

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, index) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  const error = (testLabels[index][0] - result) / testLabels[index][0];
  console.log(`Prediction: ${result}, Error: ${error * 100}%`);
});
