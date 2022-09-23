const tf = require("@tensorflow/tfjs");

const model = tf.sequential();

model.add(
    tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 8,
        activation: "relu",
    })
);
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.conv2d({ filters: 16, kernelSize: 3, activation: "relu" }));
model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));
model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
});

const initModel = async (_xTrain, _yTrain, _xTest, _yTest) => {
    const BATCH_SIZE = 512;

    let xTrain = tf.tensor(_xTrain);
    let yTrain = tf.tensor(_yTrain);
    let xTest = tf.tensor(_xTest);
    let yTest = tf.tensor(_yTest);
    xTrain = xTrain.reshape([5500, 28, 28, 1]);
    xTest = xTest.reshape([1000, 28, 28, 1]);

    const onBatchEnd = (batch, logs) => {
        console.log("==========================");
        console.log(batch);
        console.log(logs.accuracy);
    };

    console.log("training model");
    model.fit(xTrain, yTrain, {
        batchSize: BATCH_SIZE,
        validationData: [xTest, yTest],
        epochs: 20,
        shuffle: true,
        callbacks: { onBatchEnd: onBatchEnd },
    });
    console.log("model trained");
};

module.exports = { model, initModel };
