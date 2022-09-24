const tf = require("@tensorflow/tfjs-node");
let isTrained = false;

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
    console.log(typeof xTrain);
    xTrain = xTrain.reshape([5500, 28, 28, 1]);
    xTest = xTest.reshape([1000, 28, 28, 1]);

    const onEpochEnd = (batch, logs) => {
        console.log(batch);
        console.log(logs);
    };

    console.log("training model");
    model
        .fit(xTrain, yTrain, {
            batchSize: BATCH_SIZE,
            validationData: [xTest, yTest],
            epochs: 20,
            shuffle: true,
            callbacks: { onEpochEnd },
        })
        .then(() => {
            isTrained = true;
            console.log("model trained");
        });
};

const predict = (imgData) => {
    const tensor = tf.tensor([imgData], [1, 28, 28, 1]); // convert the image data received in form of array into a tensor of the right shape the imgData is in [],
    // becuase sent over http an arry of length 1 becomes just the sole element of itself
    const prediction = model.predict(tensor);
    const pIndex = tf.argMax(prediction, 1).arraySync()[0]; // converts the one-hot encoded label into an array (.data() => object; .array => array; sync => synchronously)
    return pIndex;
};

const getIsTrained = () => {
    return isTrained;
};

module.exports = { initModel, predict, getIsTrained };
