const express = require("express");
const { getData } = require("./data");
const app = express();
const port = 3000;
const { model, initModel } = require("./model");

let xTrain = [];
let yTrain = [];
let xTest = [];
let yTest = [];

getData((_xTrain, _yTrain, _xTest, _yTest) => {
    xTrain = _xTrain;
    yTrain = _yTrain;
    xTest = _xTest;
    yTest = _yTest;
    initModel(_xTrain, _yTrain, _xTest, _yTest);
});

app.get("/", (req, res) => {
    res.sendFile(__dirname + "/home.html");
});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`);
});

app.get("/predict", (req, res) => {
    const prediction = model.predict(xTest[0]);
    res.send(prediction);
});
