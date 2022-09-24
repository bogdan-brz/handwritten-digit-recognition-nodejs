const express = require("express");
const path = require("path");
const app = express();
const port = 3000;
const { getData } = require("./data");
const { initModel, predict, getIsTrained } = require("./model");

getData((_xTrain, _yTrain, _xTest, _yTest) => {
    initModel(_xTrain, _yTrain, _xTest, _yTest);
});

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "./"));

app.use(express.static("./"));

app.get("/", (req, res) => {
    res.render(__dirname + "/home.ejs");
});

app.listen(port, () => {
    console.log(`Example app listening on port ${port}`);
});

app.post("/predict", (req, res) => {
    console.log("in predict");
    if (!getIsTrained())
        return res.send("wait, the model hasn't finished training");
    console.log(req.body);
    const prediction = predict(req.body);
    res.send(prediction);
});
