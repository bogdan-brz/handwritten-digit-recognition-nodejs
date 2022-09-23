const fs = require("fs");
const { parse } = require("csv-parse");

const getData = async (callback) => {
    let train = [];
    let test = [];
    await new Promise((resolve, reject) => {
        fs.createReadStream("./data/mnist_train.csv")
            .pipe(parse({ delimiter: ",", from_line: 2 }))
            .on("data", function (row) {
                train.push(row);
            })
            .on("end", function () {
                resolve();
            })
            .on("error", function (error) {
                console.log(error.message);
                reject();
            });
    });
    await new Promise((resolve, reject) => {
        fs.createReadStream("./data/mnist_test.csv")
            .pipe(parse({ delimiter: ",", from_line: 2 }))
            .on("data", function (row) {
                test.push(row);
            })
            .on("end", function () {
                resolve();
            })
            .on("error", function (error) {
                console.log(error.message);
                reject();
            });
    });
    const xTrain = train
        .map((array) => array.slice(1).map((el) => el / 255.0))
        .slice(0, 5500);
    const yTrain = train
        .map((array) => {
            const oneHot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            oneHot[array[0]] = 1;
            return oneHot;
        })
        .slice(0, 5500);
    const xTest = test
        .map((array) => array.slice(1).map((el) => el / 255.0))
        .slice(0, 1000);
    const yTest = test
        .map((array) => {
            const oneHot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
            oneHot[array[0]] = 1;
            return oneHot;
        })
        .slice(0, 1000);

    console.log("data loaded");
    callback(xTrain, yTrain, xTest, yTest);
};

module.exports = { getData };
