const canvas = document.getElementById("canvas");
const rawImage = document.getElementById("canvasimg");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, 280, 280);
const saveButton = document.getElementById("sb");
const clearButton = document.getElementById("cb");
let pos = { x: 0, y: 0 };

const setPosition = (e) => {
    pos.x = e.clientX - 100; //sets pos (used for cursor canvas curor placement) to click position minnus 100, 100
    pos.y = e.clientY - 100;
};

const draw = (e) => {
    if (e.buttons != 1) return;
    ctx.beginPath(); // tells the context a new path will be drawn (doesn't actually start drawing anything)
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.strokeStyle = "white";
    ctx.moveTo(pos.x, pos.y); // places the context "cursor" on pos, so subsequent lineTo will draw something starting here
    setPosition(e); // changes pos from the last pos to current mouse pos (last pos can either be last mouse movement or triggered by mouseenter or mousedown)
    ctx.lineTo(pos.x, pos.y); // draws line to pos
    ctx.stroke(); // actually visualizes the path drawn onto the cavas
    rawImage.src = canvas.toDataURL("image/png"); // places the canvas data into an image that is identical and lays on top of the canvas.
    // purpose of this is that the tf.browser.fromPixels function needs an image not a canvas to work with.
    // also, you can't assign the src in save() because the DOM needs to update before you then pull the pixels out of the image, and it doesn't have time to do so inside one function call.
};

const erase = () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
    // paints the canvas black, and therfore earases all paths. This erase() should also clear the rawImage.src but whatever
};

const save = async () => {
    const imgData = ctx.getImageData(0, 0, 280, 280);
    const raw = tf.browser.fromPixels(rawImage, 1); // gets pixels from image opy of canvas
    const resized = tf.image.resizeBilinear(raw, [28, 28]); // cuts down on pixels 10 fold (280 x 280 x 1 to 28 x 28 x 1)
    const tensor = resized.expandDims(0); // adds a dimension at the front to indicate the tensor only has one datapoint to predict (1 x 28 x 28 x 1)
    const response = await axios({
        method: "POST",
        url: "/predict",
        data: { tensor },
    });
    console.log(response.data);
    alert(response.data);
};

canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mousedown", setPosition);
canvas.addEventListener("mouseenter", setPosition);
saveButton.addEventListener("click", save);
clearButton.addEventListener("click", erase);
