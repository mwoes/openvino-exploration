const { createServer } = require("http");
const { Server } = require("socket.io");
const express = require("express");
const util = require('util');
const jimp = require('jimp');

const app = express();
const httpServer = createServer(app);

const io = new Server(httpServer);

var processing = false;

io.on('connection', (socket) => {
  socket.on('image', (data) => {
    if(data.image && !processing) {
      var img = data.buffer;
      main(img).then(() => {processing = false}).catch();
    }
  });
});

const { ageGender } = require('./ageGender.js');
const { emotion } = require('./emotion.js');
const { faceD, normImg } = require('./detector.js');
const { getLandmarks } = require('./landmarks.js');
const { headposeR } = require('./headpose.js');

var faces = [];

async function main(image) {
  processing = true;
  faceNum = [];
  tmpImg = image;
  faceNum = await faceD(image);
  if(faceNum.results.length > 0) {
    for (var i=0; i< faceNum.results.length; i++) {
      var ageG = await ageGender(faceNum.results[i]);
      var emo = await emotion(faceNum.results[i]);
      var landmarks = await getLandmarks(faceNum.results[i]);
      var pose = await headposeR(faceNum.results[i]);
      var cons = {
        ts: Date.now(),
        dims: faceNum.results[i].dims,
        age: ageG.age,
        gender: ageG.gen,
        emo: emo.emotion,
        landmarks: landmarks,
        pose: pose
      };
      faces.push(cons);
      // top line
	  faceNum.img.scan(cons.dims.x, cons.dims.y, cons.dims.w, 2, fillColor(0xFF0000FF));
      // bottom line
	  faceNum.img.scan(cons.dims.x, cons.dims.y+cons.dims.h, cons.dims.w, 2, fillColor(0xFF0000FF));
      // left line
	  faceNum.img.scan(cons.dims.x, cons.dims.y, 2, cons.dims.h, fillColor(0xFF0000FF));
      // right line
	  faceNum.img.scan(cons.dims.x+cons.dims.w, cons.dims.y, 2, cons.dims.h, fillColor(0xFF0000FF));
      // write info
      const font = await jimp.loadFont(jimp.FONT_SANS_32_WHITE);

      faceNum.img.print(font, cons.dims.x, cons.dims.y+cons.dims.h+2, {text: "AGE: " + cons.age + " " + cons.gender});
      faceNum.img.print(font, cons.dims.x, cons.dims.y+cons.dims.h+32, {text: "MOOD: " + cons.emo});

//	  faceNum.img.write('./outputs/rects.jpg');

  }
    console.clear();
    console.log(util.inspect(faces, {showHidden: false, depth: null}))
    faces = [];
    return 1;
  } else {
    console.clear();
    console.log('NO-AUDIENCE');
    return 1;
  }
}

function fillColor(color) {
  return function (x, y, offset) {
    this.bitmap.data.writeUInt32BE(color, offset, true);
  }
};

httpServer.listen(3030);
console.log("STARTED");
