const { createServer } = require("http");
const { Server } = require("socket.io");
const express = require("express");
const util = require('util');
const jimp = require('jimp');
const { Core, getVersion } = require('../../lib/inference-engine-node');

const app = express();
app.use(express.static('public'))
const httpServer = createServer(app);

const io = new Server(httpServer);

var processing = false;
var initialized = false;

var device_name = "CPU";

io.on('connection', (socket) => {
  socket.on('image', (data) => {
    if(data.image && !processing && initialized) {
      var img = data.buffer;
      main(img).then(() => {processing = false}).catch();
    }
  });
  socket.on('live', (data) => {
    io.emit('live', data);
  });
});

const { ageGender, ageGenEngine } = require('./ageGender.js');
const { emotion, emoEngine } = require('./emotion.js');
const { faceD, normImg, faceEngine } = require('./detector.js');
const { getLandmarks, landmarksEngine } = require('./landmarks.js');
const { headposeR, headposeEngine } = require('./headpose.js');
const { getIdentityRaw, identEngine } = require('./identify.js');


var faces = [];

async function main(image) {
  processing = true;
  faceNum = [];
  tmpImg = image;
  faceNum = await faceD(image);
  if(faceNum.results.length > 0) {
    for (var i=0; i< faceNum.results.length; i++) {
      var landmarks = await getLandmarks(faceNum.results[i]);
      var pose = await headposeR(faceNum.results[i]);
      var identity = await getIdentityRaw(faceNum.results[i], landmarks, pose);
      var ageG = await ageGender(faceNum.results[i]);
      var emo = await emotion(faceNum.results[i]);
      var cons = {
        ts: Date.now(),
        dims: faceNum.results[i].dims,
        age: ageG.age,
        gender: { gender: ageG.gen, raw: ageG.raw },
        emo: emo.emotion,
        landmarks: landmarks,
        pose: pose,
        identity: identity.vect,
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
      // nose-tip
      faceNum.img.scan(cons.landmarks.noseTip[0], cons.landmarks.noseTip[1], 8, 8, fillColor(0xFF0000FF));
      // eyes
      faceNum.img.scan(cons.landmarks.leftEye[0], cons.landmarks.leftEye[1], 8, 8, fillColor(0xFF0000FF));
      faceNum.img.scan(cons.landmarks.rightEye[0], cons.landmarks.rightEye[1], 8, 8, fillColor(0xFF0000FF));
      // lips
      faceNum.img.scan(cons.landmarks.leftLip[0], cons.landmarks.leftLip[1], 8, 8, fillColor(0xFF0000FF));
      faceNum.img.scan(cons.landmarks.rightLip[0], cons.landmarks.rightLip[1], 8, 8, fillColor(0xFF0000FF));
      // write info
      const font = await jimp.loadFont(jimp.FONT_SANS_32_WHITE);

      faceNum.img.print(font, cons.dims.x, cons.dims.y+cons.dims.h+2, {text: "AGE: " + cons.age + " " + cons.gender.gender});
      faceNum.img.print(font, cons.dims.x, cons.dims.y+cons.dims.h+32, {text: "MOOD: " + cons.emo});

//	  faceNum.img.write('./outputs/rects.jpg');
	  const rects = await faceNum.img.getBase64Async(jimp.MIME_JPEG)
      io.emit('rects', {image: true, data: rects, faces: {age: cons.age, gender: cons.gender.gender, emotion: cons.emo, dims: cons.dims}});

  }
//    console.clear();
//    console.log(util.inspect(faces, {showHidden: false, depth: null}))
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

(async () => {
	await faceEngine(device_name);
        await identEngine(device_name);
	await ageGenEngine(device_name);
	await emoEngine(device_name);
	await landmarksEngine(device_name);
	await headposeEngine(device_name);
	initialized = true;
	console.log("ENGINES INITIALIZED");
})();
httpServer.listen(3030);
console.log("STARTED");
