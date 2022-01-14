const { createServer } = require("http");
const { Server } = require("socket.io");
const express = require("express");
const util = require('util');

const app = express();
const httpServer = createServer(app);

const io = new Server(httpServer);

var processing = false;

const jimp = require('jimp');

const { ageGender } = require('./ageGender.js');
const { emotion } = require('./emotion.js');
const { faceD } = require('./detector.js');
const { getLandmarks } = require('./recognizer.js');

var faces = [];

const option_definitions = [
  {
    name: 'image',
    alias: 'i',
    type: String,
    description: 'Required. Path to an image file.'
  }
];

const commandLineArgs = require('command-line-args');
const commandLineUsage = require('command-line-usage');

const options = commandLineArgs(option_definitions);

if (options.help || !options.image) {
    const usage = commandLineUsage([
      {
        header: 'Image Inference Tester',
        content:
            'Feed me an image file to get immediate results.'
      },
      {header: 'Options', optionList: option_definitions}
    ]);
    console.log(usage);
    process.exit(0);
  }

async function main() {
  var image = await jimp.read(options.image);
  processing = true;
  faceNum = [];
  faceNum = await faceD(image);
  if(faceNum.length > 0) {
  for(var i=0; i<faceNum.length; i++) {
      var ageG = await ageGender(faceNum[i]);
      var emo = await emotion(faceNum[i]);
      var landmarks = await getLandmarks(faceNum[i]);
      var cons = {
        ts: Date.now(),
        dims: faceNum[i].dims,
        age: ageG.age,
        gender: ageG,
        emo: emo.emotion,
        landmarks: landmarks
      };
      faces.push(cons);
  }

//    console.clear();
    console.log(util.inspect(faces, {showHidden: false, depth: null}))
    faces = [];
    return 1;
  } else {
    console.clear();
    console.log('NO-AUDIENCE');
    return 1;
  }
}

main().then().catch();

//httpServer.listen(3030);
//console.log("STARTED");
