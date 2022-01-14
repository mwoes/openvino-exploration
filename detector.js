const { Core, getVersion } = require('../../lib/inference-engine-node');

const jimp = require('jimp');
const fs = require('fs').promises;
const { performance } = require('perf_hooks');

const {
  binPathFromXML,
  showInputOutputInfo
} = require('../common');

var core_face, model_face, bin_path_face, net_face, inputs_info_face, outputs_info_face, input_info_face, output_info_face, exec_net_face, input_dims_face, input_info_face_name, output_info_face_name;

async function faceEngine(device_name) {
	core_face = new Core();
	model_face = '/opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml';
	bin_path_face = binPathFromXML(model_face);
	net_face = await core_face.readNetwork(model_face, bin_path_face);
	inputs_info_face = net_face.getInputsInfo();
	outputs_info_face = net_face.getOutputsInfo();
	input_info_face = inputs_info_face[0];
	output_info_face = outputs_info_face[0];
	input_info_face.setLayout('nhwc');
	input_info_face.setPrecision('u8');
	exec_net_face = await core_face.loadNetwork(net_face, device_name);
	input_dims_face = input_info_face.getDims();
	input_info_face_name = input_info_face.name();
	output_info_face_name = output_info_face.name();
}

var normImg;

async function faceD(img) {

    var results = [];

    var resultsObj = {}

    var resultsArr = [];

    var dims;

    const image_path = img;

    const input_h_face = input_dims_face[2];
    const input_w_face = input_dims_face[3];

    // MAKE A COPY OF THE FACE IMAGE TO SCALE
    var agImage = await jimp.read(image_path);
    var image = await jimp.read(image_path);
    image.contain(1000,1000);
//    image.write('./outputs/input.jpg');

    if (agImage.bitmap.height !== input_h_face &&
    agImage.bitmap.width !== input_w_face) {
    agImage.background(0xFFFFFFFF);
    agImage.contain(agImage.bitmap.width, agImage.bitmap.width);
    agImage.resize(input_w_face, input_h_face, jimp.RESIZE_BILINEAR);
    }

  normImg = await jimp.read(image);

  let infer_req_face;
  infer_req_face = exec_net_face.createInferRequest();
  const input_blob_face = infer_req_face.getBlob(input_info_face.name());
  const input_data_face = new Uint8Array(input_blob_face.wmap());

  agImage.scan(0, 0, agImage.bitmap.width, agImage.bitmap.height, function (x, y, hdx) {
    let h = Math.floor(hdx / 4) * 3;
    input_data_face[h + 2] = agImage.bitmap.data[hdx + 0];  // R
    input_data_face[h + 1] = agImage.bitmap.data[hdx + 1];  // G
    input_data_face[h + 0] = agImage.bitmap.data[hdx + 2];  // B
  });

//  input_blob_face.unmap();

  infer_req_face.infer();

  const output_blob_face = infer_req_face.getBlob(output_info_face.name());
  const output_data_face = new Float32Array(output_blob_face.rmap());

  for (i = 0, len = output_data_face.length; i < len; i += 7) {
    if(output_data_face[i+2] > 0.8 && ((output_data_face[i+5] * image.bitmap.width) - (output_data_face[i+3] * image.bitmap.width)) >= 30) {
      results.push(output_data_face.slice(i, i + 7));
    }
  }

  var counter = 0;

  if(results.length > 0) {
    results.forEach(item => {

      dims = {
         x: parseInt(item[3] * image.bitmap.width) - 10,
         y: parseInt(item[4] * image.bitmap.height) + 10,
         w: (parseInt(item[5] * image.bitmap.width)+10) - parseInt((item[3] * image.bitmap.width)-10),
         h: (parseInt(item[6] * image.bitmap.height)+10) - parseInt((item[4] * image.bitmap.height)-10)
      };

    resultsObj = {
      confidence: parseInt(item[2]* 100),
      dims: dims,
      img: null
    };

     let holdImg = new jimp({ data:image.bitmap.data, width: image.bitmap.width, height: image.bitmap.height});

      holdImg.crop((dims.x-((dims.h-dims.w)/2)), (dims.y), (dims.h), (dims.h))
      .contain(250,250)
//      .resize(250,250)
      .background(0xFFFFFFFF)
//      .greyscale()
//      .write('./outputs/detector-' + counter + '.jpg');
      counter++;

    resultsObj.img = holdImg;
    resultsArr.push(resultsObj);
    resultsObj = {};
    });
    counter = 0;
  }

	var combined = { results: resultsArr, img: normImg };
//    console.log(resultsArr.length);
    return combined;
}

module.exports = { faceD, normImg, faceEngine };
