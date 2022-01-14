const { Core, getVersion } = require('../../lib/inference-engine-node');

const jimp = require('jimp');
const fs = require('fs').promises;
const { performance } = require('perf_hooks');

const {
  binPathFromXML
} = require('../common');

var results;

var core_emotion, model_emotion, bin_path_emotion, net_emotion, inputs_info_emotion, outputs_info_emotion, input_info_emotion, output_info_emotion, exec_net_emotion, input_dims_emotion, input_info_emotion_name, output_info_emotion_name;

async function emoEngine(device_name) {
	core_emotion = new Core();
	model_emotion = '/opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/intel/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003.xml';
	bin_path_emotion = binPathFromXML(model_emotion);
	net_emotion = await core_emotion.readNetwork(model_emotion, bin_path_emotion);
	inputs_info_emotion = net_emotion.getInputsInfo();
	outputs_info_emotion = net_emotion.getOutputsInfo();
	input_info_emotion = inputs_info_emotion[0];
	output_info_emotion = outputs_info_emotion[0];
	input_info_emotion.setLayout('nhwc');
	input_info_emotion.setPrecision('u8');
	exec_net_emotion = await core_emotion.loadNetwork(net_emotion, device_name);
	input_dims_emotion = input_info_emotion.getDims();
	input_info_emotion_name = input_info_emotion.name();
	output_info_emotion_name = output_info_emotion.name();
}

async function emotion(img) {

    const image = img.img;
    const input_h_emotion = input_dims_emotion[2];
    const input_w_emotion = input_dims_emotion[3];

    // MAKE A COPY OF THE FACE IMAGE TO SCALE
    let agImage = image;

    if (agImage.bitmap.height !== input_h_emotion &&
    agImage.bitmap.width !== input_w_emotion) {
    agImage.resize(input_w_emotion, input_h_emotion, jimp.RESIZE_BILINEAR);
    }

  // START EMOTIONAL ESTIMATION
  let infer_req_emotion;
  let infer_time_emotion = [];
  start_time = performance.now();
  infer_req_emotion = exec_net_emotion.createInferRequest();
  const input_blob_emotion = infer_req_emotion.getBlob(input_info_emotion.name());
  const input_data_emotion = new Uint8Array(input_blob_emotion.wmap());

  agImage.scan(0, 0, agImage.bitmap.width, agImage.bitmap.height, function (x, y, hdx) {
    let h = Math.floor(hdx / 4) * 3;
    input_data_emotion[h + 2] = agImage.bitmap.data[hdx + 0];  // R
    input_data_emotion[h + 1] = agImage.bitmap.data[hdx + 1];  // G
    input_data_emotion[h + 0] = agImage.bitmap.data[hdx + 2];  // B
  });

  input_blob_emotion.unmap();
  start_time = performance.now();

  infer_req_emotion.infer();

  infer_time_emotion.push(performance.now() - start_time);

  const output_blob_emotion = infer_req_emotion.getBlob(output_info_emotion.name());
  const output_data_emotion = new Float32Array(output_blob_emotion.rmap());

  var emote = 0;
  var curTop = null;

  for (var i=0; i < output_data_emotion.length; i++) {
    if(output_data_emotion[i] > emote) {
      emote = output_data_emotion[i];
      curTop = i;
    }
  }

  var moods = ["neutral", "happy", "sad", "surprise", "anger"];

  results = {
    emotion: moods[curTop],
    raw: output_data_emotion
  }

    return results;
}

module.exports = { emotion, emoEngine };
