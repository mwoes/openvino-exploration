const { Core, getVersion } = require('../../lib/inference-engine-node');

const jimp = require('jimp');
const fs = require('fs').promises;
const { performance } = require('perf_hooks');

const {
  binPathFromXML
} = require('../common');

var results;

var core_headpose, model_headpose, bin_path_headpose, net_headpose, inputs_info_headpose, outputs_info_headpose, input_info_headpose, output_info_yaw, output_info_pitch, output_info_roll, exec_net_headpose, input_dims_headpose;

async function headposeEngine(device_name) {
	core_headpose = new Core();
	model_headpose = '/opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml';
	bin_path_headpose = binPathFromXML(model_headpose);
	net_headpose = await core_headpose.readNetwork(model_headpose, bin_path_headpose);
	inputs_info_headpose = net_headpose.getInputsInfo();
	outputs_info_headpose = net_headpose.getOutputsInfo();
	input_info_headpose = inputs_info_headpose[0];
    output_info_yaw = outputs_info_headpose[0];
    output_info_pitch = outputs_info_headpose[1];
    output_info_roll = outputs_info_headpose[2];	input_info_headpose.setLayout('nhwc');
	input_info_headpose.setPrecision('u8');
	exec_net_headpose = await core_headpose.loadNetwork(net_headpose, device_name);
	input_dims_headpose = input_info_headpose.getDims();
}

async function headposeR(img) {

    const image = img.img;
    const input_h_headpose = input_dims_headpose[2];
    const input_w_headpose = input_dims_headpose[3];

    // MAKE A COPY OF THE FACE IMAGE TO SCALE
    let agImage = image;

    if (agImage.bitmap.height !== input_h_headpose &&
    agImage.bitmap.width !== input_w_headpose) {
    agImage.resize(input_w_headpose, input_h_headpose, jimp.RESIZE_BILINEAR);
    }

  // START EMOTIONAL ESTIMATION
  let infer_req_headpose;
  let infer_time_headpose = [];
  start_time = performance.now();
  infer_req_headpose = exec_net_headpose.createInferRequest();
  const input_blob_headpose = infer_req_headpose.getBlob(input_info_headpose.name());
  const input_data_headpose = new Uint8Array(input_blob_headpose.wmap());

  agImage.scan(0, 0, agImage.bitmap.width, agImage.bitmap.height, function (x, y, hdx) {
    let h = Math.floor(hdx / 4) * 3;
    input_data_headpose[h + 2] = agImage.bitmap.data[hdx + 0];  // R
    input_data_headpose[h + 1] = agImage.bitmap.data[hdx + 1];  // G
    input_data_headpose[h + 0] = agImage.bitmap.data[hdx + 2];  // B
  });

  input_blob_headpose.unmap();
  start_time = performance.now();

  infer_req_headpose.infer();

  infer_time_headpose.push(performance.now() - start_time);

  const output_blob_yaw = infer_req_headpose.getBlob(output_info_yaw.name());
  const output_data_yaw = new Float32Array(output_blob_yaw.rmap());

  const output_blob_pitch = infer_req_headpose.getBlob(output_info_pitch.name());
  const output_data_pitch = new Float32Array(output_blob_pitch.rmap());

  const output_blob_roll = infer_req_headpose.getBlob(output_info_roll.name());
  const output_data_roll = new Float32Array(output_blob_roll.rmap());

  results = {
    yaw: output_data_yaw[0],
    pitch: output_data_pitch[0],
    roll: output_data_roll[0]
  }

    return results;
}

module.exports = { headposeR, headposeEngine };
