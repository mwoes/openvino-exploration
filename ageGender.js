const { Core, getVersion } = require('../../lib/inference-engine-node');

const jimp = require('jimp');
const fs = require('fs').promises;
const { performance } = require('perf_hooks');

const {
  binPathFromXML
} = require('../common');

var results;
var counter = 0;

var core_age_gender, model_age_gender, bin_path_age_gender, net_age_gender, inputs_info_age_gender, outputs_info_age_gender, input_info_age_gender, output_info_age, output_info_gender, exec_net_age_gender, input_dims_age_gender, input_info_age_gender_name, output_info_age_gender_name;

async function ageGenEngine(device_name) {
	core_age_gender = new Core();
	model_age_gender = '/opt/intel/openvino_2020.3.194/deployment_tools/open_model_zoo/tools/downloader/intel/age-gender-recognition-retail-0013/FP32/age-gender-recognition-retail-0013.xml';
	bin_path_age_gender = binPathFromXML(model_age_gender);
	net_age_gender = await core_age_gender.readNetwork(model_age_gender, bin_path_age_gender);
	inputs_info_age_gender = net_age_gender.getInputsInfo();
	outputs_info_age_gender = net_age_gender.getOutputsInfo();
	input_info_age_gender = inputs_info_age_gender[0];
    output_info_age = outputs_info_age_gender[0];
    output_info_gender = outputs_info_age_gender[1];
	input_info_age_gender.setLayout('nhwc');
	input_info_age_gender.setPrecision('u8');
	exec_net_age_gender = await core_age_gender.loadNetwork(net_age_gender, device_name);
	input_dims_age_gender = input_info_age_gender.getDims();
}

async function ageGender(img) {

    var consolidatedResults = [];

    const input_h_age_gender = input_dims_age_gender[2];
    const input_w_age_gender = input_dims_age_gender[3];

    var image = await jimp.read(img.img);

    // MAKE A COPY OF THE FACE IMAGE TO SCALE
    const agImage = await jimp.read(image);

    if (agImage.bitmap.height !== input_h_age_gender &&
    image.bitmap.width !== input_w_age_gender) {
    image.resize(input_w_age_gender, input_h_age_gender);
//    image.write('./outputs/ageGender_age_gender' + counter + '.jpg');
    }

    // START AGE & GENDER ESTIMATION
    let infer_req_age_gender;
    let infer_time_age_gender = [];
    start_time = performance.now();
    infer_req_age_gender = exec_net_age_gender.createInferRequest();
    var input_blob_age_gender = infer_req_age_gender.getBlob(input_info_age_gender.name());
    var input_data_age_gender = new Uint8Array(input_blob_age_gender.wmap());

    agImage.scan(0, 0, agImage.bitmap.width, agImage.bitmap.height, function (x, y, idx) {
    let i = Math.floor(idx / 4) * 3;
    input_data_age_gender[i + 2] = agImage.bitmap.data[idx + 0];  // R
    input_data_age_gender[i + 1] = agImage.bitmap.data[idx + 1];  // G
    input_data_age_gender[i + 0] = agImage.bitmap.data[idx + 2];  // B
    });

    input_blob_age_gender.unmap();
    start_time = performance.now();

    infer_req_age_gender.infer();

    infer_time_age_gender.push(performance.now() - start_time);

    var output_blob_age = infer_req_age_gender.getBlob(output_info_age.name());
    var output_data_age = new Float32Array(output_blob_age.rmap());

    var output_blob_gender = infer_req_age_gender.getBlob(output_info_gender.name());
    output_blob_gender.unmap();
    var output_data_gender = new Float32Array(output_blob_gender.rmap());

    var female = parseInt((output_data_gender[0] * 100),10);
    var male = parseInt((output_data_gender[1] * 100),10);

    var ageResult = (output_data_age[0] * 100).toFixed(0);
    var genResult = null;

    if ((female - male) > 0) {
      genResult = "FEMALE";
    } else if((male - female) > 20) {
      genResult = "MALE";
    } else {
      genResult = "NON-BINARY"
    }

    results = { age: ageResult, gen: genResult, raw: [ female, male], counter: counter };
    counter++;


    return results;
}

module.exports = { ageGender, ageGenEngine };
