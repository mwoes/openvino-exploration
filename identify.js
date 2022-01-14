const { Core, getVersion } = require('../../lib/inference-engine-node');

const jimp = require('jimp');
const fs = require('fs').promises;
const { performance } = require('perf_hooks');

const similarity = require('compute-cosine-similarity');

const {
  binPathFromXML,
  showInputOutputInfo
} = require('../common');

var core_identity, model_identity, bin_path_identity, net_identity, inputs_info_identity, outputs_info_identity, input_info_identity, output_info_identity, exec_net_identity, input_dims_identity, input_info_identity_name, output_info_identity_name;

var identities = [];

async function identEngine(device_name) {
	core_identity = new Core();
	model_identity = './models/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml';
	bin_path_identity = binPathFromXML(model_identity);
	net_identity = await core_identity.readNetwork(model_identity, bin_path_identity);
	inputs_info_identity = net_identity.getInputsInfo();
	outputs_info_identity = net_identity.getOutputsInfo();
	input_info_identity = inputs_info_identity[0];
	output_info_identity = outputs_info_identity[0];
	input_info_identity.setLayout('nhwc');
	input_info_identity.setPrecision('u8');
	exec_net_identity = await core_identity.loadNetwork(net_identity, device_name);
	input_dims_identity = input_info_identity.getDims();
	input_info_identity_name = input_info_identity.name();
	output_info_identity_name = output_info_identity.name();
}


async function getIdentityRaw(img, landmarks, pose) {

var results = [];

var resultsObj = {
  vect: identities
};

    const image = img.img;

    const agImage = await jimp.read(image);

    const input_dims_identity = input_info_identity.getDims();
    const input_h_identity = input_dims_identity[2];
    const input_w_identity = input_dims_identity[3];

    // MAKE A COPY OF THE FACE IMAGE TO SCALE

    if (agImage.bitmap.height !== input_h_identity &&
    agImage.bitmap.width !== input_w_identity) {
    agImage.contain(input_w_identity, input_h_identity);
    agImage.rotate(pose.pitch);
    }

    agImage.write('./outputs/ident.jpg');

  // START EMOTIONAL ESTIMATION
  let infer_req_identity;
  let infer_time_identity = [];
  infer_req_identity = exec_net_identity.createInferRequest();
  const input_blob_identity = infer_req_identity.getBlob(input_info_identity.name());
  const input_data_identity = new Uint8Array(input_blob_identity.wmap());

  agImage.scan(0, 0, agImage.bitmap.width, agImage.bitmap.height, function (x, y, hdx) {
    let h = Math.floor(hdx / 4) * 3;
    input_data_identity[h + 2] = agImage.bitmap.data[hdx + 0];  // R
    input_data_identity[h + 1] = agImage.bitmap.data[hdx + 1];  // G
    input_data_identity[h + 0] = agImage.bitmap.data[hdx + 2];  // B
  });

  input_blob_identity.unmap();

  infer_req_identity.infer();

  const output_blob_identity = infer_req_identity.getBlob(output_info_identity.name());
  const output_data_identity = new Float32Array(output_blob_identity.rmap());
//  console.table(output_data_identity);

  output_data_identity.forEach((data, i) => {
    data = parseInt(data * 100);
    output_data_identity[i] = data;
  });

  results = Array.from(output_data_identity);

 if(identities.length < 1) { identities.push(results); };

  var idIndex = 0;
  var confidenceMax = 0;

  if(identities.length > 0) {
    for(var i=0; i<identities.length; i++) {
      var simout = parseInt(similarity( results, identities[i] ) * 1000);
      console.log(i + ":" + simout);
      if(simout < 650 && simout > 0) { identities.push(results); };
      if(simout > confidenceMax) {
        confidenceMax = simout;
        idIndex = i;
      }
    }
  }

  resultsObj.vect = idIndex + ":" + confidenceMax;

  return resultsObj;
}

module.exports = { getIdentityRaw, identEngine };
