function loadWeights(url, pc, func){
	return _loadWeights([{url, model: window.model}], pc, func);
}

function _loadWeights(configArr, progressContainer, __init__){
	let Buffer = new BSON().serialize({}).constructor

	// send an xhr Request so as to show event progress
	var config = configArr.map((c)=>{
		var weightsRequest = new XMLHttpRequest();
		weightsRequest.open('GET', c.url);
		weightsRequest.responseType = "arraybuffer";
		return {xhr: weightsRequest, ...c}
	});

	// initialise progress bar
	var pbar = new ProgressBar({
		config,
		container: progressContainer,
		done: function(results){
			results.forEach(({event, model})=>{
				{
					var response = new Buffer(event.currentTarget.response);
					var data = new BSON().deserialize(response);
					model.weights = flux.convertArrays_(data).weights;
				}
			});
			__init__()
		},
		err: console.log
	})

	window.onload = ()=>{
		pbar.start();
		config.forEach(({xhr}) => xhr.send());
	}

}