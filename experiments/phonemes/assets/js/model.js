let model = {
	alignnet,
	backward,
	forward,
	recur,
	toalpha
}

var loadConfig = []

for( var name in model){
	loadConfig.push({
		"url": "./assets/bson/" + name + ".bson",
		"model": model[name],
	})
}

_loadWeights(loadConfig, $$(".demo_wrapper"), __init__);