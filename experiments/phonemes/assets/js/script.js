(function (obj){
	Object.assign(obj, {Model, Result})
	function Model(model, res){
		this.latest = -1;

		this.predict = function(input){
			var index = this.latest + 1;
			this.latest = index;
			var scope = this;
			console.log("predict")
			setTimeout(() => scope.predict_(index, input), 100)
		}

		this.predict_ = function predict(index, input){
			if(index != this.latest)return;
			console.log("predict_")
			this.latest = -1;

			// return model(pre(input)).data().then(data => res.show(post(data)));

			return res.show("Model not loaded...")
		}

		// get real alphabet list and phoneme list
		let alphabet = new Array(26).fill(0).map((_, i)=> String.fromCharCode(i + "A".charCodeAt(0)))

		var pre = (t) => {
			var n = [];
			for(var i of t){
				n.push(alphabet.indexOf(i));
			}
			return tf.oneHot(tf.tensor1d(n), alphabet.length - 1);
		};

		var post = function(data){ 
			return data
		};
	}

	function Result(ele){
		this.show = (text)=>{
			ele.innerText = text;
			return text;
		}
	}
})(window)

var __init__ = function(){
	var result = new Result($$(".output_box p"));
	model = new Model(null, result);
	$$(".input_box input").addEventListener("keyup", function(event){
		var value = event.target.value;
		model.predict(value);
	})
}

window.onload = __init__