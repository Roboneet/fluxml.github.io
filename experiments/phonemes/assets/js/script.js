(function (obj){
	Object.assign(obj, {Model, Result})
	function Model(model, res){
		this.latest = -1;
		this.input = "";
		var alphabet = null, phones = null;
		var START = ":", END = "/";


		this.predict = function(input){
			var index = this.latest + 1;
			this.latest = index;
			var scope = this;
			console.log("predict")
			setTimeout(() => scope.predict_(index, input), 100)
		}

		this.predict_ = async function predict(index, input){
			if(index != this.latest ||
			 input == this.input ||
			 input.length == 0 ||
			 !alphabet || !phones) return;

			console.log("predict_")
			this.latest = -1;
			this.input = input;
			res.show("Calculating...")

			var tokens = await tokenise(input);
			tokens = encode(tokens);
			var p = [START];
			this.predict_next(p, 0, tokens);
			
			

		}

		this.predict_next = async function(p, i, tokens){
			var scope = this;

			if(i >= 50)return;
			var phone = await onehot(p[i], phones)
			var dist = decode(tokens, phone);
			var next = await choose(dist);
			this.show(p);
			if( next == END ) return;
			p.push(next);
			var r = () => scope.predict_next(p, i + 1, tokens);
			requestAnimationFrame(r);
		}

		this.show = (p) =>	
			res.show(p.slice(1).join(" "))
		
		// get real alphabet list and phoneme list	
		var p = ["./assets/bson/alphabet.bson","./assets/bson/phones.bson"].map(flux.fetchData)
		Promise.all(p).then(res =>{
			alphabet = [END, ...res[0].alph];
			phones = [START, END, ...res[1].ph];
		})

		var tokenise = (t) => {
			var n = [];
			var l = null;
			for(var i of t)
				n.push(onehot(i, alphabet));

			return Promise.all(n);
		};

		var onehot = (k, arr) => 
			tf.oneHot(tf.tensor1d([arr.indexOf(k)], 'int32'), arr.length)
			.data().then(t => tf.tensor1d(t));
		

		var choose = async function(t){
			// weighted sample ?
			var index = await tf.argMax(t).data();
			return phones[index[0]];
		};

		var encode = function(tokens){
			var hf = tokens.map(token => model.forward(token));
			var hb = tokens.slice().reverse().map(token => 
				model.backward(token)).reverse()
			return [...Array(hf.length).keys()].map((i) =>
				tf.concat([hf[i], hb[i]]));
		}

		var decode = function (tokens, phone) {
			var a = model.recur.getStates()[0][1].clone();

			var b = tokens
			  .map(t => tf.concat([t, a]))
			  .map(e => model.alignnet(e));
			var weights = asoftmax(b);
			var context = tokens.map((h, i) =>
			  tf.mul(h, weights[i])).reduce(tf.add);
			
			var y = model.recur(tf.concat([phone, context]));
			return tf.softmax(model.toalpha(y));
		}

		var asoftmax = (xs) =>{
			xs = xs.map(x => tf.exp(x));
			var sum = tf.sum(tf.stack(xs));
			return xs.map(x => tf.div(x,sum));
		}
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
	model = new Model(model, result);
	$$(".input_box input").addEventListener("keyup", function(event){
		var value = event.target.value;
		model.predict(value);
	})
}

window.onload = __init__