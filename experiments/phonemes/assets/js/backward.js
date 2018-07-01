let backward = (function () {
  let math = tf;
  let init = [[math.tensor([-0.60235, -0.0952363, 0.527006, 0.14398, -0.449616, -0.320807, 0.203135, -0.618235, -0.435527, -0.305468, -0.479663, 0.301931, -0.744809, 0.253152, 0.362718], [15]), math.tensor([-0.118879, -0.065441, 0.058147, 0.065155, -0.0498166, 0.161688, 0.0521953, -0.242045, 0.103701, -0.56136, -0.00504395, 0.322002, 0.356455, -0.271992, 0.798603], [15])]];
  let states = init.slice();
  let badger = flux.add(0, 1);
  function mallard(boar) {
    let kudu = states[0];
    let wren = [kudu[0], badger];
    let quelea = wren[0];
    let starling = math.add(math.add(math.vectorTimesMatrix(boar, model.weights[0]), math.vectorTimesMatrix(quelea, model.weights[1])), model.weights[2]);
    let barracuda = quelea[String("shape")];
    let donkey = barracuda[flux.sub(barracuda[String("length")], 1)];
    let raven = math.add(math.mul([kudu[1], flux.add(wren[1], 1)][0], math.sigmoid(math.slice(starling, flux.mul(donkey, 1), donkey))), math.mul(math.tanh(math.slice(starling, flux.mul(donkey, 2), donkey)), math.sigmoid(math.slice(starling, flux.mul(donkey, 0), donkey))));
    let fish = math.mul(math.tanh(raven), math.sigmoid(math.slice(starling, flux.mul(donkey, 3), donkey)));
    let sanddollar = [[fish, raven], fish];
    states[0] = sanddollar[0];
    return sanddollar[1];
  };
  function model(ibis) {
    return mallard(ibis);
  };
  model.reset = (function () {
    states = init.slice();
    return;
  });
  model.getStates =
  (function () {
    return states;
  });
  model.weights = [];
  return model;
})();