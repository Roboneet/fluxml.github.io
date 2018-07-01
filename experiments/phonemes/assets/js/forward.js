let forward = (function () {
  let math = tf;
  let init = [[math.tensor([0.0543909, -0.245524, -0.6524, 0.64433, -0.202386, 0.167879, 0.762736, -0.190532, -0.326335, 0.0670347, -0.568173, 0.106152, -0.136771, -0.0410403, -0.23602], [15]), math.tensor([-0.296014, 0.240752, 0.185712, 0.0749496, 0.484515, 0.366127, 0.342897, 0.197107, -0.181077, -0.325256, -0.154643, -0.246441, 0.173582, 0.441777, -0.297895], [15])]];
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
  model.getStates= 
  (function () {
    return states;
  });
  model.weights = [];
  return model;
})();
