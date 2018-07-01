let recur = (function () {
  let math = tf;
  let init = [[math.tensor([-0.114085, 0.00553356, 0.161152, 0.106185, -0.0182924, -0.367847, -0.232405, -0.0148731, -0.00770601, 0.197976, 0.00974272, 0.0303332, -0.106213, -0.0743409, 0.082977, -0.310483, 0.0673023, -0.449977, 0.0718713, 0.228455, 0.0147451, 0.00585169, -0.212789, 0.157657, -0.135127, -0.0631522, 0.0326177, 0.285388, 0.128849, -0.151466], [30]), math.tensor([0.084045, -0.189252, 0.653564, 0.448313, -0.667096, 1.05393, -0.411399, -0.622821, 0.898865, -0.371451, 3.73213, -0.269009, -0.0423273, -0.389874, 0.540078, 0.364075, -0.681606, 0.201408, -0.730977, -1.36486, 0.0796199, 0.0118966, -0.440127, -0.192523, 0.0165335, 1.16731, 0.956047, -0.153105, 0.723848, 3.22345], [30])]];
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
