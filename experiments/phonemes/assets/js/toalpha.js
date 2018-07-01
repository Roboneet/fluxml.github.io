let toalpha = (function () {
  let math = tf;
  function badger(mallard) {
    return math.add(math.vectorTimesMatrix(mallard, model.weights[0]), model.weights[1]);
  };
  function model(boar) {
    return badger(boar);
  };
  model.weights = [];
  return model;
})();