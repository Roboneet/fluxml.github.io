using BSON, Flux

function recreate()
  m = Chain(
    Conv((3, 3), 1=>16, relu),
    x -> maxpool(x, (2,2)),
    Conv((2,2), 16=>8, relu),
    x -> maxpool(x, (2,2)),
    x -> reshape(x, :, size(x, 4)),
    Dense(288, 10), softmax)

  w = BSON.load("mnist-conv.bson")[:weights]
  onedim = (x) -> reshape(x, size(x)[3])
  perm = (x) -> permutedims(x, (3, 4, 2, 1))

  w[2] = onedim(w[2])
  w[4] = onedim(w[4])
  w[1] = perm(w[1])
  w[3] = perm(w[3])
  Flux.loadparams!(m, w)
  return m
end

m = recreate()

using Flux, Flux.Data.MNIST
using Flux: onehotbatch, argmax, crossentropy, throttle
using BSON
using Base.Iterators: repeated, partition

imgs = MNIST.images()
labels = onehotbatch(MNIST.labels(), 0:9)

tX = cat(4, float.(MNIST.images(:test)[1:1000])...)

tY = onehotbatch(MNIST.labels(:test)[1:1000], 0:9)


accuracy(x, y) = mean(argmax(m(x)) .== argmax(y))

println(accuracy(tX, tY))
