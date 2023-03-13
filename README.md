# Great model theory

A deep learning library for Scala.

## Usage

The package follows conventions commonly found in deep learning libraries like
TensorFlow.

```scala
import applications.MNIST
import autodifferentiation.Input
import layers.{Dense, InputLayer, Sigmoid}
import model.Model

val dataset = MNIST.getDataset
val xTrain = dataset._1.reshape(Array(60000, 28 * 28)).toFloat / 255
val yTrain = dataset._2.toCategorical().toFloat
val xTest = dataset._3.reshape(Array(10000, 28 * 28)).toFloat / 255
val yTest = dataset._4.toCategorical().toFloat
val input = Input[Float]("X", Array(None, Some(28 * 28)))
val inputLayer = InputLayer(input)
val dense1 = Dense.withRandomWeights(inputLayer, 128)
val activation1 = Sigmoid(dense1)
val dense2 = Dense.withRandomWeights(activation1, 10)
val activation2 = Sigmoid(dense2)
val neuralNetwork = Model(activation2)
val inputs = Map(input -> xTrain)
val lossBefore = neuralNetwork.evaluate(Map(input -> xTest), yTest)
val fittedNeuralNetwork = neuralNetwork.fit(inputs, yTrain, 10)
val lossAfter = fittedNeuralNetwork.evaluate(Map(input -> xTest), yTest)
println(s"Loss before: $lossBefore")
println(s"Loss after: $lossAfter")
```

## API docs

The full Scala docs are available
[on GitHub Pages](https://kostaleonard.github.io/great-model-theory/).

## Design goals

* Purely functional data structures: data structures do not maintain mutable
state and operations on them do not create side effects.
* Familiar interfaces: API is similar to widely-used deep learning libraries,
like TensorFlow.
