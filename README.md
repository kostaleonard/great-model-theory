# Great model theory

A deep learning library for Scala.

## Usage

The package follows conventions commonly found in deep learning libraries like
TensorFlow.

```scala
import layers.Dense
import model.Model
import ndarray.NDArray

val BATCH_SIZE_PLACEHOLDER = 1
val numFeatures = 4
val hiddenSize = 3
val outputSize = 2
val input = Input[Float]("X", List(BATCH_SIZE_PLACEHOLDER, numFeatures))
val inputLayer = InputLayer(input)
val dense1 = Dense.withRandomWeights(inputLayer, hiddenSize)
val dense2 = Dense.withRandomWeights(dense1, outputSize)
val model = Model(dense2)
val sampleBatchSize = 2
val inputs =
  Map(input -> NDArray.ones[Float](List(sampleBatchSize, numFeatures)))
val outputs = model(inputs)
```

## API docs

The full Scala docs are available
[on GitHub Pages](https://kostaleonard.github.io/great-model-theory/).

## Design goals

* Purely functional data structures: data structures do not maintain mutable
state and operations on them do not create side effects.[^1]
* Familiar interfaces: API is similar to widely-used deep learning libraries,
like TensorFlow.

[^1]: In rare cases, we allow side effects that the user sanctions, such as
printing training progress and saving trained models to files.
