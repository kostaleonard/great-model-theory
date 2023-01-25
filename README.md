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
val model = new Model[Float](dense2)
val sampleBatchSize = 2
val inputs =
  Map(input -> NDArray.ones[Float](List(sampleBatchSize, numFeatures)))
val outputs = model(inputs)
```

## API docs

The full Scala docs are available
[on GitHub Pages](https://kostaleonard.github.io/great-model-theory/).
