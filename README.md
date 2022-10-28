# Igneous

A deep learning library for Scala.

## Motivation

This project is (currently) exploratory in nature. I enjoy programming in Scala, but mostly use Python day-to-day; this
work provides context and challenge for exercising my Scala skills. That fact influenced many design decisions,
including the reimplementation of NDArrays already widely available in [DL4J/ND4J](https://deeplearning4j.konduit.ai/nd4j/tutorials/quickstart).

## Usage

The package follows conventions commonly found in deep learning libraries like TensorFlow.

```scala
import layers.Dense
import model.Model
import ndarray.NDArray

val BATCH_SIZE_PLACEHOLDER = 1
val numFeatures = 4
val hiddenSize = 3
val outputSize = 2
val dense1 = new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, numFeatures), hiddenSize)
val dense2 = new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, hiddenSize), outputSize)
val model = new Model[Float](List(dense1, dense2))
val sampleBatchSize = 2
val inputs = NDArray.ones[Float](List(sampleBatchSize, numFeatures))
val outputs = model(inputs)
```

## On the package name

The phonetic equivalent of "Scala" in Russian means "rock" or "crag." I happened to be doing something in Russian
around the time that I made this project, hence "igneous."
