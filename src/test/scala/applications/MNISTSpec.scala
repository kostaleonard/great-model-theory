package applications

import autodifferentiation.{Constant, Input, Mean, Square, Subtract}
import layers.{Dense, InputLayer, Sigmoid}
import model.Model
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MNISTSpec extends AnyFlatSpec with Matchers {
  "An MNIST" should "return the MNIST dataset" in {
    //TODO fixture for MNIST dataset so test suite only downloads once per run
    val dataset = MNIST.getDataset
    val xTrain = dataset._1
    val yTrain = dataset._2
    assert(xTrain.shape sameElements Array(60000, 28, 28))
    assert(xTrain.flatten().forall(pixel => pixel >= 0 && pixel < 256))
    assert(yTrain.shape sameElements Array(60000))
    assert(yTrain.flatten().forall(label => label >= 0 && label < 10))
  }

  it should "be easy to train a model on MNIST" in {
    //TODO update README with trimmed version of this example, plus the output
    //TODO fixture for MNIST dataset so test suite only downloads once per run
    val dataset = MNIST.getDataset
    //TODO make these floats
    //TODO use full dataset (no slice)
    //val xTrain = dataset._1.reshape(Array(60000, 28 * 28)).toFloat / 255
    //val yTrain = dataset._2.toCategorical().toFloat
    val xTrain = dataset._1.reshape(Array(60000, 28 * 28)).slice(Array(Some(Array.range(0, 4)), None)).toDouble / 255
    val yTrain = dataset._2.slice(Array(Some(Array.range(0, 4)))).toCategorical(numClasses = Some(10)).toDouble
    assert(xTrain.flatten().forall(pixel => pixel >= 0 && pixel <= 1))
    assert(yTrain.flatten().forall(label => label == 0 || label == 1))
    val input = Input[Double]("X", Array(None, Some(28 * 28)))
    val inputLayer = InputLayer(input)
    val dense1 = Dense.withRandomWeights(inputLayer, 128)
    val activation1 = Sigmoid(dense1)
    val dense2 = Dense.withRandomWeights(activation1, 10)
    val activation2 = Sigmoid(dense2)
    val model = Model(activation2)
    val inputs = Map(input -> xTrain)
    val lossFunctionBefore = Mean(
      Square(Subtract(model.outputLayer.getComputationGraph, Constant(yTrain)))
    )
    val lossBefore = lossFunctionBefore.compute(inputs).flatten().head
    val fittedModel = model.fit(inputs, yTrain, 10, learningRate = 1e-4)
    val lossFunctionAfter = Mean(
      Square(
        Subtract(fittedModel.outputLayer.getComputationGraph, Constant(yTrain))
      )
    )
    val lossAfter = lossFunctionAfter.compute(inputs).flatten().head
    assert(lossAfter < lossBefore)
  }
}
