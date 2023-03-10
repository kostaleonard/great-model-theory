package applications

import autodifferentiation.{Constant, Input, Mean, Square, Subtract}
import layers.{Dense, InputLayer}
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
    //TODO yTrain needs to be one-hot
    val xTrain = dataset._1.toFloat.reshape(Array(60000, 28 * 28)) / NDArray.ofValue[Float](Array(60000, 28 * 28), 255)
    /*
    val xTrain = dataset._1.toFloat.reshape(Array(60000, 28 * 28)) / 255
    val yTrain = dataset._2.toFloat
    val input = Input[Float]("X", Array(None, Some(28 * 28)))
    val inputLayer = InputLayer(input)
    val dense1 = Dense.withRandomWeights(inputLayer, 128)
    val dense2 = Dense.withRandomWeights(dense1, 10)
    val model = Model(dense2)
    val inputs = Map(input -> xTrain)
    val lossFunctionBefore = Mean(
      Square(Subtract(model.outputLayer.getComputationGraph, Constant(yTrain)))
    )
    val lossBefore = lossFunctionBefore.compute(inputs).flatten().head
    val fittedModel = model.fit(inputs, yTrain, 10, learningRate = 1e-2)
    val lossFunctionAfter = Mean(
      Square(
        Subtract(fittedModel.outputLayer.getComputationGraph, Constant(yTrain))
      )
    )
    val lossAfter = lossFunctionAfter.compute(inputs).flatten().head
    assert(lossAfter < lossBefore)
     */
  }
}
