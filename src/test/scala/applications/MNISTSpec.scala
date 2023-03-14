package applications

import autodifferentiation.{Constant, Input, Mean, Square, Subtract}
import layers.{Dense, InputLayer, Sigmoid}
import model.Model
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MNISTSpec extends AnyFlatSpec with Matchers {
  private val dataset = MNIST.getDataset

  "An MNIST" should "return the MNIST dataset" in {
    val xTrain = dataset._1
    val yTrain = dataset._2
    val xTest = dataset._3
    val yTest = dataset._4
    assert(xTrain.shape sameElements Array(60000, 28, 28))
    assert(xTrain.flatten().forall(pixel => pixel >= 0 && pixel < 256))
    assert(yTrain.shape sameElements Array(60000))
    assert(yTrain.flatten().forall(label => label >= 0 && label < 10))
    assert(xTest.shape sameElements Array(10000, 28, 28))
    assert(xTest.flatten().forall(pixel => pixel >= 0 && pixel < 256))
    assert(yTest.shape sameElements Array(10000))
    assert(yTest.flatten().forall(label => label >= 0 && label < 10))
  }

  // We ignore this test because it is too slow to converge, but we keep it as
  // our goal for the initial release.
  ignore should "be easy to train a model on MNIST" in {
    val xTrain = dataset._1.reshape(Array(60000, 28 * 28)).toFloat / 255
    val yTrain = dataset._2.toCategorical().toFloat
    assert(xTrain.flatten().forall(pixel => pixel >= 0 && pixel <= 1))
    assert(yTrain.flatten().forall(label => label == 0 || label == 1))
    val input = Input[Float]("X", Array(None, Some(28 * 28)))
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
    val fittedModel = model.fit(inputs, yTrain, 10)
    val lossFunctionAfter = Mean(
      Square(
        Subtract(fittedModel.outputLayer.getComputationGraph, Constant(yTrain))
      )
    )
    val lossAfter = lossFunctionAfter.compute(inputs).flatten().head
    assert(lossAfter < lossBefore)
  }
}
