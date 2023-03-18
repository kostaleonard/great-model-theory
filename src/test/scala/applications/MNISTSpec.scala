package applications

import autodifferentiation.{Constant, Input, Mean, Square, Subtract}
import layers.{Dense, InputLayer, Sigmoid}
import model.Model
import ndarray.NDArray
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

  // TODO do not ignore
  it should "be easy to train a model on MNIST" in {
    //TODO remove slicing
    val numSamples = 512
    val xTrain = (dataset._1.reshape(Array(60000, 28 * 28)).toFloat / NDArray.ofValue[Float](Array(60000, 28 * 28), 255)).slice(Array(Some(Array.range(0, numSamples)), None))
    val yTrain = dataset._2.toCategorical().toFloat.slice(Array(Some(Array.range(0, numSamples)), None))
    //TODO broadcasting is slow, so we eliminate it here--make issue
    //val xTrain = dataset._1.reshape(Array(60000, 28 * 28)).toFloat / NDArray.ofValue[Float](Array(60000, 28 * 28), 255)
    //val yTrain = dataset._2.toCategorical().toFloat
    assert(xTrain.flatten().forall(pixel => pixel >= 0 && pixel <= 1))
    assert(yTrain.flatten().forall(label => label == 0 || label == 1))
    // TODO users should not have to know about anything in autodifferentiaion module. Refactor and update README example.
    val input = Input[Float]("X", Array(None, Some(28 * 28)))
    val inputLayer = InputLayer(input)
    val dense1 = Dense.withRandomWeights(inputLayer, 128)
    val activation1 = Sigmoid(dense1)
    val dense2 = Dense.withRandomWeights(activation1, 10)
    val activation2 = Sigmoid(dense2)
    val model = Model(activation2)
    val inputs = Map("X" -> xTrain)
    val lossFunctionBefore = Mean(
      Square(Subtract(model.outputLayer.getComputationGraph, Constant(yTrain)))
    )
    val lossBefore = lossFunctionBefore.compute(inputs).flatten().head
    //TODO remove debugging
    println(lossBefore)
    val predictionsBefore = model.predict(inputs)
    println(predictionsBefore)
    //TODO very poor argmax
    val predictionsBeforeArgMax = predictionsBefore.reduce(arr => arr.flatten().indexOf(arr.flatten().max), 1)
    val labels = dataset._2.slice(Array(Some(Array.range(0, numSamples))))
    println(predictionsBeforeArgMax)
    println(labels)
    val accuracyBefore = (predictionsBeforeArgMax == labels).map(b => if(b) 1 else 0).sum / numSamples.toFloat
    println(accuracyBefore)
    val fittedModel = model.fit(inputs, yTrain, 10, learningRate = 1e-2)
    val lossFunctionAfter = Mean(
      Square(
        Subtract(fittedModel.outputLayer.getComputationGraph, Constant(yTrain))
      )
    )
    val lossAfter = lossFunctionAfter.compute(inputs).flatten().head
    assert(lossAfter < lossBefore)
    val predictionsAfter = fittedModel.predict(inputs)
    val predictionsAfterArgmax = predictionsAfter.reduce(arr => arr.flatten().indexOf(arr.flatten().max), 1)
    println(predictionsAfterArgmax)
    println(labels)
    val accuracyAfter = (predictionsAfterArgmax == labels).map(b => if(b) 1 else 0).sum / numSamples.toFloat
    println(accuracyAfter)
  }
}
