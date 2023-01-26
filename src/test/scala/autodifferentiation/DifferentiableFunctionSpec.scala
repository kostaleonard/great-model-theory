package autodifferentiation

import layers.{Dense, InputLayer}
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DifferentiableFunctionSpec extends AnyFlatSpec with Matchers {
  "A DifferentiableFunction" should "return its output shape (1)" in {
    val input = Input[Float]("X", Array(1))
    val addition = Add(input, Constant(NDArray.ones[Float](Array(1))))
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(1))
  }

  it should "return its output shape, with broadcasting (2 x 2)" in {
    val input = Input[Float]("X", Array(2, 2))
    val addition = Add(input, Constant(NDArray.ones[Float](Array(1))))
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(2, 2))
  }

  "A Constant" should "return its preset value when computed" in {
    val value = NDArray.ones[Int](Array(3))
    val constant = Constant(value)
    val output = constant.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayEquals value)
  }

  it should "return a gradient of all zeros matching the shape of its preset value" in {
    val value = NDArray.ones[Float](Array(3))
    val constant = Constant(value)
    // The equation f() = c does not use a variable, but we need a variable to
    // compute the gradient df/dx (you can't differentiate with respect to
    // nothing).
    val placeholderVariable = Input[Float]("X", Array(1))
    val gradient = constant.gradient(placeholderVariable)
    val output = gradient.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayEquals NDArray.zeros(value.shape))
  }

  "A Variable" should "have gradient 1 with respect to itself" in {
    val modelParameter = ModelParameter[Float]("Theta", NDArray.ofValue(Array(2, 3), 5))
    val gradient = modelParameter.gradient(modelParameter)
    val output = gradient.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals NDArray.ones(Array(2, 3)))
  }

  it should "have gradient 0 with respect to other variables" in {
    val modelParameter = ModelParameter[Float]("Theta", NDArray.ofValue(Array(2, 3), 5))
    val placeholderVariable = Input[Float]("X", Array(1))
    // Take the gradient of f() = Theta with respect to unrelated variable X.
    val gradient = modelParameter.gradient(placeholderVariable)
    val output = gradient.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals NDArray.zeros(Array(2, 3)))
  }

  "A ModelParameter" should "return its current value when computed" in {
    val value = NDArray.ofValue[Float](Array(2, 3), 5)
    val modelParameter = ModelParameter[Float]("Theta", value)
    val output = modelParameter.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals value)
  }

  "An Input" should "return the user-supplied value when computed" in {
    val input = Input[Float]("X", Array(2, 2))
    val value = NDArray.ofValue[Float](input.shape, 4)
    val output = input.compute(Map(input -> value))
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals value)
  }

  it should "fail to compute when the user-supplied value does not match the Input shape" in {
    //TODO Inputs need a shape argument so that we can compute output shapes and instantiate model parameters. I want to see those transformations (e.g., in Dense) fail as a sanity check before we implement this logic.
    val numFeatures = 4
    val input = Input[Float]("X", Array(1, numFeatures))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, 2).get
    val output = dense(Map(input -> NDArray.zeros(Array(numFeatures))))
    assert(output.isFailure)
  }

  it should "fail to compute when the user does not supply a necessary Input" in {
    val input = Input[Float]("X", Array(2, 2))
    val output = input.compute(Map.empty)
    assert(output.isFailure)
  }
}
