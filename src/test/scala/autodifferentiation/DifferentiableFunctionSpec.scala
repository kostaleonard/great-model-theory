package autodifferentiation

import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.reflect.{ClassTag, classTag}
import scala.util.Try

class DifferentiableFunctionSpec extends AnyFlatSpec with Matchers {

  /** Numerically computes the gradient of the function for the inputs.
    *
    * If you plan to use this function to check gradients, use Double precision.
    */
  private def computeGradientWithFiniteDifferences[T: ClassTag](
      f: DifferentiableFunction[T],
      withRespectToInput: Input[T],
      inputs: Map[Input[T], NDArray[T]],
      epsilon: Double = 1e-5
  )(implicit num: Fractional[T]): Try[NDArray[T]] = {
    val epsilonArray = (classTag[T] match {
      case _ if classTag[T] == classTag[Float] =>
        NDArray[Float](List(epsilon.toFloat))
      case _ if classTag[T] == classTag[Double] =>
        NDArray[Double](List(epsilon))
    }).asInstanceOf[NDArray[T]]
    val inputsMinusEpsilon = inputs.updated(
      withRespectToInput,
      (inputs(withRespectToInput) - epsilonArray).get
    )
    val inputsPlusEpsilon = inputs.updated(
      withRespectToInput,
      (inputs(withRespectToInput) + epsilonArray).get
    )
    val outputMinusEpsilon = f.compute(inputsMinusEpsilon)
    val outputPlusEpsilon = f.compute(inputsPlusEpsilon)
    if (outputMinusEpsilon.isFailure) outputMinusEpsilon
    else if (outputPlusEpsilon.isFailure) outputPlusEpsilon
    else {
      val difference = outputPlusEpsilon.get - outputMinusEpsilon.get
      if (difference.isFailure) difference
      else
        (classTag[T] match {
          case _ if classTag[T] == classTag[Float] =>
            difference.get.asInstanceOf[NDArray[Float]] / NDArray[Float](
              List(2 * epsilon.toFloat)
            )
          case _ if classTag[T] == classTag[Double] =>
            difference.get.asInstanceOf[NDArray[Double]] / NDArray[Double](
              List(2 * epsilon)
            )
        }).asInstanceOf[Try[NDArray[T]]]
    }
  }

  "A DifferentiableFunction" should "return its output shape (1)" in {
    val input = Input[Float]("X", Array(Some(1)))
    val addition = Add(input, Constant(NDArray.ones[Float](Array(1))))
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(1)))
  }

  it should "return its output shape, with broadcasting (2 x 2)" in {
    val input = Input[Float]("X", Array(Some(2), Some(2)))
    val addition = Add(input, Constant(NDArray.ones[Float](Array(1))))
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), Some(2)))
  }

  it should "be able to express gradient descent" in {
    val numFeatures = 3
    val inputX = Input[Float]("X", Array(None, Some(numFeatures)))
    val numOutputs = 2
    val inputY = Input[Float]("Y", Array(None, Some(numOutputs)))
    val weights = ModelParameter[Float](
      "weights",
      NDArray.arange(Array(numFeatures, numOutputs))
    )
    val biases =
      ModelParameter[Float]("biases", NDArray.ones(Array(numOutputs)))
    val dense = Add(DotProduct(inputX, weights), biases)
    val loss = Square(Subtract(dense, inputY))
    val weightsGradient = loss.gradient(weights)
    val biasesGradient = loss.gradient(biases)
    // The function we are trying to model is f(x) = x0 ^ 2 - x1
    val batchX = NDArray[Float](List(1, 3, 2, 4, 9, 1, 2, 2, 2)).reshape(
      Array(3, numFeatures)
    )
    val batchY = NDArray[Float](List(-2, 7, 2))
    val learningRate = 1e-3f
    val nextStepWeightsGradient =
      weightsGradient.compute(Map(inputX -> batchX, inputY -> batchY))
    assert(nextStepWeightsGradient.isSuccess)
    val nextStepWeightsValue = (weights.value - (NDArray(
      List(learningRate)
    ) * nextStepWeightsGradient.get).get).get
    val nextStepWeights = ModelParameter[Float]("weights", nextStepWeightsValue)
    val nextStepBiasesGradient =
      biasesGradient.compute(Map(inputX -> batchX, inputY -> batchY))
    assert(nextStepBiasesGradient.isSuccess)
    val nextStepBiasesValue = (biases.value - (NDArray(
      List(learningRate)
    ) * nextStepBiasesGradient.get).get).get
    val nextStepBiases = ModelParameter[Float]("biases", nextStepBiasesValue)
    val nextStepDense = Add(DotProduct(inputX, nextStepWeights), nextStepBiases)
    val nextStepLoss = Square(Subtract(nextStepDense, inputY))
    // Compare losses from previous step and next step; loss should decrease.
    val lossOnBatch = loss.compute(Map(inputX -> batchX, inputY -> batchY))
    assert(lossOnBatch.isSuccess)
    val nextStepLossOnBatch =
      nextStepLoss.compute(Map(inputX -> batchX, inputY -> batchY))
    assert(nextStepLossOnBatch.isSuccess)
    val lossOnBatchSum = lossOnBatch.get.sum
    val nextStepLossOnBatchSum = nextStepLossOnBatch.get.sum
    assert(nextStepLossOnBatchSum < lossOnBatchSum)
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
    val placeholderVariable = Input[Float]("X", Array(Some(1)))
    val gradient = constant.gradient(placeholderVariable)
    val output = gradient.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayEquals NDArray.zeros(value.shape))
  }

  it should "return its shape" in {
    val value = NDArray.ones[Int](Array(3))
    val constant = Constant(value)
    val shape = constant.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get.flatten sameElements value.shape)
  }

  "A Variable" should "have gradient 1 with respect to itself" in {
    val modelParameter =
      ModelParameter[Float]("Theta", NDArray.ofValue(Array(2, 3), 5))
    val gradient = modelParameter.gradient(modelParameter)
    val output = gradient.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals NDArray.ones(Array(2, 3)))
  }

  it should "have gradient 0 with respect to other variables" in {
    val modelParameter =
      ModelParameter[Float]("Theta", NDArray.ofValue(Array(2, 3), 5))
    val placeholderVariable = Input[Float]("X", Array(Some(1)))
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

  it should "return its output shape" in {
    val value = NDArray.ofValue[Float](Array(2, 3), 5)
    val modelParameter = ModelParameter[Float]("Theta", value)
    val shape = modelParameter.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get.flatten sameElements value.shape)
  }

  "An Input" should "return the user-supplied value when computed" in {
    val input = Input[Float]("X", Array(Some(2), Some(2)))
    val value = NDArray.ofValue[Float](input.shapeWithPlaceholders.flatten, 4)
    val output = input.compute(Map(input -> value))
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals value)
  }

  it should "accept any value for the placeholder dimension when computed" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value1 = NDArray.ofValue[Float](Array(1, 2), 4)
    val output1 = input.compute(Map(input -> value1))
    assert(output1.isSuccess)
    assert(output1.get arrayApproximatelyEquals value1)
    val value2 = NDArray.ofValue[Float](Array(3, 2), 4)
    val output2 = input.compute(Map(input -> value2))
    assert(output2.isSuccess)
    assert(output2.get arrayApproximatelyEquals value2)
  }

  it should "fail to compute when the user-supplied value does not match the Input shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value = NDArray.ofValue[Float](Array(1, 3), 4)
    val output = input.compute(Map(input -> value))
    assert(output.isFailure)
  }

  it should "fail to compute when the user-supplied value does not have the same number of dimensions as the Input shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value = NDArray.ofValue[Float](Array(2), 4)
    val output = input.compute(Map(input -> value))
    assert(output.isFailure)
  }

  it should "fail to compute when the user does not supply a necessary Input" in {
    val input = Input[Float]("X", Array(Some(2), Some(2)))
    val output = input.compute(Map.empty)
    assert(output.isFailure)
  }

  it should "returns its output shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val shape = input.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements input.shapeWithPlaceholders)
  }

  "An Add" should "return its output shape when its arguments' shapes match" in {
    val addition = Add(
      Constant(NDArray.zeros[Float](Array(2, 4))),
      Constant(NDArray.ones[Float](Array(2, 4)))
    )
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), Some(4)))
  }

  it should "return its output shape when its arguments' shapes can be broadcast" in {
    val addition = Add(
      Constant(NDArray.zeros[Float](Array(4))),
      Constant(NDArray.ones[Float](Array(2, 4)))
    )
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), Some(4)))
  }

  it should "fail to return an output shape when its arguments' shapes mismatch" in {
    val addition = Add(
      Constant(NDArray.zeros[Float](Array(2))),
      Constant(NDArray.ones[Float](Array(2, 4)))
    )
    val shape = addition.getOutputShape
    assert(shape.isFailure)
  }

  it should "return its output shape with placeholder dimensions (None, 1 => None)" in {
    val input1 = Input[Float]("X", Array(None))
    val input2 = Input[Float]("Y", Array(Some(1)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None))
  }

  it should "return its output shape with placeholder dimensions (None x 3, 1 => None x 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(3)))
    val input2 = Input[Float]("Y", Array(Some(1)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, Some(3)))
  }

  it should "return its output shape with placeholder dimensions (None x 1, 3 => None x 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, Some(3)))
  }

  it should "return its output shape with placeholder dimensions (None x 1, 1 x None => None x None)" in {
    val input1 = Input[Float]("X", Array(None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(1), None))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, None))
  }

  it should "return its output shape with placeholder dimensions (None x None x 1, 3 => None x None x 3)" in {
    val input1 = Input[Float]("X", Array(None, None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, None, Some(3)))
  }

  it should "fail to return an output shape with invalid placeholder dimensions (None, 3)" in {
    val input1 = Input[Float]("X", Array(None))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return an output shape with invalid placeholder dimensions (None x 2, 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(2)))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return an output shape with invalid placeholder dimensions (3 x None, 3)" in {
    val input1 = Input[Float]("X", Array(Some(3), None))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return an output shape with invalid placeholder dimensions (None x 3, None x 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(3)))
    val input2 = Input[Float]("Y", Array(None, Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape.isFailure)
  }

  it should "compute the addition of two functions" in {
    val addition = Add(
      Constant(NDArray(List(2, -2, -1, 1)).reshape(Array(2, 2))),
      Constant(NDArray(List(9, 1, 0, 2)).reshape(Array(2, 2)))
    )
    val output = addition.compute(Map.empty)
    assert(output.isSuccess)
    val expected = NDArray(List(11, -1, -1, 3)).reshape(Array(2, 2))
    assert(output.get arrayEquals expected)
  }

  it should "compute the addition of two functions, with broadcasting" in {
    val input1 = Input[Int]("X", Array(None, Some(3)))
    val input2 = Input[Int]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val output = addition.compute(
      Map(
        input1 -> NDArray(List(1, 2, 3, 4, 5, 6)).reshape(Array(2, 3)),
        input2 -> NDArray(List(-2, 4, 3))
      )
    )
    assert(output.isSuccess)
    val expected = NDArray(List(-1, 6, 6, 2, 9, 9)).reshape(Array(2, 3))
    assert(output.get arrayEquals expected)
  }

  it should "fail to compute the addition of two functions with mismatching shapes" in {
    val addition = Add[Int](
      Constant(NDArray.ones(Array(2, 3))),
      Constant(NDArray.ones(Array(2, 2)))
    )
    val output = addition.compute(Map.empty)
    assert(output.isFailure)
  }

  it should "get the gradient of the addition of two constants" in {
    val addition = Add[Float](
      Constant(NDArray[Float](List(2, -2, -1, 1)).reshape(Array(2, 2))),
      Constant(NDArray[Float](List(9, 1, 0, 2)).reshape(Array(2, 2)))
    )
    val gradient = addition.gradient(Input[Float]("X", Array(None)))
    val output = gradient.compute(Map.empty)
    assert(output.isSuccess)
    val expected = NDArray[Float](List(0, 0, 0, 0)).reshape(Array(2, 2))
    assert(output.get arrayApproximatelyEquals expected)
  }

  it should "get the gradient of the addition of two variables" in {
    val inputX = Input[Float]("X", Array(None, Some(3)))
    val inputY = Input[Float]("Y", Array(Some(1)))
    val addition = Add(inputX, inputY)
    val gradientX = addition.gradient(inputX)
    val outputX = gradientX.compute(Map.empty)
    assert(outputX.isSuccess)
    val expectedX = NDArray.ones[Float](Array(1))
    assert(outputX.get arrayApproximatelyEquals expectedX)
    val gradientY = addition.gradient(inputY)
    val outputY = gradientY.compute(Map.empty)
    assert(outputY.isSuccess)
    val expectedY = NDArray.ones[Float](Array(1))
    assert(outputY.get arrayApproximatelyEquals expectedY)
  }

  it should "get the gradient of the addition of two functions using the chain rule" in {
    val inputX = Input[Float]("X", Array(None, Some(3)))
    val inputY = Input[Float]("Y", Array(Some(1)))
    val addition = Add(Square(inputX), inputY)
    val gradientX = addition.gradient(inputX)
    val valueX = NDArray[Float](List(1, -2, 0, 3, 2, 1)).reshape(Array(2, 3))
    val outputX = gradientX.compute(Map(inputX -> valueX))
    assert(outputX.isSuccess)
    val expectedX = (valueX * NDArray(List(2))).get
    assert(outputX.get arrayApproximatelyEquals expectedX)
    val gradientY = addition.gradient(inputY)
    val outputY = gradientY.compute(Map(inputX -> valueX))
    assert(outputY.isSuccess)
    val expectedY = NDArray.ones[Float](Array(1))
    assert(outputY.get arrayApproximatelyEquals expectedY)
  }

  "A DotProduct with 1D arrays (vector inner product)" should "return its output shape (5, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(5))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(1)))
  }

  it should "fail to return an output shape on mismatching arguments (6, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(6))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return an output shape with placeholders (None, 5)" in {
    val input1 = Input[Float]("X", Array(None))
    val input2 = Input[Float]("Y", Array(Some(5)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  it should "compute its output (5, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray[Float](List(1, 2, 3, 4, 5))),
      Constant(NDArray[Float](List(2, -1, 0, 0, 4)))
    )
    val output = dotProduct.compute(Map.empty)
    assert(output.isSuccess)
    assert(output.get arrayApproximatelyEquals NDArray(List(20)))
  }

  it should "fail to compute its output on mismatching arguments (6, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(6))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    val output = dotProduct.compute(Map.empty)
    assert(output.isFailure)
  }

  it should "compute its gradient" in {
    val inputX = Input[Double]("X", Array(Some(5)))
    val inputY = Input[Double]("Y", Array(Some(5)))
    val dotProduct = DotProduct(inputX, inputY)
    val gradientX = dotProduct.gradient(inputX)
    val gradientY = dotProduct.gradient(inputY)
    val valueX = NDArray[Double](List(1, 2, 3, 4, 5))
    val valueY = NDArray[Double](List(2, -1, 0, 0, 4))
    val inputs = Map(inputX -> valueX, inputY -> valueY)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputX, inputs).get
    val numericGradientYOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputY, inputs).get
    // TODO remove debugging
    println(numericGradientXOnInputs)
    println(numericGradientYOnInputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    val gradientYOnInputs = gradientY.compute(inputs)
    assert(gradientXOnInputs.isSuccess)
    assert(
      gradientXOnInputs.get arrayApproximatelyEquals numericGradientXOnInputs
    )
    assert(gradientYOnInputs.isSuccess)
    assert(
      gradientYOnInputs.get arrayApproximatelyEquals numericGradientYOnInputs
    )
  }

  "A DotProduct with 2D arrays (matmul)" should "return its output shape (2 x 4, 4 x 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 4))),
      Constant(NDArray.ones[Float](Array(4, 3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), Some(3)))
  }

  it should "return its output shape with placeholder dimensions (None x 1, 1 x 5)" in {
    val input1 = Input[Float]("X", Array(None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(1), Some(5)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, Some(5)))
  }

  it should "return its output shape with placeholder dimensions (2 x 3, 3 x None)" in {
    val input1 = Input[Float]("X", Array(Some(2), Some(3)))
    val input2 = Input[Float]("Y", Array(Some(3), None))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), None))
  }

  it should "return its output shape with placeholder dimensions (None x 3, 3 x None)" in {
    val input1 = Input[Float]("X", Array(None, Some(3)))
    val input2 = Input[Float]("Y", Array(Some(3), None))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, None))
  }

  it should "fail to return its output shape on mismatching arguments (2 x 3, 4 x 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3))),
      Constant(NDArray.ones[Float](Array(4, 3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return its output shape with placeholders (2 x 3, None x 3)" in {
    val input1 = Input[Float]("X", Array(Some(2), Some(3)))
    val input2 = Input[Float]("Y", Array(None, Some(3)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  "A DotProduct with an N-D array and 1D array (last axis inner product)" should "return its output shape (2 x 3, 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3))),
      Constant(NDArray.ones[Float](Array(3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2)))
  }

  it should "return its output shape (3 x 5 x 2, 2)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(3, 5, 2))),
      Constant(NDArray.ones[Float](Array(2)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(3), Some(5)))
  }

  it should "return its output shape with placeholders (2 x None x 2, 2)" in {
    val input1 = Input[Float]("X", Array(Some(2), None, Some(2)))
    val input2 = Input[Float]("Y", Array(Some(2)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), None))
  }

  it should "fail to return its output shape on mismatching arguments (2 x 5 , 2)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3))),
      Constant(NDArray.ones[Float](Array(2)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return its output shape with placeholders (2 x None, 2)" in {
    val input1 = Input[Float]("X", Array(Some(2), None))
    val input2 = Input[Float]("Y", Array(Some(2)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  "A DotProduct between N-D arrays (multidimensional inner product)" should "return its output shape (2 x 3 x 4, 4 x 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3, 4))),
      Constant(NDArray.ones[Float](Array(4, 5)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), Some(3), Some(5)))
  }

  it should "return its output shape (3 x 5 x 2, 1 x 2 x 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(3, 5, 2))),
      Constant(NDArray.ones[Float](Array(1, 2, 3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(3), Some(5), Some(1), Some(3)))
  }

  it should "return its output shape with placeholders (2 x None x 4, 4 x 5)" in {
    val input1 = Input[Float]("X", Array(Some(2), None, Some(4)))
    val input2 = Input[Float]("Y", Array(Some(4), Some(5)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(Some(2), None, Some(5)))
  }

  it should "return its output shape with placeholders (None x 1 x 4, 2 x 4 x None)" in {
    val input1 = Input[Float]("X", Array(None, Some(1), Some(4)))
    val input2 = Input[Float]("Y", Array(Some(2), Some(4), None))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isSuccess)
    assert(shape.get sameElements Array(None, Some(1), Some(2), None))
  }

  it should "fail to return its output shape on mismatching arguments (2 x 3 x 4, 1 x 4)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3, 4))),
      Constant(NDArray.ones[Float](Array(1, 4)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }

  it should "fail to return its output shape with placeholders (2 x 3 x None, 4 x 5)" in {
    val input1 = Input[Float]("X", Array(Some(2), Some(3), None))
    val input2 = Input[Float]("Y", Array(Some(4), Some(5)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape.isFailure)
  }
}
