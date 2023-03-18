package autodifferentiation

import exceptions.ShapeException
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.reflect.{ClassTag, classTag}

class DifferentiableFunctionSpec extends AnyFlatSpec with Matchers {

  /** Numerically computes the gradient of the function for the inputs.
    *
    * This function computes accurate gradients, but is very expensive. It
    * computes the function for every element in the input array.
    *
    * If you plan to use this function to check gradients, use Double precision.
    * This function computes the gradient separately for every element of the
    * input.
    */
  private def computeGradientWithFiniteDifferences[T: ClassTag](
      f: DifferentiableFunction[T],
      withRespectToInput: String,
      inputs: Map[String, NDArray[T]],
      epsilon: Double = 1e-5
  )(implicit num: Fractional[T]): NDArray[T] = {
    val epsilonAsT = (classTag[T] match {
      case _ if classTag[T] == classTag[Float]  => epsilon.toFloat
      case _ if classTag[T] == classTag[Double] => epsilon
    }).asInstanceOf[T]
    val inputArray = inputs(withRespectToInput)
    var differences = NDArray.zeros[T](inputArray.shape)
    inputArray.indices.foreach { idx =>
      val inputsMinusEpsilonAtIndex = inputs.updated(
        withRespectToInput,
        inputArray.updated(idx, num.minus(inputArray(idx), epsilonAsT))
      )
      val inputsPlusEpsilonAtIndex = inputs.updated(
        withRespectToInput,
        inputArray.updated(idx, num.plus(inputArray(idx), epsilonAsT))
      )
      val outputMinusEpsilonAtIndex =
        f.compute(inputsMinusEpsilonAtIndex).sum
      val outputPlusEpsilonAtIndex = f.compute(inputsPlusEpsilonAtIndex).sum
      val difference =
        num.minus(outputPlusEpsilonAtIndex, outputMinusEpsilonAtIndex)
      val slope = num.div(difference, num.times(num.fromInt(2), epsilonAsT))
      differences = differences.updated(idx, slope)
    }
    differences
  }

  "A DifferentiableFunction" should "return its output shape (1)" in {
    val input = Input[Float]("X", Array(Some(1)))
    val addition = Add(input, Constant(NDArray.ones[Float](Array(1))))
    val shape = addition.getOutputShape
    assert(shape sameElements Array(Some(1)))
  }

  it should "return its output shape, with broadcasting (2 x 2)" in {
    val input = Input[Float]("X", Array(Some(2), Some(2)))
    val addition = Add(input, Constant(NDArray.ones[Float](Array(1))))
    val shape = addition.getOutputShape
    assert(shape sameElements Array(Some(2), Some(2)))
  }

  it should "be able to express gradient descent" in {
    val numFeatures = 3
    val inputX = Input[Double]("X", Array(None, Some(numFeatures)))
    val numOutputs = 2
    val inputY = Input[Double]("Y", Array(None, Some(numOutputs)))
    val weights = ModelParameter[Double](
      "weights",
      NDArray.arange(Array(numFeatures, numOutputs))
    )
    val biases =
      ModelParameter[Double]("biases", NDArray.ones(Array(numOutputs)))
    val dense = Add(DotProduct(inputX, weights), biases)
    val loss = Mean(Square(Subtract(dense, inputY)))
    // The function we are trying to model is f(x) = (x0 ^ 2 - x1, 2 * x2)
    val batchSize = 4
    val batchX =
      NDArray[Double](Array(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(batchSize, numFeatures)
      )
    val batchY = NDArray[Double](Array(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(batchSize, numOutputs)
    )
    val inputs = Map("X" -> batchX, "Y" -> batchY)
    val execution = loss.computeAllComponentFunctions(inputs)
    val gradients = loss.backpropagateAllComponentFunctions(execution)
    val learningRate = 1e-3
    val nextStepWeightsValue = weights.value - (gradients.get(weights) * NDArray(
      Array(learningRate)
    ))
    val nextStepBiasesValue =
      biases.value - (gradients.get(biases) * NDArray(Array(learningRate)))
    val nextStepWeights =
      ModelParameter[Double]("weights", nextStepWeightsValue)
    val nextStepBiases = ModelParameter[Double]("biases", nextStepBiasesValue)
    val nextStepDense = Add(DotProduct(inputX, nextStepWeights), nextStepBiases)
    val nextStepLoss = Mean(Square(Subtract(nextStepDense, inputY)))
    // Compare losses from previous step and next step; loss should decrease.
    val lossOnBatch = loss.compute(inputs)
    val nextStepLossOnBatch =
      nextStepLoss.compute(inputs)
    val lossOnBatchSum = lossOnBatch.sum
    val nextStepLossOnBatchSum = nextStepLossOnBatch.sum
    assert(nextStepLossOnBatchSum < lossOnBatchSum)
  }

  it should "have numerically accurate gradients in gradient descent" in {
    val numFeatures = 3
    val inputX = Input[Double]("X", Array(None, Some(numFeatures)))
    val numOutputs = 2
    val inputY = Input[Double]("Y", Array(None, Some(numOutputs)))
    // We make weights and biases Inputs rather than ModelParameters here so
    // that we can perturb them to get the numerical gradient.
    val weightsValue = NDArray.arange[Double](Array(numFeatures, numOutputs))
    val weights = Input[Double]("weights", weightsValue.shape.map(Some(_)))
    val biasesValue = NDArray.ones[Double](Array(numOutputs))
    val biases = Input[Double]("biases", biasesValue.shape.map(Some(_)))
    val dense = Add(DotProduct(inputX, weights), biases)
    val loss = Mean(Square(Subtract(dense, inputY)))
    // The function we are trying to model is f(x) = (x0 ^ 2 - x1, 2 * x2)
    val batchSize = 4
    val batchX =
      NDArray[Double](Array(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(batchSize, numFeatures)
      )
    val batchY = NDArray[Double](Array(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(batchSize, numOutputs)
    )
    val inputs = Map(
      "X" -> batchX,
      "Y" -> batchY,
      "weights" -> weightsValue,
      "biases" -> biasesValue
    )
    val execution = loss.computeAllComponentFunctions(inputs)
    val gradients = loss.backpropagateAllComponentFunctions(execution)
    val numericGradientsWeights =
      computeGradientWithFiniteDifferences(
        loss,
        weights.name,
        inputs
      )
    val numericGradientsBiases =
      computeGradientWithFiniteDifferences(
        loss,
        biases.name,
        inputs
      )
    assert(
      gradients.get(weights) arrayApproximatelyEquals numericGradientsWeights
    )
    assert(
      gradients.get(biases) arrayApproximatelyEquals numericGradientsBiases
    )
  }

  // We ignore this test because it is slow.
  ignore should "be able to drive the training loss arbitrarily close to 0 in gradient descent" in {
    val numFeatures = 3
    val inputX = Input[Double]("X", Array(None, Some(numFeatures)))
    val numOutputs = 2
    val inputY = Input[Double]("Y", Array(None, Some(numOutputs)))
    val initialWeights = ModelParameter[Double](
      "weights",
      NDArray.arange(Array(numFeatures, numOutputs))
    )
    val initialBiases =
      ModelParameter[Double]("biases", NDArray.ones(Array(numOutputs)))
    val dense = Add(DotProduct(inputX, initialWeights), initialBiases)
    val loss = Mean(Square(Subtract(dense, inputY)))
    // The function we are trying to model is f(x) = (x0 ^ 2 - x1, 2 * x2)
    val batchSize = 4
    val batchX =
      NDArray[Double](Array(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(batchSize, numFeatures)
      )
    val batchY = NDArray[Double](Array(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(batchSize, numOutputs)
    )
    val inputs = Map("X" -> batchX, "Y" -> batchY)
    val initialLoss = loss.compute(inputs)
    val learningRate = 1e-2
    // To keep this test fast, we use a small number of steps. You can get
    // arbitrarily low loss by running for longer (at 50k steps, the loss is
    // about 1e-17).
    val numSteps = 500
    var nextStepWeights = initialWeights
    var nextStepBiases = initialBiases
    var nextStepLoss = loss
    (0 until numSteps).foreach { _ =>
      val execution = nextStepLoss.computeAllComponentFunctions(inputs)
      val gradients = nextStepLoss.backpropagateAllComponentFunctions(execution)
      val nextStepWeightsValue =
        nextStepWeights.value - (gradients.get(nextStepWeights) * NDArray(
          Array(learningRate)
        ))
      val nextStepBiasesValue =
        nextStepBiases.value - (gradients.get(nextStepBiases) * NDArray(
          Array(learningRate)
        ))
      nextStepWeights = ModelParameter[Double]("weights", nextStepWeightsValue)
      nextStepBiases = ModelParameter[Double]("biases", nextStepBiasesValue)
      val nextStepDense =
        Add(DotProduct(inputX, nextStepWeights), nextStepBiases)
      nextStepLoss = Mean(Square(Subtract(nextStepDense, inputY)))
    }
    val finalLoss =
      nextStepLoss.compute(inputs)
    assert(initialLoss.sum > 200)
    assert(finalLoss.sum < 1)
  }

  "A Constant" should "return its preset value when computed" in {
    val value = NDArray.ones[Int](Array(3))
    val constant = Constant(value)
    val output = constant.compute(Map.empty)
    assert(output arrayEquals value)
  }

  it should "return its shape" in {
    val value = NDArray.ones[Int](Array(3))
    val constant = Constant(value)
    val shape = constant.getOutputShape
    assert(shape.flatten sameElements value.shape)
  }

  "A ModelParameter" should "return its current value when computed" in {
    val value = NDArray.ofValue[Float](Array(2, 3), 5)
    val modelParameter = ModelParameter[Float]("Theta", value)
    val output = modelParameter.compute(Map.empty)
    assert(output arrayApproximatelyEquals value)
  }

  it should "return its output shape" in {
    val value = NDArray.ofValue[Float](Array(2, 3), 5)
    val modelParameter = ModelParameter[Float]("Theta", value)
    val shape = modelParameter.getOutputShape
    assert(shape.flatten sameElements value.shape)
  }

  "An Input" should "return the user-supplied value when computed" in {
    val input = Input[Float]("X", Array(Some(2), Some(2)))
    val value = NDArray.ofValue[Float](input.shapeWithPlaceholders.flatten, 4)
    val output = input.compute(Map("X" -> value))
    assert(output arrayApproximatelyEquals value)
  }

  it should "accept any value for the placeholder dimension when computed" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value1 = NDArray.ofValue[Float](Array(1, 2), 4)
    val output1 = input.compute(Map("X" -> value1))
    assert(output1 arrayApproximatelyEquals value1)
    val value2 = NDArray.ofValue[Float](Array(3, 2), 4)
    val output2 = input.compute(Map("X" -> value2))
    assert(output2 arrayApproximatelyEquals value2)
  }

  it should "fail to compute when the user-supplied value does not match the Input shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value = NDArray.ofValue[Float](Array(1, 3), 4)
    assertThrows[ShapeException](input.compute(Map("X" -> value)))
  }

  it should "fail to compute when the user-supplied value does not have the same number of dimensions as the Input shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value = NDArray.ofValue[Float](Array(2), 4)
    assertThrows[ShapeException](input.compute(Map("X" -> value)))
  }

  it should "fail to compute when the user does not supply a necessary Input" in {
    val input = Input[Float]("X", Array(Some(2), Some(2)))
    assertThrows[NoSuchElementException](input.compute(Map.empty))
  }

  it should "returns its output shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val shape = input.getOutputShape
    assert(shape sameElements input.shapeWithPlaceholders)
  }

  "A Mean" should "return its output shape" in {
    val mean = Mean(
      Constant(NDArray.arange[Float](Array(2, 4)))
    )
    val shape = mean.getOutputShape
    assert(shape sameElements Array(Some(1)))
  }

  it should "compute the mean of all elements" in {
    val mean = Mean(
      Constant(NDArray.arange[Float](Array(2, 4)))
    )
    val output = mean.compute(Map.empty)
    assert(output arrayApproximatelyEquals NDArray(Array(3.5f)))
  }

  "A Negate" should "return its output shape" in {
    val negation = Negate(
      Constant(NDArray.arange[Float](Array(2, 4)))
    )
    val shape = negation.getOutputShape
    assert(shape sameElements Array(Some(2), Some(4)))
  }

  it should "compute the negation of all elements" in {
    val negation = Negate(
      Constant(NDArray[Int](Array(1, 2, 3, 4, 5, 6)).reshape(Array(2, 3)))
    )
    val output = negation.compute(Map.empty)
    val expected =
      NDArray[Int](Array(-1, -2, -3, -4, -5, -6)).reshape(Array(2, 3))
    assert(output arrayEquals expected)
  }

  "A Reciprocal" should "return its output shape" in {
    val reciprocal = Reciprocal(
      Constant(NDArray.arange[Float](Array(2, 4)))
    )
    val shape = reciprocal.getOutputShape
    assert(shape sameElements Array(Some(2), Some(4)))
  }

  it should "return its output shape with placeholder dimensions" in {
    val input = Input[Float]("X", Array(None, Some(2), None, Some(4)))
    val reciprocal = Reciprocal(input)
    val shape = reciprocal.getOutputShape
    assert(shape sameElements Array(None, Some(2), None, Some(4)))
  }

  it should "compute the reciprocal of all elements" in {
    val reciprocal = Reciprocal(
      Constant(
        NDArray[Double](Array(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
      )
    )
    val output = reciprocal.compute(Map.empty)
    val expected = NDArray[Double](
      Array(1, 0.5, 0.3333333333333333, 0.25, 0.2, 0.1666666,
        0.14285714285714285, 0.125)
    ).reshape(Array(2, 4))
    assert(output arrayApproximatelyEquals expected)
  }

  "An Exp" should "return its output shape" in {
    val exp = Exp(
      Constant(NDArray.arange[Float](Array(2, 4)))
    )
    val shape = exp.getOutputShape
    assert(shape sameElements Array(Some(2), Some(4)))
  }

  it should "compute the exponentiation of all elements" in {
    val exp = Exp(
      Constant(NDArray[Double](Array(0, 1, 2, -3, 4)))
    )
    val output = exp.compute(Map.empty)
    val expected = NDArray[Double](
      Array(1, Math.exp(1.0), Math.exp(2.0), Math.exp(-3.0), Math.exp(4.0))
    )
    assert(output arrayApproximatelyEquals expected)
  }

  "An Add" should "return its output shape when its arguments' shapes match" in {
    val addition = Add(
      Constant(NDArray.zeros[Float](Array(2, 4))),
      Constant(NDArray.ones[Float](Array(2, 4)))
    )
    val shape = addition.getOutputShape
    assert(shape sameElements Array(Some(2), Some(4)))
  }

  it should "return its output shape when its arguments' shapes can be broadcast" in {
    val addition = Add(
      Constant(NDArray.zeros[Float](Array(4))),
      Constant(NDArray.ones[Float](Array(2, 4)))
    )
    val shape = addition.getOutputShape
    assert(shape sameElements Array(Some(2), Some(4)))
  }

  it should "fail to return an output shape when its arguments' shapes mismatch" in {
    val addition = Add(
      Constant(NDArray.zeros[Float](Array(2))),
      Constant(NDArray.ones[Float](Array(2, 4)))
    )
    assertThrows[ShapeException](addition.getOutputShape)
  }

  it should "return its output shape with placeholder dimensions (None, 1 => None)" in {
    val input1 = Input[Float]("X", Array(None))
    val input2 = Input[Float]("Y", Array(Some(1)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape sameElements Array(None))
  }

  it should "return its output shape with placeholder dimensions (None x 3, 1 => None x 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(3)))
    val input2 = Input[Float]("Y", Array(Some(1)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape sameElements Array(None, Some(3)))
  }

  it should "return its output shape with placeholder dimensions (None x 1, 3 => None x 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape sameElements Array(None, Some(3)))
  }

  it should "return its output shape with placeholder dimensions (None x 1, 1 x None => None x None)" in {
    val input1 = Input[Float]("X", Array(None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(1), None))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape sameElements Array(None, None))
  }

  it should "return its output shape with placeholder dimensions (None x None x 1, 3 => None x None x 3)" in {
    val input1 = Input[Float]("X", Array(None, None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape sameElements Array(None, None, Some(3)))
  }

  it should "fail to return an output shape with invalid placeholder dimensions (None, 3)" in {
    val input1 = Input[Float]("X", Array(None))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    assertThrows[ShapeException](addition.getOutputShape)
  }

  it should "fail to return an output shape with invalid placeholder dimensions (None x 2, 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(2)))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    assertThrows[ShapeException](addition.getOutputShape)
  }

  it should "fail to return an output shape with invalid placeholder dimensions (3 x None, 3)" in {
    val input1 = Input[Float]("X", Array(Some(3), None))
    val input2 = Input[Float]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    assertThrows[ShapeException](addition.getOutputShape)
  }

  it should "return its output shape with matching placeholder dimensions (None x 3, None x 3)" in {
    val input1 = Input[Float]("X", Array(None, Some(3)))
    val input2 = Input[Float]("Y", Array(None, Some(3)))
    val addition = Add(input1, input2)
    val shape = addition.getOutputShape
    assert(shape sameElements Array(None, Some(3)))
  }

  it should "compute the addition of two functions" in {
    val addition = Add(
      Constant(NDArray(Array(2, -2, -1, 1)).reshape(Array(2, 2))),
      Constant(NDArray(Array(9, 1, 0, 2)).reshape(Array(2, 2)))
    )
    val output = addition.compute(Map.empty)
    val expected = NDArray(Array(11, -1, -1, 3)).reshape(Array(2, 2))
    assert(output arrayEquals expected)
  }

  it should "compute the addition of two functions, with broadcasting" in {
    val input1 = Input[Int]("X", Array(None, Some(3)))
    val input2 = Input[Int]("Y", Array(Some(3)))
    val addition = Add(input1, input2)
    val output = addition.compute(
      Map(
        "X" -> NDArray(Array(1, 2, 3, 4, 5, 6)).reshape(Array(2, 3)),
        "Y" -> NDArray(Array(-2, 4, 3))
      )
    )
    val expected = NDArray(Array(-1, 6, 6, 2, 9, 9)).reshape(Array(2, 3))
    assert(output arrayEquals expected)
  }

  it should "fail to compute the addition of two functions with mismatching shapes" in {
    val addition = Add[Int](
      Constant(NDArray.ones(Array(2, 3))),
      Constant(NDArray.ones(Array(2, 2)))
    )
    assertThrows[ShapeException](addition.compute(Map.empty))
  }

  "A DotProduct with 1D arrays (vector inner product)" should "return its output shape (5, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(5))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(1)))
  }

  it should "fail to return an output shape on mismatching arguments (6, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(6))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  it should "fail to return an output shape with placeholders (None, 5)" in {
    val input1 = Input[Float]("X", Array(None))
    val input2 = Input[Float]("Y", Array(Some(5)))
    val dotProduct = DotProduct(input1, input2)
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  it should "compute its output (5, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray[Float](Array(1, 2, 3, 4, 5))),
      Constant(NDArray[Float](Array(2, -1, 0, 0, 4)))
    )
    val output = dotProduct.compute(Map.empty)
    assert(output arrayApproximatelyEquals NDArray(Array(20)))
  }

  it should "fail to compute its output on mismatching arguments (6, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(6))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    assertThrows[ShapeException](dotProduct.compute(Map.empty))
  }

  "A DotProduct with 2D arrays (matmul)" should "return its output shape (2 x 4, 4 x 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 4))),
      Constant(NDArray.ones[Float](Array(4, 3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(2), Some(3)))
  }

  it should "return its output shape with placeholder dimensions (None x 1, 1 x 5)" in {
    val input1 = Input[Float]("X", Array(None, Some(1)))
    val input2 = Input[Float]("Y", Array(Some(1), Some(5)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(None, Some(5)))
  }

  it should "return its output shape with placeholder dimensions (2 x 3, 3 x None)" in {
    val input1 = Input[Float]("X", Array(Some(2), Some(3)))
    val input2 = Input[Float]("Y", Array(Some(3), None))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(2), None))
  }

  it should "return its output shape with placeholder dimensions (None x 3, 3 x None)" in {
    val input1 = Input[Float]("X", Array(None, Some(3)))
    val input2 = Input[Float]("Y", Array(Some(3), None))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(None, None))
  }

  it should "fail to return its output shape on mismatching arguments (2 x 3, 4 x 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3))),
      Constant(NDArray.ones[Float](Array(4, 3)))
    )
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  it should "fail to return its output shape with placeholders (2 x 3, None x 3)" in {
    val input1 = Input[Float]("X", Array(Some(2), Some(3)))
    val input2 = Input[Float]("Y", Array(None, Some(3)))
    val dotProduct = DotProduct(input1, input2)
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  "A DotProduct with an N-D array and 1D array (last axis inner product)" should "return its output shape (2 x 3, 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3))),
      Constant(NDArray.ones[Float](Array(3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(2)))
  }

  it should "return its output shape (3 x 5 x 2, 2)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(3, 5, 2))),
      Constant(NDArray.ones[Float](Array(2)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(3), Some(5)))
  }

  it should "return its output shape with placeholders (2 x None x 2, 2)" in {
    val input1 = Input[Float]("X", Array(Some(2), None, Some(2)))
    val input2 = Input[Float]("Y", Array(Some(2)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(2), None))
  }

  it should "fail to return its output shape on mismatching arguments (2 x 5 , 2)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3))),
      Constant(NDArray.ones[Float](Array(2)))
    )
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  it should "fail to return its output shape with placeholders (2 x None, 2)" in {
    val input1 = Input[Float]("X", Array(Some(2), None))
    val input2 = Input[Float]("Y", Array(Some(2)))
    val dotProduct = DotProduct(input1, input2)
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  "A DotProduct between N-D arrays (multidimensional inner product)" should "return its output shape (2 x 3 x 4, 4 x 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3, 4))),
      Constant(NDArray.ones[Float](Array(4, 5)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(2), Some(3), Some(5)))
  }

  it should "return its output shape (3 x 5 x 2, 1 x 2 x 3)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(3, 5, 2))),
      Constant(NDArray.ones[Float](Array(1, 2, 3)))
    )
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(3), Some(5), Some(1), Some(3)))
  }

  it should "return its output shape with placeholders (2 x None x 4, 4 x 5)" in {
    val input1 = Input[Float]("X", Array(Some(2), None, Some(4)))
    val input2 = Input[Float]("Y", Array(Some(4), Some(5)))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(Some(2), None, Some(5)))
  }

  it should "return its output shape with placeholders (None x 1 x 4, 2 x 4 x None)" in {
    val input1 = Input[Float]("X", Array(None, Some(1), Some(4)))
    val input2 = Input[Float]("Y", Array(Some(2), Some(4), None))
    val dotProduct = DotProduct(input1, input2)
    val shape = dotProduct.getOutputShape
    assert(shape sameElements Array(None, Some(1), Some(2), None))
  }

  it should "fail to return its output shape on mismatching arguments (2 x 3 x 4, 1 x 4)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(2, 3, 4))),
      Constant(NDArray.ones[Float](Array(1, 4)))
    )
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }

  it should "fail to return its output shape with placeholders (2 x 3 x None, 4 x 5)" in {
    val input1 = Input[Float]("X", Array(Some(2), Some(3), None))
    val input2 = Input[Float]("Y", Array(Some(4), Some(5)))
    val dotProduct = DotProduct(input1, input2)
    assertThrows[ShapeException](dotProduct.getOutputShape)
  }
}
