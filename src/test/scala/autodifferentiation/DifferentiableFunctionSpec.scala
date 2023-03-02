package autodifferentiation

import exceptions.ShapeException
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.reflect.{ClassTag, classTag}

class DifferentiableFunctionSpec extends AnyFlatSpec with Matchers {

  /** Numerically computes the gradient of the function for the inputs.
    *
    * This function only computes accurate gradients on element-wise operations.
    *
    * If you plan to use this function to check gradients, use Double precision.
    */
  private def computeGradientWithFiniteDifferences[T: ClassTag](
      f: DifferentiableFunction[T],
      withRespectToInput: Input[T],
      inputs: Map[Input[T], NDArray[T]],
      epsilon: Double = 1e-5
  )(implicit num: Fractional[T]): NDArray[T] = {
    val epsilonArray = (classTag[T] match {
      case _ if classTag[T] == classTag[Float] =>
        NDArray[Float](List(epsilon.toFloat))
      case _ if classTag[T] == classTag[Double] =>
        NDArray[Double](List(epsilon))
    }).asInstanceOf[NDArray[T]]
    val inputsMinusEpsilon = inputs.updated(
      withRespectToInput,
      inputs(withRespectToInput) - epsilonArray
    )
    val inputsPlusEpsilon = inputs.updated(
      withRespectToInput,
      inputs(withRespectToInput) + epsilonArray
    )
    val outputMinusEpsilon = f.compute(inputsMinusEpsilon)
    val outputPlusEpsilon = f.compute(inputsPlusEpsilon)
    val difference = outputPlusEpsilon - outputMinusEpsilon
    (classTag[T] match {
      case _ if classTag[T] == classTag[Float] =>
        difference.asInstanceOf[NDArray[Float]] / NDArray[Float](
          List(2 * epsilon.toFloat)
        )
      case _ if classTag[T] == classTag[Double] =>
        difference.asInstanceOf[NDArray[Double]] / NDArray[Double](
          List(2 * epsilon)
        )
    }).asInstanceOf[NDArray[T]]
  }

  /** Numerically computes the gradient of the function for the inputs.
    *
    * This function computes accurate gradients, but is very expensive. It
    * computes the function for every element in the input array.
    *
    * If you plan to use this function to check gradients, use Double precision.
    * This function computes the gradient separately for every element of the
    * input.
    */
  private def computeGradientWithFiniteDifferencesAllElements[T: ClassTag](
      f: DifferentiableFunction[T],
      withRespectToInput: Input[T],
      inputs: Map[Input[T], NDArray[T]],
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
      NDArray[Double](List(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(batchSize, numFeatures)
      )
    val batchY = NDArray[Double](List(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(batchSize, numOutputs)
    )
    val inputs = Map(inputX -> batchX, inputY -> batchY)
    val execution = loss.computeAll(inputs)
    val gradients = loss.backpropagateAll(execution)
    val learningRate = 1e-3
    val nextStepWeightsValue = weights.value - (gradients(weights) * NDArray(
      List(learningRate)
    ))
    val nextStepBiasesValue =
      biases.value - (gradients(biases) * NDArray(List(learningRate)))
    val nextStepWeights =
      ModelParameter[Double]("weights", nextStepWeightsValue)
    val nextStepBiases = ModelParameter[Double]("biases", nextStepBiasesValue)
    val nextStepDense = Add(DotProduct(inputX, nextStepWeights), nextStepBiases)
    val nextStepLoss = Mean(Square(Subtract(nextStepDense, inputY)))
    // Compare losses from previous step and next step; loss should decrease.
    val lossOnBatch = loss.compute(Map(inputX -> batchX, inputY -> batchY))
    val nextStepLossOnBatch =
      nextStepLoss.compute(Map(inputX -> batchX, inputY -> batchY))
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
      NDArray[Double](List(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(batchSize, numFeatures)
      )
    val batchY = NDArray[Double](List(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(batchSize, numOutputs)
    )
    val inputs = Map(
      inputX -> batchX,
      inputY -> batchY,
      weights -> weightsValue,
      biases -> biasesValue
    )
    val execution = loss.computeAll(inputs)
    val gradients = loss.backpropagateAll(execution)
    val numericGradientsWeights =
      computeGradientWithFiniteDifferencesAllElements(
        loss,
        weights,
        inputs
      )
    val numericGradientsBiases =
      computeGradientWithFiniteDifferencesAllElements(
        loss,
        biases,
        inputs
      )
    assert(
      gradients(weights) arrayApproximatelyEquals numericGradientsWeights
    )
    assert(
      gradients(biases) arrayApproximatelyEquals numericGradientsBiases
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
      NDArray[Double](List(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(batchSize, numFeatures)
      )
    val batchY = NDArray[Double](List(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(batchSize, numOutputs)
    )
    val inputs = Map(inputX -> batchX, inputY -> batchY)
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
      val execution = nextStepLoss.computeAll(inputs)
      val gradients = nextStepLoss.backpropagateAll(execution)
      val nextStepWeightsValue =
        nextStepWeights.value - (gradients(nextStepWeights) * NDArray(
          List(learningRate)
        ))
      val nextStepBiasesValue =
        nextStepBiases.value - (gradients(nextStepBiases) * NDArray(
          List(learningRate)
        ))
      nextStepWeights = ModelParameter[Double]("weights", nextStepWeightsValue)
      nextStepBiases = ModelParameter[Double]("biases", nextStepBiasesValue)
      val nextStepDense =
        Add(DotProduct(inputX, nextStepWeights), nextStepBiases)
      nextStepLoss = Mean(Square(Subtract(nextStepDense, inputY)))
    }
    val finalLoss =
      nextStepLoss.compute(Map(inputX -> batchX, inputY -> batchY))
    assert(initialLoss.sum > 200)
    assert(finalLoss.sum < 1)
  }

  "A Constant" should "return its preset value when computed" in {
    val value = NDArray.ones[Int](Array(3))
    val constant = Constant(value)
    val output = constant.compute(Map.empty)
    assert(output arrayEquals value)
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
    assert(output arrayEquals NDArray.zeros(value.shape))
  }

  it should "return its shape" in {
    val value = NDArray.ones[Int](Array(3))
    val constant = Constant(value)
    val shape = constant.getOutputShape
    assert(shape.flatten sameElements value.shape)
  }

  "A Variable" should "have gradient 1 with respect to itself" in {
    val modelParameter =
      ModelParameter[Float]("Theta", NDArray.ofValue(Array(2, 3), 5))
    val gradient = modelParameter.gradient(modelParameter)
    val output = gradient.compute(Map.empty)
    assert(output arrayApproximatelyEquals NDArray.ones(Array(2, 3)))
  }

  it should "have gradient 0 with respect to other variables" in {
    val modelParameter =
      ModelParameter[Float]("Theta", NDArray.ofValue(Array(2, 3), 5))
    val placeholderVariable = Input[Float]("X", Array(Some(1)))
    // Take the gradient of f() = Theta with respect to unrelated variable X.
    val gradient = modelParameter.gradient(placeholderVariable)
    val output = gradient.compute(Map.empty)
    assert(output arrayApproximatelyEquals NDArray.zeros(Array(2, 3)))
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
    val output = input.compute(Map(input -> value))
    assert(output arrayApproximatelyEquals value)
  }

  it should "accept any value for the placeholder dimension when computed" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value1 = NDArray.ofValue[Float](Array(1, 2), 4)
    val output1 = input.compute(Map(input -> value1))
    assert(output1 arrayApproximatelyEquals value1)
    val value2 = NDArray.ofValue[Float](Array(3, 2), 4)
    val output2 = input.compute(Map(input -> value2))
    assert(output2 arrayApproximatelyEquals value2)
  }

  it should "fail to compute when the user-supplied value does not match the Input shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value = NDArray.ofValue[Float](Array(1, 3), 4)
    assertThrows[ShapeException](input.compute(Map(input -> value)))
  }

  it should "fail to compute when the user-supplied value does not have the same number of dimensions as the Input shape" in {
    val input = Input[Float]("X", Array(None, Some(2)))
    val value = NDArray.ofValue[Float](Array(2), 4)
    assertThrows[ShapeException](input.compute(Map(input -> value)))
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
    assert(output arrayApproximatelyEquals NDArray(List(3.5f)))
  }

  it should "compute its gradient" in {
    val inputX = Input[Double]("X", Array(Some(2), Some(4)))
    val mean = Mean(inputX)
    val gradientX = mean.gradient(inputX)
    assert(
      gradientX.getOutputShape sameElements Array(Some(1))
    )
    val valueX = NDArray.arange[Double](Array(2, 4))
    val inputs = Map(inputX -> valueX)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(mean, inputX, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
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
      Constant(NDArray[Int](List(1, 2, 3, 4, 5, 6)).reshape(Array(2, 3)))
    )
    val output = negation.compute(Map.empty)
    val expected =
      NDArray[Int](List(-1, -2, -3, -4, -5, -6)).reshape(Array(2, 3))
    assert(output arrayEquals expected)
  }

  it should "compute its gradient" in {
    val inputX = Input[Int]("X", Array(Some(2), Some(4)))
    val negation = Negate(inputX)
    val gradientX = negation.gradient(inputX)
    assert(
      gradientX.getOutputShape sameElements Array(Some(2), Some(4))
    )
    val valueX = NDArray.arange[Int](Array(2, 4))
    assert(
      gradientX.compute(Map(inputX -> valueX)) arrayEquals NDArray
        .ones[Int](Array(2, 4))
        .negate
    )
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
        NDArray[Double](List(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
      )
    )
    val output = reciprocal.compute(Map.empty)
    val expected = NDArray[Double](
      List(1, 0.5, 0.3333333333333333, 0.25, 0.2, 0.1666666,
        0.14285714285714285, 0.125)
    ).reshape(Array(2, 4))
    assert(output arrayApproximatelyEquals expected)
  }

  it should "compute its gradient" in {
    val inputX = Input[Double]("X", Array(Some(2), Some(4)))
    val reciprocal = Reciprocal(inputX)
    val gradientX = reciprocal.gradient(inputX)
    val valueX =
      NDArray[Double](List(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
    val inputs = Map(inputX -> valueX)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(reciprocal, inputX, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
  }

  it should "compute its gradient with chain rule (1 / -X ^ 2)" in {
    val inputX = Input[Double]("X", Array(Some(2), Some(4)))
    val reciprocal = Reciprocal(Negate(Square(inputX)))
    val gradientX = reciprocal.gradient(inputX)
    val valueX =
      NDArray[Double](List(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
    val inputs = Map(inputX -> valueX)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(reciprocal, inputX, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
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
      Constant(NDArray[Double](List(0, 1, 2, -3, 4)))
    )
    val output = exp.compute(Map.empty)
    val expected = NDArray[Double](
      List(1, Math.exp(1.0), Math.exp(2.0), Math.exp(-3.0), Math.exp(4.0))
    )
    assert(output arrayApproximatelyEquals expected)
  }

  it should "compute its gradient" in {
    val inputX = Input[Double]("X", Array(Some(2), Some(4)))
    val exp = Exp(inputX)
    val gradientX = exp.gradient(inputX)
    val valueX =
      NDArray[Double](List(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
    val inputs = Map(inputX -> valueX)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(exp, inputX, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
    // For exp(X), we can also check that the gradient is the same as computed.
    val computed = exp.compute(inputs)
    assert(computed arrayApproximatelyEquals gradientXOnInputs)
  }

  it should "compute its gradient with placeholders and chain rule (exp(1 / X))" in {
    val inputX = Input[Double]("X", Array(None, Some(4)))
    val exp = Exp(Reciprocal(inputX))
    val gradientX = exp.gradient(inputX)
    val valueX =
      NDArray[Double](List(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
    val inputs = Map(inputX -> valueX)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(exp, inputX, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
  }

  it should "compute its gradient with placeholders and chain rule (sigmoid: 1 / (1 + exp(-X)))" in {
    val inputX = Input[Double]("X", Array(None, Some(4)))
    val exp =
      Reciprocal(Add(Constant(NDArray.ones(Array(1))), Exp(Negate(inputX))))
    val gradientX = exp.gradient(inputX)
    val valueX =
      NDArray[Double](List(4, 6, 1, 4, -2, -3, 9, 0)).reshape(Array(2, 4))
    val inputs = Map(inputX -> valueX)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(exp, inputX, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
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
      Constant(NDArray(List(2, -2, -1, 1)).reshape(Array(2, 2))),
      Constant(NDArray(List(9, 1, 0, 2)).reshape(Array(2, 2)))
    )
    val output = addition.compute(Map.empty)
    val expected = NDArray(List(11, -1, -1, 3)).reshape(Array(2, 2))
    assert(output arrayEquals expected)
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
    val expected = NDArray(List(-1, 6, 6, 2, 9, 9)).reshape(Array(2, 3))
    assert(output arrayEquals expected)
  }

  it should "fail to compute the addition of two functions with mismatching shapes" in {
    val addition = Add[Int](
      Constant(NDArray.ones(Array(2, 3))),
      Constant(NDArray.ones(Array(2, 2)))
    )
    assertThrows[ShapeException](addition.compute(Map.empty))
  }

  it should "get the gradient of the addition of two constants" in {
    val addition = Add[Float](
      Constant(NDArray[Float](List(2, -2, -1, 1)).reshape(Array(2, 2))),
      Constant(NDArray[Float](List(9, 1, 0, 2)).reshape(Array(2, 2)))
    )
    val gradient = addition.gradient(Input[Float]("X", Array(None)))
    val output = gradient.compute(Map.empty)
    val expected = NDArray[Float](List(0, 0, 0, 0)).reshape(Array(2, 2))
    assert(output arrayApproximatelyEquals expected)
  }

  it should "get the gradient of the addition of two variables" in {
    val inputX = Input[Float]("X", Array(None, Some(3)))
    val inputY = Input[Float]("Y", Array(Some(1)))
    val addition = Add(inputX, inputY)
    val gradientX = addition.gradient(inputX)
    val outputX = gradientX.compute(Map.empty)
    val expectedX = NDArray.ones[Float](Array(1))
    assert(outputX arrayApproximatelyEquals expectedX)
    val gradientY = addition.gradient(inputY)
    val outputY = gradientY.compute(Map.empty)
    val expectedY = NDArray.ones[Float](Array(1))
    assert(outputY arrayApproximatelyEquals expectedY)
  }

  it should "get the gradient of the addition of two functions using the chain rule" in {
    val inputX = Input[Float]("X", Array(None, Some(3)))
    val inputY = Input[Float]("Y", Array(Some(1)))
    val addition = Add(Square(inputX), inputY)
    val gradientX = addition.gradient(inputX)
    val valueX = NDArray[Float](List(1, -2, 0, 3, 2, 1)).reshape(Array(2, 3))
    val outputX = gradientX.compute(Map(inputX -> valueX))
    val expectedX = valueX * NDArray(List(2))
    assert(outputX arrayApproximatelyEquals expectedX)
    val gradientY = addition.gradient(inputY)
    val outputY = gradientY.compute(Map(inputX -> valueX))
    val expectedY = NDArray.ones[Float](Array(1))
    assert(outputY arrayApproximatelyEquals expectedY)
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
      Constant(NDArray[Float](List(1, 2, 3, 4, 5))),
      Constant(NDArray[Float](List(2, -1, 0, 0, 4)))
    )
    val output = dotProduct.compute(Map.empty)
    assert(output arrayApproximatelyEquals NDArray(List(20)))
  }

  it should "fail to compute its output on mismatching arguments (6, 5)" in {
    val dotProduct = DotProduct(
      Constant(NDArray.zeros[Float](Array(6))),
      Constant(NDArray.ones[Float](Array(5)))
    )
    assertThrows[ShapeException](dotProduct.compute(Map.empty))
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
      computeGradientWithFiniteDifferences(dotProduct, inputX, inputs)
    val numericGradientYOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputY, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    val gradientYOnInputs = gradientY.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
    assert(
      gradientYOnInputs.shape sameElements numericGradientYOnInputs.shape
    )
    assert(
      gradientYOnInputs arrayApproximatelyEquals numericGradientYOnInputs
    )
  }

  it should "compute its gradient with chain rule (2 * X dot Y)" in {
    val inputX = Input[Double]("X", Array(Some(5)))
    val inputY = Input[Double]("Y", Array(Some(5)))
    val dotProduct =
      DotProduct[Double](Multiply(Constant(NDArray(List(2))), inputX), inputY)
    val gradientX = dotProduct.gradient(inputX)
    val gradientY = dotProduct.gradient(inputY)
    val valueX = NDArray[Double](List(1, 2, 3, 4, 5))
    val valueY = NDArray[Double](List(2, -1, 0, 0, 4))
    val inputs = Map(inputX -> valueX, inputY -> valueY)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputX, inputs)
    val numericGradientYOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputY, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    val gradientYOnInputs = gradientY.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
    assert(
      gradientYOnInputs.shape sameElements numericGradientYOnInputs.shape
    )
    assert(
      gradientYOnInputs arrayApproximatelyEquals numericGradientYOnInputs
    )
  }

  it should "compute its gradient with chain rule (2 * X dot Y ^ 2)" in {
    val inputX = Input[Double]("X", Array(Some(5)))
    val inputY = Input[Double]("Y", Array(Some(5)))
    val dotProduct = DotProduct[Double](
      Multiply(inputX, Constant(NDArray(List(2)))),
      Square(inputY)
    )
    val gradientX = dotProduct.gradient(inputX)
    val gradientY = dotProduct.gradient(inputY)
    val valueX = NDArray[Double](List(1, 2, 3, 4, 5))
    val valueY = NDArray[Double](List(2, -1, 0, 0, 4))
    val inputs = Map(inputX -> valueX, inputY -> valueY)
    val numericGradientXOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputX, inputs)
    val numericGradientYOnInputs =
      computeGradientWithFiniteDifferences(dotProduct, inputY, inputs)
    val gradientXOnInputs = gradientX.compute(inputs)
    val gradientYOnInputs = gradientY.compute(inputs)
    assert(
      gradientXOnInputs.shape sameElements numericGradientXOnInputs.shape
    )
    assert(
      gradientXOnInputs arrayApproximatelyEquals numericGradientXOnInputs
    )
    assert(
      gradientYOnInputs.shape sameElements numericGradientYOnInputs.shape
    )
    assert(
      gradientYOnInputs arrayApproximatelyEquals numericGradientYOnInputs
    )
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
