package autodifferentiation

import exceptions.ShapeException
import ndarray.NDArray

import scala.reflect.ClassTag

/** A function that is differentiable with respect to its variables.
  *
  * Used to define a compute graph for neural networks.
  *
  * @tparam T
  *   The array element type.
  */
trait DifferentiableFunction[T] {
  implicit val classTag: ClassTag[T]

  /** Returns the output of the function for the given input values.
    *
    * @param inputs
    *   A Map of `Input` objects to tensors of arbitrary shape.
    */
  def compute(inputs: Map[Input[T], NDArray[T]]): NDArray[T] = computeAll(
    inputs
  ).outputs(this)

  /** Returns the output of every component function for the given input values.
    *
    * For example, if this function is Mean(Square(Input(X))), then the result
    * will contain entries for the output of Mean, Square, and Input(X). The
    * value associated with Input(X) will be the same as that supplied by the
    * user.
    *
    * @param inputs
    *   A Map of `Input` objects to tensors of arbitrary shape.
    */
  def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T]

  /** Returns the gradient for a given argument based on the output gradient.
    *
    * This method is the analogue of defvjp() in the autograd Python library. It
    * defines the Vector-Jacobian Products that pass gradient information from
    * the function output to each of its inputs.
    *
    * @param execution
    *   The results of function execution on particular inputs. Used to retrieve
    *   the inputs and outputs of DifferentiableFunctions throughout
    *   backpropagation.
    * @param outputGradient
    *   The gradient of the final output function (often the loss function) with
    *   respect to this function. If this function is the final output function,
    *   then the gradient should be 1.
    * @param withRespectToArg
    *   The index of the argument with respect to which to compute the input
    *   gradient, starting at 0. For example, if this function is Subtract(a,
    *   b), supplying 0 would result in outputGradient and 1 would result in
    *   outputGradient * -1.
    */
  def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T]

  /** Returns the gradient with respect to all component functions.
    *
    * @param execution
    *   The results of function execution on particular inputs. Used to retrieve
    *   the inputs and outputs of DifferentiableFunctions throughout
    *   backpropagation.
    * @param num
    *   The implicit numeric conversion.
    * @return
    *   The gradient of this function with respect to all component functions.
    *   The gradient of this function with respect to itself is 1. Gradients
    *   always match the shape of the function with which they correspond.
    */
  def backpropagateAll(
      execution: DifferentiableFunctionExecution[T]
  )(implicit
      num: Numeric[T]
  ): Map[DifferentiableFunction[T], NDArray[T]] = {
    val lastStepGradient = NDArray.ones[T](execution.outputs(this).shape)
    backpropagateAllRecursive(execution, lastStepGradient)
  }

  private def backpropagateAllRecursive(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T]
  ): Map[DifferentiableFunction[T], NDArray[T]] = {
    val parents = getParents
    val parentGradients =
      parents.indices.map(idx => backpropagate(execution, outputGradient, idx))
    val parentBackpropagationResults = parents.indices.map(idx =>
      parents(idx)
        .backpropagateAllRecursive(execution, parentGradients(idx))
    )
    Map(this -> outputGradient) ++ parentBackpropagationResults.flatten
  }

  /** Returns the gradient of the function with respect to a variable.
    *
    * @param withRespectToVariable
    *   The variable with which to compute the gradient. If we call this
    *   DifferentiableFunction y and the variable x, this operation produces
    *   dy/dx.
    */
  def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T]

  /** Returns the set of all inputs to the function. */
  def getInputs: Set[Input[T]]

  /** Returns the output shape of the function with possible placeholders. */
  def getOutputShape: Array[Option[Int]]

  /** Returns the DifferentiableFunctions on which this function depends. */
  def getParents: List[DifferentiableFunction[T]]
}

/** Contains the results of function execution on particular inputs.
  *
  * Used during backpropagation.
  *
  * @param inputs
  *   A Map of `Input` objects to tensors of arbitrary shape.
  * @param outputs
  *   The output of every component function for the given input values.
  * @tparam T
  *   The array element type.
  */
case class DifferentiableFunctionExecution[T](
    inputs: Map[Input[T], NDArray[T]],
    outputs: Map[DifferentiableFunction[T], NDArray[T]]
)

/** A constant (has 0 gradient).
  *
  * @param value
  *   The constant's value.
  * @tparam T
  *   The array element type.
  */
case class Constant[T](value: NDArray[T])(
    override implicit val classTag: ClassTag[T]
) extends DifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] =
    DifferentiableFunctionExecution(inputs, Map(this -> value))

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) NDArray.zeros(value.shape)
    else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape))

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape: Array[Option[Int]] = value.shape.map(Some(_))

  override def getParents: List[DifferentiableFunction[T]] = List.empty
}

/** A variable (has potentially non-zero gradient).
  *
  * @tparam T
  *   The array element type.
  */
trait Variable[T] extends DifferentiableFunction[T] {
  val name: String
  implicit val classTag: ClassTag[T]

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) {
      val varExecution = execution.outputs(this)
      NDArray.ones(varExecution.shape)
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    if (withRespectToVariable == this) Constant(NDArray.ones(getGradientShape))
    else Constant(NDArray.zeros(getGradientShape))

  /** Returns the shape of the returned gradient. */
  private def getGradientShape: Array[Int] = getOutputShape.map {
    case Some(dimension) => dimension
    case None            => 1
  }
}

/** A model parameter.
  *
  * @param name
  *   The name of the variable.
  * @param value
  *   The current value of the parameter.
  * @tparam T
  *   The array element type.
  */
case class ModelParameter[T](
    override val name: String,
    value: NDArray[T]
)(override implicit val classTag: ClassTag[T])
    extends Variable[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] =
    DifferentiableFunctionExecution(inputs, Map(this -> value))

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape: Array[Option[Int]] = value.shape.map(Some(_))

  override def getParents: List[DifferentiableFunction[T]] = List.empty
}

/** An input variable that users supply.
  *
  * Passes user-defined values to the computation graph.
  *
  * @param name
  *   The name of the variable.
  * @param shapeWithPlaceholders
  *   The shape of the array containing the variable, with None for dimensions
  *   that can vary at run time (e.g., the batch dimension).
  * @tparam T
  *   The array element type.
  */
case class Input[T](
    override val name: String,
    shapeWithPlaceholders: Array[Option[Int]]
)(override implicit val classTag: ClassTag[T])
    extends Variable[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = inputs(this)
    if (
      value.shape.length == shapeWithPlaceholders.length &&
      shapeWithPlaceholders.indices.forall(idx =>
        shapeWithPlaceholders(idx) match {
          case Some(dimension) => value.shape(idx) == dimension
          case _               => true
        }
      )
    ) DifferentiableFunctionExecution(inputs, Map(this -> value))
    else
      throw new ShapeException(
        s"Input $name expects values of shape ${shapeWithPlaceholders
            .mkString("Array(", ", ", ")")}, but got ${value.shape
            .mkString("Array(", ", ", ")")}"
      )
  }

  override def getInputs: Set[Input[T]] = Set(this)

  override def getOutputShape: Array[Option[Int]] = shapeWithPlaceholders

  override def getParents: List[DifferentiableFunction[T]] = List.empty
}

/** Sums the results of a function.
  *
  * @param a
  *   The function to sum.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Sum[T](a: DifferentiableFunction[T])(
    implicit num: Numeric[T],
    override implicit val classTag: ClassTag[T]
) extends DifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = a.computeAll(inputs)
    value.copy(outputs =
      value.outputs + (this -> NDArray(Array(value.outputs(a).sum)))
    )
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (!outputGradient.shape.sameElements(Array(1)))
      throw new ShapeException(
        s"Expected output gradient to have shape 1, but found ${outputGradient.shape
            .mkString("Array(", ", ", ")")}"
      )
    else if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val outputGradientOnlyElement = outputGradient.flatten().head
      NDArray.ofValue(aExecution.shape, outputGradientOnlyElement)
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = ???

  override def getInputs: Set[Input[T]] = ???

  override def getOutputShape: Array[Option[Int]] = {
    // We need to get a's output shape to make sure there are no errors.
    val _ = a.getOutputShape
    Array(Some(1))
  }

  override def getParents: List[DifferentiableFunction[T]] = List(a)
}

/** Computes the mean of the results of a function.
  *
  * @param a
  *   The function to sum.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Mean[T](a: DifferentiableFunction[T])(
    implicit num: Fractional[T],
    override implicit val classTag: ClassTag[T]
) extends DifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = a.computeAll(inputs)
    value.copy(outputs =
      value.outputs + (this -> NDArray(Array(value.outputs(a).mean)))
    )
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (!outputGradient.shape.sameElements(Array(1)))
      throw new ShapeException(
        s"Expected output gradient to have shape 1, but found ${outputGradient.shape
            .mkString("Array(", ", ", ")")}"
      )
    else if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val numRepetitions = aExecution.shape.product
      val outputGradientOnlyElement = outputGradient.flatten().head
      val scaledOutputGradientOnlyElement =
        num.div(outputGradientOnlyElement, num.fromInt(numRepetitions))
      NDArray.ofValue(aExecution.shape, scaledOutputGradientOnlyElement)
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = Mean(a.gradient(withRespectToVariable))

  override def getInputs: Set[Input[T]] = a.getInputs

  override def getOutputShape: Array[Option[Int]] = {
    // We need to get a's output shape to make sure there are no errors.
    val _ = a.getOutputShape
    Array(Some(1))
  }

  override def getParents: List[DifferentiableFunction[T]] = List(a)
}

/** A differentiable function with one arguments that operates on all elements.
  *
  * This function's output shape is the same shape as its input.
  *
  * @tparam T
  *   The array element type.
  */
trait UnaryElementWiseDifferentiableFunction[T]
    extends DifferentiableFunction[T] {
  val a: DifferentiableFunction[T]

  override def getOutputShape: Array[Option[Int]] = a.getOutputShape

  override def getParents: List[DifferentiableFunction[T]] = List(a)

  override def getInputs: Set[Input[T]] = a.getInputs
}

/** Negates the results of a function.
  *
  * @param a
  *   The function to negate.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Negate[T](override val a: DifferentiableFunction[T])(
    implicit num: Numeric[T],
    override implicit val classTag: ClassTag[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = a.computeAll(inputs)
    value.copy(outputs = value.outputs + (this -> value.outputs(a).negate))
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) outputGradient.negate
    else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = Negate(a.gradient(withRespectToVariable))
}

/** Squares the results of a function.
  *
  * @param a
  *   The function to square.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Square[T](override val a: DifferentiableFunction[T])(
    implicit num: Numeric[T],
    override implicit val classTag: ClassTag[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = a.computeAll(inputs)
    value.copy(outputs = value.outputs + (this -> value.outputs(a).square))
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val aExecutionDouble = aExecution * NDArray(Array(num.fromInt(2)))
      aExecutionDouble * outputGradient
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Multiply(
      Multiply(Constant(NDArray[T](Array(num.fromInt(2)))), a),
      a.gradient(withRespectToVariable)
    )
}

/** Returns the reciprocal of the results of a function.
  *
  * @param a
  *   The function to invert.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Reciprocal[T](override val a: DifferentiableFunction[T])(
    implicit num: Fractional[T],
    override implicit val classTag: ClassTag[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = a.computeAll(inputs)
    value.copy(outputs = value.outputs + (this -> value.outputs(a).reciprocal))
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val aExecutionSquared = aExecution.square
      outputGradient.negate / aExecutionSquared
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Multiply(Reciprocal(Negate(Square(a))), a.gradient(withRespectToVariable))
}

/** Returns the exponentiation of the results of a function (f(x) = pow(e, x)).
  *
  * @param a
  *   The function to exponentiate.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Exp[T](override val a: DifferentiableFunction[T])(
    implicit num: Fractional[T],
    override implicit val classTag: ClassTag[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val value = a.computeAll(inputs)
    value.copy(outputs = value.outputs + (this -> value.outputs(a).exp))
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      aExecution * outputGradient
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Multiply(this, a.gradient(withRespectToVariable))
}

/** A differentiable function with two arguments that broadcasts its operations.
  *
  * This function broadcasts the results of its operations together based on the
  * NDArray broadcasting rules. Its output shape is the result of this
  * broadcast.
  *
  * @tparam T
  *   The array element type.
  */
trait BinaryDifferentiableFunctionWithBroadcast[T]
    extends DifferentiableFunction[T] {
  val a: DifferentiableFunction[T]
  val b: DifferentiableFunction[T]

  override def getOutputShape: Array[Option[Int]] =
    getBroadcastShapeWithPlaceholders(a.getOutputShape, b.getOutputShape)

  private def getBroadcastShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Array[Option[Int]] = {
    val finalNumDimensions = aShape.length max bShape.length
    val aOnesPaddedShape =
      aShape.reverse.padTo(finalNumDimensions, Some(1)).reverse
    val bOnesPaddedShape =
      bShape.reverse.padTo(finalNumDimensions, Some(1)).reverse
    val shapesMatch = (0 until finalNumDimensions).forall(idx =>
      (aOnesPaddedShape(idx) == bOnesPaddedShape(idx)) ||
        aOnesPaddedShape(idx).contains(1) ||
        bOnesPaddedShape(idx).contains(1)
    )
    if (shapesMatch)
      (0 until finalNumDimensions)
        .map(idx =>
          (aOnesPaddedShape(idx), bOnesPaddedShape(idx)) match {
            case (Some(aDimension), Some(bDimension)) =>
              Some(aDimension max bDimension)
            case _ => None
          }
        )
        .toArray
    else
      throw new ShapeException(s"Could not broadcast arrays of shape ${aShape
          .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}")
  }

  override def getParents: List[DifferentiableFunction[T]] = List(a, b)

  /** Removes broadcasted dimensions by summing along them.
    *
    * @param targetShape
    *   The shape to which to transform the gradient through unbroadcasting.
    * @param outputGradient
    *   The gradient of the final output function (often the loss function) with
    *   respect to this function. The shape of this array is the shape of the
    *   output of this DifferentiableFunction, which may have required
    *   broadcasting of inputs.
    * @return
    *   outputGradient unbroadcasted to targetShape.
    */
  protected def unbroadcast(
      targetShape: Array[Int],
      outputGradient: NDArray[T]
  )(implicit num: Numeric[T]): NDArray[T] = {
    var unbroadcastGradient = outputGradient
    while (unbroadcastGradient.shape.length > targetShape.length) {
      unbroadcastGradient = unbroadcastGradient.sumAxis(0)
    }
    targetShape.indices.foreach(idx =>
      if (targetShape(idx) == 1)
        unbroadcastGradient = unbroadcastGradient.sumAxis(idx, keepDims = true)
    )
    unbroadcastGradient
  }

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs
}

/** Adds the results of two functions.
  *
  * @param a
  *   The left hand side.
  * @param b
  *   The right hand side.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Add[T](
    override val a: DifferentiableFunction[T],
    override val b: DifferentiableFunction[T]
)(implicit num: Numeric[T], override implicit val classTag: ClassTag[T])
    extends BinaryDifferentiableFunctionWithBroadcast[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val aValue = a.computeAll(inputs)
    val bValue = b.computeAll(inputs)
    val result = aValue.outputs(a) + bValue.outputs(b)
    DifferentiableFunctionExecution(
      inputs,
      aValue.outputs ++ bValue.outputs + (this -> result)
    )
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0)
      unbroadcast(execution.outputs(a).shape, outputGradient)
    else if (withRespectToArg == 1)
      unbroadcast(execution.outputs(b).shape, outputGradient)
    else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Add(a.gradient(withRespectToVariable), b.gradient(withRespectToVariable))
}

/** Subtracts the results of two functions.
  *
  * @param a
  *   The left hand side.
  * @param b
  *   The right hand side.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Subtract[T](
    override val a: DifferentiableFunction[T],
    override val b: DifferentiableFunction[T]
)(implicit num: Numeric[T], override implicit val classTag: ClassTag[T])
    extends BinaryDifferentiableFunctionWithBroadcast[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val aValue = a.computeAll(inputs)
    val bValue = b.computeAll(inputs)
    val result = aValue.outputs(a) - bValue.outputs(b)
    DifferentiableFunctionExecution(
      inputs,
      aValue.outputs ++ bValue.outputs + (this -> result)
    )
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0)
      unbroadcast(execution.outputs(a).shape, outputGradient)
    else if (withRespectToArg == 1)
      unbroadcast(execution.outputs(b).shape, outputGradient.negate)
    else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Subtract(
      a.gradient(withRespectToVariable),
      b.gradient(withRespectToVariable)
    )
}

/** Element-wise multiplies the results of two functions.
  *
  * @param a
  *   The left hand side.
  * @param b
  *   The right hand side.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Multiply[T](
    override val a: DifferentiableFunction[T],
    override val b: DifferentiableFunction[T]
)(implicit num: Numeric[T], override implicit val classTag: ClassTag[T])
    extends BinaryDifferentiableFunctionWithBroadcast[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val aValue = a.computeAll(inputs)
    val bValue = b.computeAll(inputs)
    val result = aValue.outputs(a) * bValue.outputs(b)
    DifferentiableFunctionExecution(
      inputs,
      aValue.outputs ++ bValue.outputs + (this -> result)
    )
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) {
      val bExecution = execution.outputs(b)
      unbroadcast(
        execution.outputs(a).shape,
        outputGradient * bExecution
      )
    } else if (withRespectToArg == 1) {
      val aExecution = execution.outputs(a)
      unbroadcast(
        execution.outputs(b).shape,
        outputGradient * aExecution
      )
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Add(
      Multiply(a.gradient(withRespectToVariable), b),
      Multiply(a, b.gradient(withRespectToVariable))
    )
}

/** Computes the dot product of the results of two functions.
  *
  * @param a
  *   The left hand side.
  * @param b
  *   The right hand side.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class DotProduct[T](
    a: DifferentiableFunction[T],
    b: DifferentiableFunction[T]
)(implicit num: Numeric[T], override implicit val classTag: ClassTag[T])
    extends DifferentiableFunction[T] {
  override def computeAll(
      inputs: Map[Input[T], NDArray[T]]
  ): DifferentiableFunctionExecution[T] = {
    val aValue = a.computeAll(inputs)
    val bValue = b.computeAll(inputs)
    val result = aValue.outputs(a) dot bValue.outputs(b)
    DifferentiableFunctionExecution(
      inputs,
      aValue.outputs ++ bValue.outputs + (this -> result)
    )
  }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): NDArray[T] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val bExecution = execution.outputs(b)
      if (aExecution.shape.length == 1 && bExecution.shape.length == 1)
        bExecution * outputGradient
      else if (aExecution.shape.length == 2 && bExecution.shape.length == 2)
        outputGradient dot bExecution.transpose
      else ???
    } else if (withRespectToArg == 1) {
      val aExecution = execution.outputs(a)
      val bExecution = execution.outputs(b)
      if (aExecution.shape.length == 1 && bExecution.shape.length == 1)
        aExecution * outputGradient
      else if (aExecution.shape.length == 2 && bExecution.shape.length == 2)
        aExecution.transpose dot outputGradient
      else ???
    } else
      throw new IllegalArgumentException(
        s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    gradientFromShapes(
      a.getOutputShape,
      b.getOutputShape,
      withRespectToVariable
    )

  private def gradientFromShapes(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]],
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    if (aShape.length == 1 && bShape.length == 1)
      vectorInnerProductGradient(aShape, bShape, withRespectToVariable)
    else if (aShape.length == 2 && bShape.length == 2)
      matmulGradient(aShape, bShape, withRespectToVariable)
    else ???

  /** Returns the shape of the dot product on two 1D arrays. */
  private def vectorInnerProductGradient(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]],
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = {
    val aVectorLength = aShape.head
    val bVectorLength = bShape.head
    if (aVectorLength.isEmpty || bVectorLength.isEmpty)
      throw new ShapeException(
        "Cannot get the vector inner product gradient with placeholder dimensions"
      )
    else if (aVectorLength.get != bVectorLength.get)
      throw new ShapeException(
        s"Arrays must have matching shape for vector inner product, but found ${aShape
            .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
      )
    else
      gradientUnbroadcastZeros(
        a.gradient(withRespectToVariable),
        b.gradient(withRespectToVariable)
      )
  }

  private def gradientUnbroadcastZeros(
      aGradient: DifferentiableFunction[T],
      bGradient: DifferentiableFunction[T]
  ): DifferentiableFunction[T] = {
    val omitAGradient = aGradient match {
      case Constant(value) if value arrayEquals NDArray.zeros(Array(1)) => true
      case _                                                            => false
    }
    val omitBGradient = bGradient match {
      case Constant(value) if value arrayEquals NDArray.zeros(Array(1)) => true
      case _                                                            => false
    }
    if (omitAGradient && omitBGradient)
      Constant(NDArray.zeros(Array(1)))
    else if (omitAGradient) DotProduct(a, bGradient)
    else if (omitBGradient) DotProduct(aGradient, b)
    else Add(DotProduct(aGradient, b), DotProduct(a, bGradient))
  }

  /** Returns the shape of the dot product on two 2D arrays. */
  private def matmulGradient(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]],
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = {
    val j1 = aShape.tail.head
    val j2 = bShape.head
    if (j1.isEmpty || j2.isEmpty)
      throw new ShapeException(
        "Cannot get matmul gradient with placeholder in middle dimension"
      )
    else if (j1.get != j2.get)
      throw new ShapeException(
        s"Arrays must have matching middle dimension for matmul, but found ${aShape
            .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
      )
    else
      Add(
        DotProduct(a.gradient(withRespectToVariable), b),
        DotProduct(a, b.gradient(withRespectToVariable))
      )
  }

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs

  override def getOutputShape: Array[Option[Int]] =
    getDotProductShapeWithPlaceholders(a.getOutputShape, b.getOutputShape)

  private def getDotProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Array[Option[Int]] =
    if (aShape.length == 1 && bShape.length == 1)
      getVectorInnerProductShapeWithPlaceholders(aShape, bShape)
    else if (aShape.length == 2 && bShape.length == 2)
      getMatMulShapeWithPlaceholders(aShape, bShape)
    else if (bShape.length == 1)
      getLastAxisInnerProductShapeWithPlaceholders(aShape, bShape)
    else if (aShape.length > 1 && bShape.length > 1)
      getMultidimensionalInnerProductShapeWithPlaceholders(aShape, bShape)
    else
      throw new ShapeException(
        s"dot undefined for shapes ${aShape.mkString("Array(", ", ", ")")} and ${bShape
            .mkString("Array(", ", ", ")")}"
      )

  /** Returns the shape of the dot product on two 1D arrays. */
  private def getVectorInnerProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Array[Option[Int]] = {
    val aVectorLength = aShape.head
    val bVectorLength = bShape.head
    if (aVectorLength.isEmpty || bVectorLength.isEmpty)
      throw new ShapeException(
        "Cannot get the vector inner product shape with placeholder dimensions"
      )
    else if (aVectorLength.get != bVectorLength.get)
      throw new ShapeException(
        s"Arrays must have matching shape for vector inner product, but found ${aShape
            .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
      )
    else Array(Some(1))
  }

  /** Returns the shape of the dot product on two 2D arrays. */
  private def getMatMulShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Array[Option[Int]] = {
    val i = aShape.head
    val j1 = aShape.tail.head
    val j2 = bShape.head
    val k = bShape.tail.head
    if (j1.isEmpty || j2.isEmpty)
      throw new ShapeException(
        "Cannot get matmul shape with placeholder in middle dimension"
      )
    else if (j1.get != j2.get)
      throw new ShapeException(
        s"Arrays must have matching middle dimension for matmul, but found ${aShape
            .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
      )
    else Array(i, k)
  }

  /** Returns the shape of the dot product on an N-D array and 1D array. */
  private def getLastAxisInnerProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Array[Option[Int]] =
    if (aShape.last.isEmpty || bShape.head.isEmpty)
      throw new ShapeException(
        "Cannot get last axis inner product shape with placeholder in last dimension"
      )
    else if (aShape.last.get != bShape.head.get)
      throw new ShapeException(
        s"Arrays must have matching last dimension for last axis inner product, but found ${aShape
            .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
      )
    else aShape.dropRight(1)

  /** Returns the shape of the dot product between N-D arrays. */
  private def getMultidimensionalInnerProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Array[Option[Int]] =
    if (aShape.last.isEmpty || bShape(bShape.length - 2).isEmpty)
      throw new ShapeException(
        "Cannot get multidimensional inner product shape with placeholder in match dimension"
      )
    else if (aShape.last.get != bShape(bShape.length - 2).get)
      throw new ShapeException(
        s"${aShape.mkString("Array(", ", ", ")")} last dimension and ${bShape
            .mkString("Array(", ", ", ")")} second to last dimension must match for multidimensional inner product"
      )
    else aShape.dropRight(1) ++ bShape.dropRight(2) :+ bShape.last

  override def getParents: List[DifferentiableFunction[T]] = List(a, b)
}
