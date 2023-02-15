package autodifferentiation

import exceptions.ShapeException
import ndarray.NDArray

import scala.reflect.{ClassTag, classTag}
import scala.util.{Failure, Success, Try}

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
  def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = computeAll(
    inputs
  ) match {
    case Success(execution) => Success(execution.outputs(this))
    case Failure(failure)   => Failure(failure)
  }

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
  ): Try[DifferentiableFunctionExecution[T]]

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
    * -outputGradient.
    */
  def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]]

  // TODO docstring
  // TODO returns a map of all component functions to their gradients. We can then just filter all the ModelParameters and update them.
  def backpropagateAll(
      execution: DifferentiableFunctionExecution[T]
  )(implicit
      num: Numeric[T]
  ): Try[Map[DifferentiableFunction[T], NDArray[T]]] = {
    val lastStepGradient = NDArray.ones[T](execution.outputs(this).shape)
    backpropagateAllRecursive(execution, lastStepGradient)
  }

  private def backpropagateAllRecursive(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T]
  ): Try[Map[DifferentiableFunction[T], NDArray[T]]] = {
    val parents = getParents
    val parentGradients = parents.indices.map(idx =>
      backpropagate(execution, outputGradient, idx).get
    )
    val parentBackpropagationResults = parents.indices.map(idx =>
      parents(idx)
        .backpropagateAllRecursive(execution, parentGradients(idx))
        .get
    )
    Success(Map(this -> outputGradient) ++ parentBackpropagationResults.flatten)
  }

  // TODO define gradient in terms of backpropagation--not super important because we only need first derivative for neural nets. But nice feature for release 2.
  /** Returns the gradient of the function with respect to a variable.
    *
    * @param withRespectToVariable
    *   The variable with which to compute the gradient. If we call this
    *   DifferentiableFunction y and the variable x, this operation produces
    *   dy/dx.
    */
  def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]]

  /** Returns the set of all inputs to the function. */
  def getInputs: Set[Input[T]]

  /** Returns the output shape of the function with possible placeholders. */
  def getOutputShape: Try[Array[Option[Int]]]

  // TODO remove this function
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
  ): Try[DifferentiableFunctionExecution[T]] =
    Success(DifferentiableFunctionExecution(inputs, Map(this -> value)))

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0) Success(NDArray.zeros(value.shape))
    else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = Success(
    Constant(NDArray.zeros(value.shape))
  )

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape: Try[Array[Option[Int]]] = Success(
    value.shape.map(Some(_))
  )

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
  ): Try[NDArray[T]] = {
    if (withRespectToArg == 0) {
      val varExecution = execution.outputs(this)
      Success(NDArray.ones(varExecution.shape))
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )
  }

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = getGradientShape match {
    case Success(shape) =>
      if (withRespectToVariable == this) Success(Constant(NDArray.ones(shape)))
      else Success(Constant(NDArray.zeros(shape)))
    case Failure(failure) => Failure(failure)
  }

  /** Returns the shape of the returned gradient. */
  private def getGradientShape: Try[Array[Int]] = getOutputShape match {
    case Success(shape) =>
      Success(shape.map {
        case Some(dimension) => dimension
        case None            => 1
      })
    case Failure(failure) => Failure(failure)
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
  ): Try[DifferentiableFunctionExecution[T]] =
    Success(DifferentiableFunctionExecution(inputs, Map(this -> value)))

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape: Try[Array[Option[Int]]] = Success(
    value.shape.map(Some(_))
  )

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
  ): Try[DifferentiableFunctionExecution[T]] =
    inputs.get(this) match {
      case Some(value) =>
        if (
          value.shape.length == shapeWithPlaceholders.length &&
          shapeWithPlaceholders.indices.forall(idx =>
            shapeWithPlaceholders(idx) match {
              case Some(dimension) => value.shape(idx) == dimension
              case _               => true
            }
          )
        ) Success(DifferentiableFunctionExecution(inputs, Map(this -> value)))
        else
          Failure(
            new ShapeException(
              s"Input $name expects values of shape ${shapeWithPlaceholders
                  .mkString("Array(", ", ", ")")}, but got ${value.shape
                  .mkString("Array(", ", ", ")")}"
            )
          )
      case None =>
        Failure(new NoSuchElementException(s"Input $name is not defined"))
    }

  override def getInputs: Set[Input[T]] = Set(this)

  override def getOutputShape: Try[Array[Option[Int]]] = Success(
    shapeWithPlaceholders
  )

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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) =>
        Success(
          value.copy(outputs =
            value.outputs + (this -> NDArray(List(value.outputs(a).sum)))
          )
        )
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (!outputGradient.shape.sameElements(Array(1)))
      Failure(
        new ShapeException(
          s"Expected output gradient to have shape 1, but found ${outputGradient.shape
              .mkString("Array(", ", ", ")")}"
        )
      )
    else if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val outputGradientOnlyElement = outputGradient.flatten().head
      Success(NDArray.ofValue(aExecution.shape, outputGradientOnlyElement))
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = ???

  override def getInputs: Set[Input[T]] = ???

  override def getOutputShape: Try[Array[Option[Int]]] =
    a.getOutputShape match {
      case Success(_) => Success(Array(Some(1)))
      case failure    => failure
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) =>
        Success(
          value.copy(outputs =
            value.outputs + (this -> NDArray(List(value.outputs(a).mean)))
          )
        )
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (!outputGradient.shape.sameElements(Array(1)))
      Failure(
        new ShapeException(
          s"Expected output gradient to have shape 1, but found ${outputGradient.shape
              .mkString("Array(", ", ", ")")}"
        )
      )
    else if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val numRepetitions = aExecution.shape.product
      val outputGradientOnlyElement = outputGradient.flatten().head
      val scaledOutputGradientOnlyElement =
        num.div(outputGradientOnlyElement, num.fromInt(numRepetitions))
      Success(
        NDArray.ofValue(aExecution.shape, scaledOutputGradientOnlyElement)
      )
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] =
    a.gradient(withRespectToVariable) match {
      case Success(aGradient) => Success(Mean(aGradient))
      case failure            => failure
    }

  override def getInputs: Set[Input[T]] = ???

  override def getOutputShape: Try[Array[Option[Int]]] =
    a.getOutputShape match {
      case Success(_) => Success(Array(Some(1)))
      case failure    => failure
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

  override def getOutputShape: Try[Array[Option[Int]]] = a.getOutputShape

  override def getParents: List[DifferentiableFunction[T]] = List(a)
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) =>
        Success(
          value.copy(outputs =
            value.outputs + (this -> value.outputs(a).negate)
          )
        )
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0) Success(outputGradient.negate)
    else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = a.gradient(withRespectToVariable) match {
    case Success(aGradient) => Success(Negate(aGradient))
    case failure            => failure
  }

  override def getInputs: Set[Input[T]] = ???
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) =>
        Success(
          value.copy(outputs =
            value.outputs + (this -> value.outputs(a).square)
          )
        )
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val aExecutionDouble = (aExecution * NDArray(List(num.fromInt(2)))).get
      aExecutionDouble * outputGradient
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = a.gradient(withRespectToVariable) match {
    case Success(aGradient) =>
      Success(
        Multiply(
          Multiply(Constant(NDArray[T](List(num.fromInt(2)))), a),
          aGradient
        )
      )
    case failure => failure
  }

  override def getInputs: Set[Input[T]] = ???
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) =>
        Success(
          value.copy(outputs =
            value.outputs + (this -> value.outputs(a).reciprocal)
          )
        )
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      val aExecutionSquared = aExecution.square
      outputGradient.negate / aExecutionSquared
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] =
    a.gradient(withRespectToVariable) match {
      case Success(aGradient) =>
        Success(Multiply(Reciprocal(Negate(Square(a))), aGradient))
      case failure => failure
    }

  override def getInputs: Set[Input[T]] = ???
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) =>
        Success(
          value.copy(outputs = value.outputs + (this -> value.outputs(a).exp))
        )
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0) {
      val aExecution = execution.outputs(a)
      aExecution * outputGradient
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] =
    a.gradient(withRespectToVariable) match {
      case Success(aGradient) => Success(Multiply(this, aGradient))
      case failure            => failure
    }

  override def getInputs: Set[Input[T]] = ???
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

  override def getOutputShape: Try[Array[Option[Int]]] =
    a.getOutputShape match {
      case Success(aShape) =>
        b.getOutputShape match {
          case Success(bShape) =>
            getBroadcastShapeWithPlaceholders(aShape, bShape)
          case failure => failure
        }
      case failure => failure
    }

  private def getBroadcastShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Try[Array[Option[Int]]] = {
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
      Success(
        (0 until finalNumDimensions)
          .map(idx =>
            (aOnesPaddedShape(idx), bOnesPaddedShape(idx)) match {
              case (Some(aDimension), Some(bDimension)) =>
                Some(aDimension max bDimension)
              case _ => None
            }
          )
          .toArray
      )
    else
      Failure(
        new ShapeException(s"Could not broadcast arrays of shape ${aShape
            .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}")
      )
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(aValue) =>
        b.computeAll(inputs) match {
          case Success(bValue) =>
            aValue.outputs(a) + bValue.outputs(b) match {
              case Success(result) =>
                Success(
                  DifferentiableFunctionExecution(
                    inputs,
                    aValue.outputs ++ bValue.outputs + (this -> result)
                  )
                )
              case Failure(failure) => Failure(failure)
            }
          case failure => failure
        }
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0)
      Success(unbroadcast(execution.outputs(a).shape, outputGradient))
    else if (withRespectToArg == 1)
      Success(unbroadcast(execution.outputs(b).shape, outputGradient))
    else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = a.gradient(withRespectToVariable) match {
    case Success(aGradient) =>
      b.gradient(withRespectToVariable) match {
        case Success(bGradient) => Success(Add(aGradient, bGradient))
        case failure            => failure
      }
    case failure => failure
  }

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(aValue) =>
        b.computeAll(inputs) match {
          case Success(bValue) =>
            aValue.outputs(a) - bValue.outputs(b) match {
              case Success(result) =>
                Success(
                  DifferentiableFunctionExecution(
                    inputs,
                    aValue.outputs ++ bValue.outputs + (this -> result)
                  )
                )
              case Failure(failure) => Failure(failure)
            }
          case failure => failure
        }
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0)
      Success(unbroadcast(execution.outputs(a).shape, outputGradient))
    else if (withRespectToArg == 1)
      Success(unbroadcast(execution.outputs(b).shape, outputGradient.negate))
    else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = a.gradient(withRespectToVariable) match {
    case Success(aGradient) =>
      b.gradient(withRespectToVariable) match {
        case Success(bGradient) => Success(Subtract(aGradient, bGradient))
        case failure            => failure
      }
    case failure => failure
  }

  override def getInputs: Set[Input[T]] = ???
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(aValue) =>
        b.computeAll(inputs) match {
          case Success(bValue) =>
            aValue.outputs(a) * bValue.outputs(b) match {
              case Success(result) =>
                Success(
                  DifferentiableFunctionExecution(
                    inputs,
                    aValue.outputs ++ bValue.outputs + (this -> result)
                  )
                )
              case Failure(failure) => Failure(failure)
            }
          case failure => failure
        }
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
    if (withRespectToArg == 0) {
      val bExecution = execution.outputs(b)
      Success(
        unbroadcast(
          execution.outputs(a).shape,
          (outputGradient * bExecution).get
        )
      )
    } else if (withRespectToArg == 1) {
      val aExecution = execution.outputs(a)
      Success(
        unbroadcast(
          execution.outputs(b).shape,
          (outputGradient * aExecution).get
        )
      )
    } else
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] =
    a.gradient(withRespectToVariable) match {
      case Success(aGradient) =>
        b.gradient(withRespectToVariable) match {
          case Success(bGradient) =>
            Success(Add(Multiply(aGradient, b), Multiply(a, bGradient)))
          case failure => failure
        }
      case failure => failure
    }

  override def getInputs: Set[Input[T]] = ???
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
  ): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(aValue) =>
        b.computeAll(inputs) match {
          case Success(bValue) =>
            aValue.outputs(a) dot bValue.outputs(b) match {
              case Success(result) =>
                Success(
                  DifferentiableFunctionExecution(
                    inputs,
                    aValue.outputs ++ bValue.outputs + (this -> result)
                  )
                )
              case Failure(failure) => Failure(failure)
            }
          case failure => failure
        }
      case failure => failure
    }

  override def backpropagate(
      execution: DifferentiableFunctionExecution[T],
      outputGradient: NDArray[T],
      withRespectToArg: Int
  ): Try[NDArray[T]] =
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
      Failure(
        new IllegalArgumentException(
          s"withRespectToArg was $withRespectToArg, but the valid choices are {0, 1}."
        )
      )

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = a.getOutputShape match {
    case Success(aShape) =>
      b.getOutputShape match {
        case Success(bShape) =>
          gradientFromShapes(aShape, bShape, withRespectToVariable)
        case Failure(failure) => Failure(failure)
      }
    case Failure(failure) => Failure(failure)
  }

  private def gradientFromShapes(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]],
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] =
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
  ): Try[DifferentiableFunction[T]] = {
    val aVectorLength = aShape.head
    val bVectorLength = bShape.head
    if (aVectorLength.isEmpty || bVectorLength.isEmpty)
      Failure(
        new ShapeException(
          "Cannot get the vector inner product gradient with placeholder dimensions"
        )
      )
    else if (aVectorLength.get != bVectorLength.get)
      Failure(
        new ShapeException(
          s"Arrays must have matching shape for vector inner product, but found ${aShape
              .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
        )
      )
    else
      a.gradient(withRespectToVariable) match {
        case Success(aGradient) =>
          b.gradient(withRespectToVariable) match {
            case Success(bGradient) =>
              gradientUnbroadcastZeros(aGradient, bGradient)
            case failure => failure
          }
        case failure => failure
      }
  }

  private def gradientUnbroadcastZeros(
      aGradient: DifferentiableFunction[T],
      bGradient: DifferentiableFunction[T]
  ): Try[DifferentiableFunction[T]] = {
    val omitAGradient = aGradient match {
      case Constant(value) if value arrayEquals NDArray.zeros(Array(1)) => true
      case _                                                            => false
    }
    val omitBGradient = bGradient match {
      case Constant(value) if value arrayEquals NDArray.zeros(Array(1)) => true
      case _                                                            => false
    }
    if (omitAGradient && omitBGradient)
      Success(Constant(NDArray.zeros(Array(1))))
    else if (omitAGradient) Success(DotProduct(a, bGradient))
    else if (omitBGradient) Success(DotProduct(aGradient, b))
    else Success(Add(DotProduct(aGradient, b), DotProduct(a, bGradient)))
  }

  /** Returns the shape of the dot product on two 2D arrays. */
  private def matmulGradient(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]],
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = {
    val j1 = aShape.tail.head
    val j2 = bShape.head
    if (j1.isEmpty || j2.isEmpty)
      Failure(
        new ShapeException(
          "Cannot get matmul gradient with placeholder in middle dimension"
        )
      )
    else if (j1.get != j2.get)
      Failure(
        new ShapeException(
          s"Arrays must have matching middle dimension for matmul, but found ${aShape
              .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
        )
      )
    else
      a.gradient(withRespectToVariable) match {
        case Success(aGradient) =>
          b.gradient(withRespectToVariable) match {
            case Success(bGradient) =>
              Success(Add(DotProduct(aGradient, b), DotProduct(a, bGradient)))
            case failure => failure
          }
        case failure => failure
      }
  }

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs

  override def getOutputShape: Try[Array[Option[Int]]] =
    a.getOutputShape match {
      case Success(aShape) =>
        b.getOutputShape match {
          case Success(bShape) =>
            getDotProductShapeWithPlaceholders(aShape, bShape)
          case failure => failure
        }
      case failure => failure
    }

  private def getDotProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Try[Array[Option[Int]]] =
    if (aShape.length == 1 && bShape.length == 1)
      getVectorInnerProductShapeWithPlaceholders(aShape, bShape)
    else if (aShape.length == 2 && bShape.length == 2)
      getMatMulShapeWithPlaceholders(aShape, bShape)
    else if (bShape.length == 1)
      getLastAxisInnerProductShapeWithPlaceholders(aShape, bShape)
    else if (aShape.length > 1 && bShape.length > 1)
      getMultidimensionalInnerProductShapeWithPlaceholders(aShape, bShape)
    else
      Failure(
        new ShapeException(
          s"dot undefined for shapes ${aShape.mkString("Array(", ", ", ")")} and ${bShape
              .mkString("Array(", ", ", ")")}"
        )
      )

  /** Returns the shape of the dot product on two 1D arrays. */
  private def getVectorInnerProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Try[Array[Option[Int]]] = {
    val aVectorLength = aShape.head
    val bVectorLength = bShape.head
    if (aVectorLength.isEmpty || bVectorLength.isEmpty)
      Failure(
        new ShapeException(
          "Cannot get the vector inner product shape with placeholder dimensions"
        )
      )
    else if (aVectorLength.get != bVectorLength.get)
      Failure(
        new ShapeException(
          s"Arrays must have matching shape for vector inner product, but found ${aShape
              .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
        )
      )
    else Success(Array(Some(1)))
  }

  /** Returns the shape of the dot product on two 2D arrays. */
  private def getMatMulShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Try[Array[Option[Int]]] = {
    val i = aShape.head
    val j1 = aShape.tail.head
    val j2 = bShape.head
    val k = bShape.tail.head
    if (j1.isEmpty || j2.isEmpty)
      Failure(
        new ShapeException(
          "Cannot get matmul shape with placeholder in middle dimension"
        )
      )
    else if (j1.get != j2.get)
      Failure(
        new ShapeException(
          s"Arrays must have matching middle dimension for matmul, but found ${aShape
              .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
        )
      )
    else Success(Array(i, k))
  }

  /** Returns the shape of the dot product on an N-D array and 1D array. */
  private def getLastAxisInnerProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Try[Array[Option[Int]]] =
    if (aShape.last.isEmpty || bShape.head.isEmpty)
      Failure(
        new ShapeException(
          "Cannot get last axis inner product shape with placeholder in last dimension"
        )
      )
    else if (aShape.last.get != bShape.head.get)
      Failure(
        new ShapeException(
          s"Arrays must have matching last dimension for last axis inner product, but found ${aShape
              .mkString("Array(", ", ", ")")} and ${bShape.mkString("Array(", ", ", ")")}"
        )
      )
    else Success(aShape.dropRight(1))

  /** Returns the shape of the dot product between N-D arrays. */
  private def getMultidimensionalInnerProductShapeWithPlaceholders(
      aShape: Array[Option[Int]],
      bShape: Array[Option[Int]]
  ): Try[Array[Option[Int]]] =
    if (aShape.last.isEmpty || bShape(bShape.length - 2).isEmpty)
      Failure(
        new ShapeException(
          "Cannot get multidimensional inner product shape with placeholder in match dimension"
        )
      )
    else if (aShape.last.get != bShape(bShape.length - 2).get)
      Failure(
        new ShapeException(
          s"${aShape.mkString("Array(", ", ", ")")} last dimension and ${bShape
              .mkString("Array(", ", ", ")")} second to last dimension must match for multidimensional inner product"
        )
      )
    else Success(aShape.dropRight(1) ++ bShape.dropRight(2) :+ bShape.last)

  override def getParents: List[DifferentiableFunction[T]] = List(a, b)
}
