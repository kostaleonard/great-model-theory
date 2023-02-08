package autodifferentiation

import exceptions.ShapeException
import ndarray.NDArray

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

/** A function that is differentiable with respect to its variables.
  *
  * Used to define a compute graph for neural networks.
  *
  * @tparam T
  *   The array element type.
  */
trait DifferentiableFunction[T] {

  /** Returns the output of the function for the given input values.
    *
    * @param inputs
    *   A Map of `Input` objects to tensors of arbitrary shape.
    */
  def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = computeAll(inputs) match {
    case Success(execution) => Success(execution.outputs(this))
    case Failure(failure) => Failure(failure)
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
  def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]]

  /** Returns the gradient for a given argument based on the output gradient.
    *
    * This method is the analogue of defvjp() in the autograd Python library. It
    * defines the Vector-Jacobian Products that pass gradient information from
    * the function output to each of its inputs.
    *
    * @param outputGradient
    *   The gradient of the final output function (often the loss function) with
    *   respect to this function. If this function is the final output function,
    *   then the gradient should be 1.
    * @param computeResult
    *   The result of this function's computation on its inputs.
    * @param withRespectToArg
    *   The index of the argument with respect to which to compute the input
    *   gradient, starting at 0. For example, if this function is
    *   Subtract(a, b), supplying 0 would result in outputGradient and 1 would
    *   result in -outputGradient.
    */
  def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]]

  //TODO docstring
  //TODO returns a map of all component functions to their gradients. We can then just filter all the ModelParameters and update them.
  def backpropagateAll(execution: DifferentiableFunctionExecution[T]): Map[DifferentiableFunction[T], NDArray[T]] = ???

  //TODO define gradient in terms of backpropagation--not super important because we only need first derivative for neural nets. But nice feature for release 2.
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
case class DifferentiableFunctionExecution[T](inputs: Map[Input[T], NDArray[T]], outputs: Map[DifferentiableFunction[T], NDArray[T]])

/** A constant (has 0 gradient).
  *
  * @param value
  *   The constant's value.
  * @tparam T
  *   The array element type.
  */
case class Constant[T: ClassTag](value: NDArray[T])
    extends DifferentiableFunction[T] {
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] =
    Success(DifferentiableFunctionExecution(inputs, Map(this -> value)))

  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = {
    if(withRespectToArg == 0) Success(NDArray.zeros(value.shape))
    else Failure(new IllegalArgumentException(s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."))
  }

  override def gradient(
      withRespectToVariable: Variable[T]
  ): Try[DifferentiableFunction[T]] = Success(
    Constant(NDArray.zeros(value.shape))
  )

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape: Try[Array[Option[Int]]] = Success(
    value.shape.map(Some(_))
  )
}

/** A variable (has potentially non-zero gradient).
  *
  * @tparam T
  *   The array element type.
  */
trait Variable[T] extends DifferentiableFunction[T] {
  val name: String
  implicit val classTag: ClassTag[T]

  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = {
    if(withRespectToArg == 0) getGradientShape match {
      case Success(shape) => Success(NDArray.ones(shape))
      case Failure(failure) => Failure(failure)
    }
    else Failure(new IllegalArgumentException(s"withRespectToArg was $withRespectToArg, but the valid choices are {0}."))
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
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] =
    Success(DifferentiableFunctionExecution(inputs, Map(this -> value)))

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape: Try[Array[Option[Int]]] = Success(
    value.shape.map(Some(_))
  )
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
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] =
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
case class Sum[T: ClassTag](a: DifferentiableFunction[T])(implicit
                                                          num: Numeric[T]
) extends DifferentiableFunction[T] {
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] =
    a.computeAll(inputs) match {
      case Success(value) => Success(value.copy(outputs = value.outputs + (this -> NDArray(List(value.outputs(a).sum)))))
      case Failure(failure)        => Failure(failure)
    }

  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

  override def gradient(
                         withRespectToVariable: Variable[T]
                       ): Try[DifferentiableFunction[T]] = ???

  override def getInputs: Set[Input[T]] = ???

  override def getOutputShape: Try[Array[Option[Int]]] =
    a.getOutputShape match {
      case Success(_) => Success(Array(Some(1)))
      case failure    => failure
    }
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
case class Mean[T: ClassTag](a: DifferentiableFunction[T])(implicit
                                                          num: Fractional[T]
) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(value) => Success(NDArray(List(value.mean)))
      case failure        => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class Negate[T](a: DifferentiableFunction[T])(implicit
    num: Numeric[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(value) => Success(value.negate)
      case failure        => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class Square[T: ClassTag](a: DifferentiableFunction[T])(implicit
    num: Numeric[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(value) => Success(value.square)
      case failure        => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class Reciprocal[T: ClassTag](a: DifferentiableFunction[T])(implicit
    num: Fractional[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(value) => Success(value.reciprocal)
      case failure        => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class Exp[T: ClassTag](a: DifferentiableFunction[T])(implicit
    num: Fractional[T]
) extends UnaryElementWiseDifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(value) => Success(value.exp)
      case failure        => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class Add[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T])(
    implicit num: Numeric[T]
) extends BinaryDifferentiableFunctionWithBroadcast[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) =>
        b.compute(inputs) match {
          case Success(bValue) => aValue + bValue
          case failure         => failure
        }
      case failure => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
    a: DifferentiableFunction[T],
    b: DifferentiableFunction[T]
)(implicit
    num: Numeric[T]
) extends BinaryDifferentiableFunctionWithBroadcast[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) =>
        b.compute(inputs) match {
          case Success(bValue) => aValue - bValue
          case failure         => failure
        }
      case failure => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class Multiply[T: ClassTag](
    a: DifferentiableFunction[T],
    b: DifferentiableFunction[T]
)(implicit
    num: Numeric[T]
) extends BinaryDifferentiableFunctionWithBroadcast[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) =>
        b.compute(inputs) match {
          case Success(bValue) => aValue * bValue
          case failure         => failure
        }
      case failure => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
case class DotProduct[T: ClassTag](
    a: DifferentiableFunction[T],
    b: DifferentiableFunction[T]
)(implicit num: Numeric[T])
    extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) =>
        b.compute(inputs) match {
          case Success(bValue) => aValue dot bValue
          case failure         => failure
        }
      case failure => failure
    }

  //TODO
  override def computeAll(inputs: Map[Input[T], NDArray[T]]): Try[DifferentiableFunctionExecution[T]] = ???

  //TODO
  override def backpropagate(outputGradient: NDArray[T], computeResult: NDArray[T], withRespectToArg: Int): Try[NDArray[T]] = ???

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
}
