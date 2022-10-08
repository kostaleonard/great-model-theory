package ndarray

import exceptions.ShapeException

import scala.reflect.{ClassTag, classTag}
import scala.util.{Failure, Success, Try}

/** An N-dimensional array.
  */
object NDArray {

  /** Returns an empty NDArray of the given type.
    *
    * @tparam T
    *   The array element type.
    */
  def empty[T: ClassTag] = new NDArray[T](Array.empty, Array.empty[T])

  /** Returns an array filled with the given value.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @param value
    *   The value to fill at every index in the array.
    * @tparam T
    *   The array element type.
    */
  def ofValue[T: ClassTag](shape: Seq[Int], value: T) =
    new NDArray[T](shape.toArray, Array.fill[T](shape.product)(value))

  // TODO force T to be a numeric type--add subclass?
  /** Returns an array filled with zeros.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def zeros[T: ClassTag](shape: Seq[Int]): NDArray[T] = classTag[T] match {
    case _ if classTag[T] == classTag[Float] =>
      NDArray.ofValue[Float](shape, 0).asInstanceOf[NDArray[T]]
    case _ if classTag[T] == classTag[Double] =>
      NDArray.ofValue[Double](shape, 0).asInstanceOf[NDArray[T]]
    case _ if classTag[T] == classTag[Int] =>
      NDArray.ofValue[Int](shape, 0).asInstanceOf[NDArray[T]]
    case _ if classTag[T] == classTag[Long] =>
      NDArray.ofValue[Long](shape, 0).asInstanceOf[NDArray[T]]
  }

  /** Returns an array filled with ones.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def ones[T: ClassTag](shape: Seq[Int]): NDArray[T] = classTag[T] match {
    case _ if classTag[T] == classTag[Float] =>
      NDArray.ofValue[Float](shape, 1).asInstanceOf[NDArray[T]]
    case _ if classTag[T] == classTag[Double] =>
      NDArray.ofValue[Double](shape, 1).asInstanceOf[NDArray[T]]
    case _ if classTag[T] == classTag[Int] =>
      NDArray.ofValue[Int](shape, 1).asInstanceOf[NDArray[T]]
    case _ if classTag[T] == classTag[Long] =>
      NDArray.ofValue[Long](shape, 1).asInstanceOf[NDArray[T]]
  }

  /** Returns an array from the given sequence.
    *
    * @param seq
    *   A sequence of array elements. The output array will have the same shape
    *   as this sequence.
    * @tparam T
    *   The array element type.
    */
  def apply[T: ClassTag](seq: Seq[T]): NDArray[T] =
    new NDArray[T](Array(seq.length), seq.toArray)

  /** Returns an array whose elements are 0, 1, 2, etc. when flattened.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def arange[T: ClassTag](shape: Seq[Int]): NDArray[T] = {
    val elements = classTag[T] match {
      case _ if classTag[T] == classTag[Float] =>
        Array.range(0, shape.product).map(_.asInstanceOf[Float])
      case _ if classTag[T] == classTag[Double] =>
        Array.range(0, shape.product).map(_.asInstanceOf[Double])
      case _ if classTag[T] == classTag[Int] =>
        Array.range(0, shape.product)
      case _ if classTag[T] == classTag[Long] =>
        Array.range(0, shape.product).map(_.asInstanceOf[Long])
    }
    new NDArray[T](shape.toArray, elements.asInstanceOf[Array[T]])
  }

  /** Returns an array whose elements are randomly initialized.
    *
    * Elements are drawn from a uniform distribution using scala.util.Random, so
    * Floats are in [0, 1), Ints are in [[Int.MinValue, Int.MaxValue]], etc.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type. Must be one of {Float, Double, Int, Long},
    *   otherwise this function will throw an error.
    */
  def random[T: ClassTag](shape: Seq[Int]): NDArray[T] = {
    val elements = classTag[T] match {
      case _ if classTag[T] == classTag[Float] =>
        Array.fill(shape.product)(scala.util.Random.nextFloat())
      case _ if classTag[T] == classTag[Double] =>
        Array.fill(shape.product)(scala.util.Random.nextDouble())
      case _ if classTag[T] == classTag[Int] =>
        Array.fill(shape.product)(scala.util.Random.nextInt())
      case _ if classTag[T] == classTag[Long] =>
        Array.fill(shape.product)(scala.util.Random.nextLong())
    }
    new NDArray[T](shape.toArray, elements.asInstanceOf[Array[T]])
  }
}

/** An N-dimensional array.
  *
  * @param shape
  *   The shape of the array. For example, an array with shape (2, 3) is a
  *   matrix with 2 rows and 3 columns.
  * @param elements
  *   The elements that make up the array. Must be of the same length as the
  *   product of all dimensions in shape.
  * @tparam T
  *   The array element type.
  */
class NDArray[T: ClassTag] private (
    val shape: Array[Int],
    val elements: Array[T]
) {
  private val strides = Array.fill[Int](shape.length)(1)
  strides.indices.reverse.drop(1).foreach { idx =>
    strides(idx) = shape(idx + 1) * strides(idx + 1)
  }

  /** Returns a flattened array of the elements in this NDArray.
    */
  def flatten(): Array[T] = elements

  /** Returns an element from the array.
    *
    * @param indices
    *   The indices to an element in the array. Must be of length
    *   [[shape.length]].
    */
  def apply(indices: Seq[Int]): T = elements(
    indices.indices.foldRight(0)((idx, accumulator) =>
      indices(idx) * strides(idx) + accumulator
    )
  )

  // TODO this should use Try/Success/Failure with ShapeException
  /** Returns an NDArray with the same elements as the input, but with the given
    * shape.
    *
    * @param targetShape
    *   The shape of the output array. The product must equal
    *   [[elements.length]].
    */
  def reshape(targetShape: Seq[Int]): NDArray[T] =
    new NDArray[T](targetShape.toArray, elements)

  // TODO add indices method to get List[Array[Int]] for all indices in order: (0, 0), (0, 1), (0, 2), ...
  // TODO can we make this class implement Iterable?

  /** Returns true if the arrays have the same shape and elements.
    *
    * @param other
    *   The array with which to compare.
    */
  def arrayEquals(other: NDArray[T]): Boolean = this == other match {
    case Success(mask) => mask.flatten().forall(identity)
    case _             => false
  }

  /** Returns false if the arrays have the same shape and elements.
    *
    * @param other
    *   The array with which to compare.
    */
  def arrayNotEquals(other: NDArray[T]): Boolean = !arrayEquals(other)

  /** Returns a mask describing the equality of the arrays.
    *
    * @param other
    *   The array with which to compare. Must be the same shape as this array.
    * @return
    *   A mask describing the equality of the arrays at each position. Each
    *   element of the mask is true if the arrays are equal at that position,
    *   false otherwise. The mask is of the same shape as the arrays. If the
    *   arrays are of different shapes, returns Failure.
    */
  def ==(other: NDArray[T]): Try[NDArray[Boolean]] =
    if (shape sameElements other.shape) {
      val thisFlat = flatten()
      val otherFlat = other.flatten()
      val mask = thisFlat.indices.map(idx => thisFlat(idx) == otherFlat(idx))
      Success(NDArray[Boolean](mask).reshape(shape.toList))
    } else
      Failure(new ShapeException("Arrays must have same shape for comparison"))

  // TODO this only works for numeric types--make this Try/Success/Fail?
  /** Returns true if the arrays have the same shape and elements within error.
    *
    * @param other
    *   The array with which to compare.
    * @param epsilon
    *   The range within which elements are considered equal, i.e.,
    *   [[abs(a - b) <= epsilon]].
    */
  def arrayApproximatelyEquals(
      other: NDArray[T],
      epsilon: Double = 1e-5
  ): Boolean = this.~=(other, epsilon = epsilon) match {
    case Success(mask) => mask.flatten().forall(identity)
    case _             => false
  }

  /** Returns false if the arrays have the same shape and elements within error.
    *
    * @param other
    *   The array with which to compare.
    * @param epsilon
    *   The range within which elements are considered equal, i.e.,
    *   [[abs(a - b) <= epsilon]].
    */
  def arrayNotApproximatelyEquals(
      other: NDArray[T],
      epsilon: Double = 1e-5
  ): Boolean = !arrayApproximatelyEquals(other, epsilon = epsilon)

  // TODO see Array[T].sum() for a way this might work with typing to force numerics
  /** Returns a mask describing the approximate equality of the arrays.
    *
    * @param other
    *   The array with which to compare. Must be the same shape as this array.
    * @param epsilon
    *   The range within which elements are considered equal, i.e.,
    *   [[abs(a - b) <= epsilon]].
    * @return
    *   A mask describing the approximate equality of the arrays at each
    *   position. Each element of the mask is true if the arrays are
    *   approximately equal at that position, false otherwise. The mask is of
    *   the same shape as the arrays. If the arrays are of different shapes,
    *   returns Failure.
    */
  def ~=(other: NDArray[T], epsilon: Double = 1e-5): Try[NDArray[Boolean]] =
    if (shape sameElements other.shape) {
      val thisFlat = flatten()
      val otherFlat = other.flatten()
      classTag[T] match {
        case _ if classTag[T] == classTag[Float] =>
          val epsilonAsFloat = epsilon.asInstanceOf[Float]
          val thisFlatAsFloat = thisFlat.asInstanceOf[Array[Float]]
          val otherFlatAsFloat = otherFlat.asInstanceOf[Array[Float]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(
              thisFlatAsFloat(idx) - otherFlatAsFloat(idx)
            ) <= epsilonAsFloat
          )
          Success(NDArray[Boolean](mask).reshape(shape.toList))
        case _ if classTag[T] == classTag[Double] =>
          val thisFlatAsDouble = thisFlat.asInstanceOf[Array[Double]]
          val otherFlatAsDouble = otherFlat.asInstanceOf[Array[Double]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(thisFlatAsDouble(idx) - otherFlatAsDouble(idx)) <= epsilon
          )
          Success(NDArray[Boolean](mask).reshape(shape.toList))
        case _ if classTag[T] == classTag[Int] =>
          val epsilonAsInt = epsilon.asInstanceOf[Int]
          val thisFlatAsInt = thisFlat.asInstanceOf[Array[Int]]
          val otherFlatAsInt = otherFlat.asInstanceOf[Array[Int]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(thisFlatAsInt(idx) - otherFlatAsInt(idx)) <= epsilonAsInt
          )
          Success(NDArray[Boolean](mask).reshape(shape.toList))
        case _ if classTag[T] == classTag[Long] =>
          val epsilonAsLong = epsilon.asInstanceOf[Long]
          val thisFlatAsLong = thisFlat.asInstanceOf[Array[Long]]
          val otherFlatAsLong = otherFlat.asInstanceOf[Array[Long]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(
              thisFlatAsLong(idx) - otherFlatAsLong(idx)
            ) <= epsilonAsLong
          )
          Success(NDArray[Boolean](mask).reshape(shape.toList))
      }
    } else
      Failure(new ShapeException("Arrays must have same shape for comparison"))

  /** Returns the result of element-wise addition of the two NDArrays.
    *
    * @param other
    *   The array to add. Must be the same shape as this array.
    * @param num
    *   An implicit parameter defining a set of numeric operations which
    *   includes the `+` operator to be used in forming the sum.
    * @tparam B
    *   The result type of the `+` operator.
    * @return
    *   An NDArray of the same size
    */
  def +[B >: T: ClassTag](
      other: NDArray[T]
  )(implicit num: Numeric[B]): NDArray[B] = {
    val thisFlat = flatten()
    val otherFlat = other.flatten()
    val result =
      thisFlat.indices.map(idx => num.plus(thisFlat(idx), otherFlat(idx)))
    NDArray(result)
  }

  /** Returns the sum of all elements.
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations which
    *   includes the `+` operator to be used in forming the sum.
    * @tparam B
    *   The result type of the `+` operator.
    */
  def sum[B >: T](implicit num: Numeric[B]): B = flatten().reduce(num.plus)

  def dot[B >: T: ClassTag](other: NDArray[T])(implicit num: Numeric[B]): NDArray[B] = {
    // TODO implement dot
    // TODO docstring
    NDArray.empty
  }

  /** Returns a slice of the NDArray.
    *
    * @param indices
    *   The
    * @return
    */
  def slice(indices: List[Option[List[Int]]]): NDArray[T] = {
    // TODO implement slice
    // TODO docstring
    NDArray.empty
  }

  /** Returns a new NDArray with dimensions of length 1 removed. */
  def squeeze(): NDArray[T] = reshape(shape.filter(_ > 1).toList)

  // TODO add toString method
}
