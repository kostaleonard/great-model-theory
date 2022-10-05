package ndarray

import scala.reflect.{ClassTag, classTag}
import scala.util.Try

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

  // TODO force T to be a numeric type
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
class NDArray[T] private (val shape: Array[Int], val elements: Array[T]) {
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

  //TODO this should use Try/Success/Failure with ShapeException
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

  //TODO add tests for these
  /** Returns true if the arrays have the same shape and elements.
    *
    * @param other
    *   The array with which to compare.
    */
  def arrayEquals(other: NDArray[T]): Boolean

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
  def ==(other: NDArray[T]): Try[NDArray[Boolean]]

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
  ): Boolean

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

  /** Returns a mask describing the approximate equality of the arrays.
    *
    * @param other
    *   The array with which to compare.
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
  def ~=(other: NDArray[T], epsilon: Double = 1e-5): Try[NDArray[Boolean]]
}
