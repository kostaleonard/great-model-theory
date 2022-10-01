package ndarray

import scala.reflect.ClassTag

/**
 * An N-dimensional array.
 */
object NDArray {

  /**
   * Returns an array filled with the given value.
   *
   * @param shape The shape of the array. For example, an array with shape (2, 3) is a matrix with 2 rows and 3 columns.
   * @param value The value to fill at every index in the array.
   * @tparam T The array element type.
   */
  def ofValue[T: ClassTag](shape: List[Int], value: T) = new NDArray[T](shape, Array.fill[T](shape.product)(value))

  /**
   * Returns an array filled with zeros.
   *
   * @param shape The shape of the array. For example, an array with shape (2, 3) is a matrix with 2 rows and 3 columns.
   * @tparam T The array element type.
   */
  def zeros[T: ClassTag](shape: List[Int]): NDArray[T] = NDArray.ofValue[T](shape, 0.asInstanceOf[T])

  /**
   * Returns an array filled with ones.
   *
   * @param shape The shape of the array. For example, an array with shape (2, 3) is a matrix with 2 rows and 3 columns.
   * @tparam T The array element type.
   */
  def ones[T: ClassTag](shape: List[Int]): NDArray[T] = NDArray.ofValue[T](shape, 1.asInstanceOf[T])

  //TODO if the input sequence does not have the same length for all rows, this will cause issues.
  //TODO warnings
  /**
   * Returns an array from the possibly nested sequence.
   *
   * @param seq A possibly nested sequence of array elements. The output array will have the same shape as this
   *            sequence.
   * @tparam T The array element type.
   */
  def apply[T: ClassTag](seq: Seq[_]): NDArray[T] = seq match {
    case nested: Seq[Seq[_]] =>
      val subarrays = nested.map(NDArray[T](_))
      new NDArray[T](subarrays.length +: subarrays.head.shape, subarrays.flatMap(_.flatten()).toArray)
    case flat: Seq[T] => new NDArray[T](List(seq.length), flat.toArray)
  }

  /**
   * Returns an array whose elements are 0, 1, 2, etc. when flattened.
   *
   * @param shape The shape of the array. For example, an array with shape (2, 3) is a matrix with 2 rows and 3 columns.
   * @tparam T The array element type.
   */
  def arange[T: ClassTag](shape: List[Int]): NDArray[T] = new NDArray[T](shape, Array.range(0, shape.product)
    .map(_.asInstanceOf[T]))
}

/**
 * An N-dimensional array.
 *
 * @param shape The shape of the array. For example, an array with shape (2, 3) is a matrix with 2 rows and 3 columns.
 * @param elements The elements that make up the array. Must be of length [[shape.product]].
 * @tparam T The array element type.
 */
class NDArray[T] private (val shape: List[Int], val elements: Array[T]) {
  private val shapeMultipliers = Array.fill[Int](shape.length)(1)
  shapeMultipliers.indices.reverse.drop(1).foreach{ idx =>
    shapeMultipliers(idx) = shape(idx + 1) * shapeMultipliers(idx + 1)
  }

  /**
   * Returns a flattened array of the elements in this NDArray.
   */
  def flatten(): Array[T] = elements

  //TODO also allow for a partial apply on arbitrary ranks
  /**
   * Returns an element from the array.
   *
   * @param indices The indices to an element in the array. Must be of length [[shape.length]].
   */
  def apply(indices: List[Int]): T = elements(
    indices.indices.foldRight(0)((idx, accumulator) => indices(idx) * shapeMultipliers(idx) + accumulator)
  )

  //TODO set an element in the array. Also would like to be able to set with an NDArray if only a subset of indices are provided

  /**
   * Returns an NDArray with the same elements as the input, but with the given shape.
   *
   * @param targetShape The shape of the output array. The product must equal [[shape.product]].
   */
  def reshape(targetShape: List[Int]): NDArray[T] = new NDArray[T](targetShape, elements)
}
