package ndarray

import scala.reflect.{ClassTag, classTag}

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

  /** Returns an array filled with zeros.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def zeros[T: ClassTag](shape: Seq[Int]): NDArray[T] =
    NDArray.ofValue[T](shape, 0.asInstanceOf[T])

  /** Returns an array filled with ones.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def ones[T: ClassTag](shape: Seq[Int]): NDArray[T] =
    NDArray.ofValue[T](shape, 1.asInstanceOf[T])

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
  def arange[T: ClassTag](shape: Seq[Int]): NDArray[T] = new NDArray[T](
    shape.toArray,
    Array
      .range(0, shape.product)
      .map(_.asInstanceOf[T])
  )

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

  /** Returns an NDArray with the same elements as the input, but with the given
    * shape.
    *
    * @param targetShape
    *   The shape of the output array. The product must equal
    *   [[elements.length]].
    */
  def reshape(targetShape: Seq[Int]): NDArray[T] =
    new NDArray[T](targetShape.toArray, elements)
}
