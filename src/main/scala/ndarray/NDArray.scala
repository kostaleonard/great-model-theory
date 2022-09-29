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

  //TODO add apply method that just returns zeros with correct shape
}

/**
 * An N-dimensional array.
 *
 * @param shape The shape of the array. For example, an array with shape (2, 3) is a matrix with 2 rows and 3 columns.
 * @param elements The elements that make up the array. Must be of length [[shape.product]].
 * @tparam T The array element type.
 */
class NDArray[T] private (shape: List[Int], elements: Array[T]) {

  /**
   * Returns a flattened array of the elements in this NDArray.
   */
  def flatten(): Array[T] = elements

  //TODO apply with indices in same length as shape accesses elements
}
