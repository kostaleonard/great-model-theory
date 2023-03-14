package ndarray

import exceptions.ShapeException

import scala.annotation.tailrec
import scala.collection.immutable.ArraySeq
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
  def ofValue[T: ClassTag](shape: Array[Int], value: T) =
    new NDArray[T](shape, Array.fill[T](shape.product)(value))

  /** Returns an array filled with zeros.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def zeros[T: ClassTag](shape: Array[Int]): NDArray[T] = classTag[T] match {
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
  def ones[T: ClassTag](shape: Array[Int]): NDArray[T] = classTag[T] match {
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
    * @param arr
    *   A array elements.
    * @tparam T
    *   The array element type.
    */
  def apply[T: ClassTag](arr: Array[T]): NDArray[T] = {
    new NDArray[T](Array(arr.length), arr)
  }

  /** Returns an array whose elements are 0, 1, 2, etc. when flattened.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type.
    */
  def arange[T: ClassTag](shape: Array[Int]): NDArray[T] = {
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
    new NDArray[T](shape, elements.asInstanceOf[Array[T]])
  }

  /** Returns an array whose elements are randomly initialized.
    *
    * Elements are drawn from a uniform distribution using scala.util.Random, so
    * Floats are in [0, 1), Ints are in [Int.MinValue, Int.MaxValue], etc.
    *
    * @param shape
    *   The shape of the array. For example, an array with shape (2, 3) is a
    *   matrix with 2 rows and 3 columns.
    * @tparam T
    *   The array element type. Must be one of {Float, Double, Int, Long},
    *   otherwise this function will throw an error.
    */
  def random[T: ClassTag](shape: Array[Int]): NDArray[T] = {
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
    new NDArray[T](shape, elements.asInstanceOf[Array[T]])
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
    elements: Array[T]
) {
  if (shape.isEmpty && !elements.isEmpty)
    throw new ShapeException(
      "Could not create array with empty shape and non-empty elements"
    )
  else if (shape.nonEmpty && shape.product != elements.length)
    throw new ShapeException(
      s"Could not create array with ${elements.length} elements to ${shape
          .mkString("Array(", ", ", ")")}"
    )
  private val strides = Array.fill[Int](shape.length)(1)
  strides.indices.reverse.drop(1).foreach { idx =>
    strides(idx) = shape(idx + 1) * strides(idx + 1)
  }

  /** Returns a flattened array of the elements in this NDArray. */
  def flatten(): Array[T] = elements

  /** Returns true if this array has no elements. */
  def isEmpty: Boolean = elements.isEmpty

  /** Returns true if this array has elements. */
  def nonEmpty: Boolean = !isEmpty

  private def getFlattenedIndex(indices: Array[Int]): Int =
    indices.indices.foldRight(0)((idx, accumulator) =>
      indices(idx) * strides(idx) + accumulator
    )

  /** Returns an element from the array.
    *
    * @param indices
    *   The indices to an element in the array. Must be of length shape.length.
    */
  def apply(indices: Array[Int]): T = elements(getFlattenedIndex(indices))

  /** Returns an NDArray with the same elements, but with the new shape.
    *
    * @param targetShape
    *   The shape of the output array. The product must equal elements.length.
    */
  def reshape(targetShape: Array[Int]): NDArray[T] =
    new NDArray[T](targetShape, elements)

  /** Returns an NDArray with the same elements converted to Float. */
  def toFloat(implicit num: Numeric[T]): NDArray[Float] =
    map(x => num.toFloat(x))

  /** Returns an NDArray with the same elements converted to Double. */
  def toDouble(implicit num: Numeric[T]): NDArray[Double] =
    map(x => num.toDouble(x))

  /** Returns an NDArray with the same elements converted to Int. */
  def toInt(implicit num: Numeric[T]): NDArray[Int] = map(x => num.toInt(x))

  /** Returns an NDArray with the same elements converted to Long. */
  def toLong(implicit num: Numeric[T]): NDArray[Long] = map(x => num.toLong(x))

  /** Returns an iterator of all element indices, in order.
    *
    * Each element in the returned iterator is an array that can be applied to
    * extract a single element. The order of these indices is last dimension
    * first. For example, if this array was of shape 4 x 3 x 2, the indices
    * would be (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), etc.
    */
  def indices: Iterator[Array[Int]] = indexIterator(shape)

  private def indexIterator(indexShape: Array[Int]): Iterator[Array[Int]] = {
    val indexStrides = Array.fill[Int](indexShape.length)(1)
    indexStrides.indices.reverse.drop(1).foreach { idx =>
      indexStrides(idx) = indexShape(idx + 1) * indexStrides(idx + 1)
    }
    Iterator.tabulate(indexShape.product) { elementIdx =>
      val indexArray = Array.fill(indexShape.length)(0)
      var remainder = elementIdx
      indexStrides.indices.foreach { strideIdx =>
        if (strideIdx == indexStrides.length - 1)
          indexArray(strideIdx) = remainder
        else {
          indexArray(strideIdx) = remainder / indexStrides(strideIdx)
          remainder = remainder % indexStrides(strideIdx)
        }
      }
      indexArray
    }
  }

  /** Returns an NDArray with the value at the indices updated.
    *
    * @param indices
    *   The indices to an element in the array. Must be of length shape.length.
    * @param element
    *   The value to fill in the new array at the given indices.
    * @return
    */
  def updated(indices: Array[Int], element: T): NDArray[T] = {
    val newElements = elements.updated(getFlattenedIndex(indices), element)
    new NDArray[T](shape, newElements)
  }

  /** Returns true if the arrays have the same shape and elements.
    *
    * @param other
    *   The array with which to compare.
    */
  def arrayEquals(other: NDArray[T]): Boolean = try {
    (this == other).flatten().forall(identity)
  } catch {
    case _: ShapeException => false
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
    *   false otherwise. The mask is of the same shape as the arrays when
    *   broadcast together. If the arrays are of incompatible shapes, returns an
    *   array with a single element containing false.
    */
  def ==(other: NDArray[T]): NDArray[Boolean] =
    if (shape sameElements other.shape) {
      val thisFlat = flatten()
      val otherFlat = other.flatten()
      val mask = thisFlat.indices.map(idx => thisFlat(idx) == otherFlat(idx))
      NDArray[Boolean](mask.toArray).reshape(shape)
    } else {
      val (broadcastThis, broadcastOther) = broadcastWith(other)
      broadcastThis == broadcastOther
    }

  /** Returns true if the arrays have the same shape and elements within error.
    *
    * @param other
    *   The array with which to compare.
    * @param epsilon
    *   The range within which elements are considered equal, i.e., abs(a - b)
    *   <= epsilon.
    */
  def arrayApproximatelyEquals(
      other: NDArray[T],
      epsilon: Double = 1e-5
  ): Boolean = try {
    this.~=(other, epsilon = epsilon).flatten().forall(identity)
  } catch {
    case _: ShapeException => false
  }

  /** Returns false if the arrays have the same shape and elements within error.
    *
    * @param other
    *   The array with which to compare.
    * @param epsilon
    *   The range within which elements are considered equal, i.e., abs(a - b)
    *   <= epsilon.
    */
  def arrayNotApproximatelyEquals(
      other: NDArray[T],
      epsilon: Double = 1e-5
  ): Boolean = !arrayApproximatelyEquals(other, epsilon = epsilon)

  /** Returns a mask describing the approximate equality of the arrays.
    *
    * @param other
    *   The array with which to compare. Must be the same shape as this array.
    * @param epsilon
    *   The range within which elements are considered equal, i.e., abs(a - b)
    *   <= epsilon.
    * @return
    *   A mask describing the approximate equality of the arrays at each
    *   position. Each element of the mask is true if the arrays are
    *   approximately equal at that position, false otherwise. The mask is of
    *   the same shape as the arrays when broadcast together. If the arrays are
    *   of incompatible shapes, throws a ShapeException.
    */
  def ~=(other: NDArray[T], epsilon: Double = 1e-5): NDArray[Boolean] =
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
          NDArray[Boolean](mask.toArray).reshape(shape)
        case _ if classTag[T] == classTag[Double] =>
          val thisFlatAsDouble = thisFlat.asInstanceOf[Array[Double]]
          val otherFlatAsDouble = otherFlat.asInstanceOf[Array[Double]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(thisFlatAsDouble(idx) - otherFlatAsDouble(idx)) <= epsilon
          )
          NDArray[Boolean](mask.toArray).reshape(shape)
        case _ if classTag[T] == classTag[Int] =>
          val epsilonAsInt = epsilon.asInstanceOf[Int]
          val thisFlatAsInt = thisFlat.asInstanceOf[Array[Int]]
          val otherFlatAsInt = otherFlat.asInstanceOf[Array[Int]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(thisFlatAsInt(idx) - otherFlatAsInt(idx)) <= epsilonAsInt
          )
          NDArray[Boolean](mask.toArray).reshape(shape)
        case _ if classTag[T] == classTag[Long] =>
          val epsilonAsLong = epsilon.asInstanceOf[Long]
          val thisFlatAsLong = thisFlat.asInstanceOf[Array[Long]]
          val otherFlatAsLong = otherFlat.asInstanceOf[Array[Long]]
          val mask = thisFlat.indices.map(idx =>
            Math.abs(
              thisFlatAsLong(idx) - otherFlatAsLong(idx)
            ) <= epsilonAsLong
          )
          NDArray[Boolean](mask.toArray).reshape(shape)
      }
    } else {
      val (broadcastThis, broadcastOther) = broadcastWith(other)
      broadcastThis.~=(broadcastOther, epsilon = epsilon)
    }

  /** Returns this array broadcast to the target shape.
    *
    * Broadcasting rules follow those in NumPy. This operation compares the
    * shape of this array and the target shape from right to left and determines
    * whether this array can be broadcast. A dimension in this array's shape is
    * compatible with a dimension in the target shape when they are equal or
    * this array's dimension is equal to 1. If the target shape has more
    * dimensions than this array's shape, missing dimensions are filled in on
    * the left with 1.
    *
    * @param targetShape
    *   The shape to which to broadcast the array.
    */
  def broadcastTo(targetShape: Array[Int]): NDArray[T] =
    if (shape.length > targetShape.length)
      throw new ShapeException(
        s"Cannot broadcast array of shape ${shape.mkString("Array(", ", ", ")")} into smaller shape ${targetShape
            .mkString("Array(", ", ", ")")}"
      )
    else {
      val onesPaddedShapeThis =
        shape.reverse.padTo(targetShape.length, 1).reverse
      val onesPaddedThis = reshape(onesPaddedShapeThis)
      onesPaddedThis.broadcastToWithMatchingNumDimensions(
        targetShape,
        targetShape.length - 1
      )
    }

  @tailrec
  private def broadcastToWithMatchingNumDimensions(
      targetShape: Array[Int],
      shapeIdx: Int
  ): NDArray[T] =
    if (shapeIdx < 0) this
    else if (shape(shapeIdx) == targetShape(shapeIdx))
      broadcastToWithMatchingNumDimensions(targetShape, shapeIdx - 1)
    else if (shape(shapeIdx) == 1) {
      val ndarrayIndices = indexIterator(shape.take(shapeIdx))
      val sliceIndices =
        if (ndarrayIndices.isEmpty)
          Iterator(Array.fill[Option[Array[Int]]](targetShape.length)(None))
        else
          ndarrayIndices.map(ndarrayIndex =>
            ndarrayIndex.map(idx => Some(Array(idx))) ++ Array.fill(
              targetShape.length - shapeIdx
            )(None)
          )
      val sliceElements = sliceIndices.flatMap(sliceIndex =>
        (0 until targetShape(shapeIdx)).flatMap(_ =>
          slice(sliceIndex).flatten()
        )
      )
      val newShape = shape.updated(shapeIdx, targetShape(shapeIdx))
      val broadcastArray = NDArray[T](sliceElements.toArray).reshape(newShape)
      broadcastArray.broadcastToWithMatchingNumDimensions(
        targetShape,
        shapeIdx - 1
      )
    } else
      throw new ShapeException(
        s"Cannot broadcast dimension of size ${shape(shapeIdx)} to ${targetShape(shapeIdx)}"
      )

  /** Returns this array and the input array broadcast to matching dimensions.
    *
    * Broadcasting rules follow those in NumPy. This operation compares the
    * shapes of the two arrays from right to left and determines the final shape
    * of the broadcast arrays. Two dimensions are compatible when they are equal
    * or one of them is equal to 1. If one of the two arrays has more dimensions
    * than the other, missing dimensions are filled in on the left with 1.
    *
    * Example: You have an array of 256 x 256 x 3 color values (i.e., an image),
    * and you want to scale each color by a different value. You multiply the
    * image array by a 1-dimensional array of length 3. During the
    * multiplication, the latter array is broadcast to 256 x 256 x 3. Note that
    * the latter array's missing dimensions were filled to 1 x 1 x 3.
    *
    * @param other
    *   The array with which to broadcast.
    */
  def broadcastWith(other: NDArray[T]): (NDArray[T], NDArray[T]) = {
    val resultShape = getBroadcastShapeWith(other)
    val broadcastThis = broadcastTo(resultShape)
    val broadcastOther = other.broadcastTo(resultShape)
    (broadcastThis, broadcastOther)
  }

  /** Returns the shape to which the arrays should be broadcast together.
    *
    * @param other
    *   The array with which to broadcast.
    * @return
    *   The shape of the two broadcast arrays.
    */
  private def getBroadcastShapeWith(other: NDArray[T]): Array[Int] = {
    val finalNumDimensions = shape.length max other.shape.length
    val onesPaddedShapeThis = shape.reverse.padTo(finalNumDimensions, 1).reverse
    val onesPaddedShapeOther =
      other.shape.reverse.padTo(finalNumDimensions, 1).reverse
    val shapesMatch = (0 until finalNumDimensions).forall(idx =>
      onesPaddedShapeThis(idx) == onesPaddedShapeOther(idx) ||
        onesPaddedShapeThis(idx) == 1 ||
        onesPaddedShapeOther(idx) == 1
    )
    if (shapesMatch)
      (0 until finalNumDimensions)
        .map(idx => onesPaddedShapeThis(idx) max onesPaddedShapeOther(idx))
        .toArray
    else
      throw new ShapeException(
        s"Could not broadcast arrays of shape ${shape
            .mkString("Array(", ", ", ")")} and ${other.shape.mkString("Array(", ", ", ")")}"
      )
  }

  /** Returns the result of element-wise addition of the two NDArrays.
    *
    * @param other
    *   The array to add. If not the same shape as this array, this function
    *   attempts a broadcast.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def +(
      other: NDArray[T]
  )(implicit num: Numeric[T]): NDArray[T] = if (
    shape sameElements other.shape
  ) {
    val thisFlat = flatten()
    val otherFlat = other.flatten()
    val result =
      thisFlat.indices.map(idx => num.plus(thisFlat(idx), otherFlat(idx)))
    NDArray(result.toArray).reshape(shape)
  } else {
    val (arr1, arr2) = broadcastWith(other)
    arr1 + arr2
  }

  /** Returns the result of element-wise subtraction of the two NDArrays.
    *
    * @param other
    *   The array to subtract. If not the same shape as this array, this
    *   function attempts a broadcast.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def -(
      other: NDArray[T]
  )(implicit num: Numeric[T]): NDArray[T] = if (
    shape sameElements other.shape
  ) {
    val thisFlat = flatten()
    val otherFlat = other.flatten()
    val result =
      thisFlat.indices.map(idx => num.minus(thisFlat(idx), otherFlat(idx)))
    NDArray(result.toArray).reshape(shape)
  } else {
    val (arr1, arr2) = broadcastWith(other)
    arr1 - arr2
  }

  /** Returns the result of element-wise multiplication of the two NDArrays.
    *
    * @param other
    *   The array to multiply. If not the same shape as this array, this
    *   function attempts a broadcast.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def *(
      other: NDArray[T]
  )(implicit num: Numeric[T]): NDArray[T] = if (
    shape sameElements other.shape
  ) {
    val thisFlat = flatten()
    val otherFlat = other.flatten()
    val result =
      thisFlat.indices.map(idx => num.times(thisFlat(idx), otherFlat(idx)))
    NDArray(result.toArray).reshape(shape)
  } else {
    val (arr1, arr2) = broadcastWith(other)
    arr1 * arr2
  }

  /** Returns the result of element-wise division of the two NDArrays.
    *
    * @param other
    *   The array to divide. If not the same shape as this array, this function
    *   attempts a broadcast.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def /(
      other: NDArray[T]
  )(implicit num: Fractional[T]): NDArray[T] = if (
    shape sameElements other.shape
  ) {
    val thisFlat = flatten()
    val otherFlat = other.flatten()
    val result =
      thisFlat.indices.map(idx => num.div(thisFlat(idx), otherFlat(idx)))
    NDArray(result.toArray).reshape(shape)
  } else {
    val (arr1, arr2) = broadcastWith(other)
    arr1 / arr2
  }

  /** Returns the result of element-wise addition by broadcasting the operand.
    *
    * @param other
    *   The number to add. This function broadcasts the operand across all
    *   elements.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def +(other: T)(implicit num: Numeric[T]): NDArray[T] =
    this + NDArray(Array(other))

  /** Returns the result of element-wise subtraction by broadcasting the
    * operand.
    *
    * @param other
    *   The number to subtract. This function broadcasts the operand across all
    *   elements.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def -(other: T)(implicit num: Numeric[T]): NDArray[T] =
    this - NDArray(Array(other))

  /** Returns the result of element-wise multiplication by broadcasting the
    * operand.
    *
    * @param other
    *   The number to multiply. This function broadcasts the operand across all
    *   elements.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def *(other: T)(implicit num: Numeric[T]): NDArray[T] =
    this * NDArray(Array(other))

  /** Returns the result of element-wise division by broadcasting the operand.
    *
    * @param other
    *   The number to divide. This function broadcasts the operand across all
    *   elements.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   An NDArray of the same size.
    */
  def /(other: T)(implicit num: Fractional[T]): NDArray[T] =
    this / NDArray(Array(other))

  /** Returns the sum of all elements.
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def sum(implicit num: Numeric[T]): T = flatten().reduce(num.plus)

  /** Returns an NDArray with elements summed along an axis.
    *
    * @param axis
    *   The axis along which to sum. This axis is eliminated in the result.
    * @param keepDims
    *   If true, do not eliminate the reduced axis; keep it with size 1.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def sumAxis(axis: Int, keepDims: Boolean = false)(implicit
      num: Numeric[T]
  ): NDArray[T] = reduce(_.sum, axis, keepDims = keepDims)

  /** Returns the mean of all elements.
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def mean(implicit num: Fractional[T]): T =
    if (isEmpty) num.zero else num.div(sum, num.fromInt(flatten().length))

  /** Returns an NDArray with all elements squared.
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def square(implicit num: Numeric[T]): NDArray[T] = map(x => num.times(x, x))

  /** Returns an NDArray with all elements inverted.
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def reciprocal(implicit num: Fractional[T]): NDArray[T] =
    map(x => num.div(num.fromInt(1), x))

  /** Returns an NDArray with all elements negated.
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def negate(implicit num: Numeric[T]): NDArray[T] = map(num.negate)

  /** Returns an NDArray with all elements exponentiated (f(x) = pow(e, x)).
    *
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def exp(implicit num: Fractional[T]): NDArray[T] = (classTag[T] match {
    case _ if classTag[T] == classTag[Float] =>
      map(x => Math.exp(num.toDouble(x)))
    case _ if classTag[T] == classTag[Double] =>
      map(x => Math.exp(num.toDouble(x)))
  }).asInstanceOf[NDArray[T]]

  /** Returns an array with axes transposed. */
  def transpose: NDArray[T] = {
    val ndarrayIndices = indexIterator(shape.reverse).map(_.reverse)
    val transposeElements =
      ndarrayIndices.map(ndarrayIndex => apply(ndarrayIndex))
    NDArray(transposeElements.toArray).reshape(shape.reverse)
  }

  /** Returns a new NDArray with dimensions of length 1 removed. */
  def squeeze(): NDArray[T] = reshape(shape.filter(_ > 1))

  /** Returns a slice of the NDArray.
    *
    * @param indices
    *   The indices on which to collect elements from the array. Each member of
    *   indices can be None (take all elements along this dimension) or Array of
    *   Int (take all elements for the values of this dimension specified in the
    *   array; if the array contains only one element, do not flatten this
    *   dimension).
    * @return
    *   A slice of the NDArray. The shape is determined by indices.
    */
  def slice(indices: Array[Option[Array[Int]]]): NDArray[T] = {
    val dimensionIndices = indices.indices
      .map(dimensionIdx =>
        indices(dimensionIdx) match {
          case None             => Array.range(0, shape(dimensionIdx))
          case Some(indexArray) => indexArray
        }
      )
      .toArray
    val resultShape = dimensionIndices.map(_.length)
    val indexStrides = Array.fill[Int](dimensionIndices.length)(1)
    indexStrides.indices.reverse.drop(1).foreach { idx =>
      indexStrides(idx) =
        dimensionIndices(idx + 1).length * indexStrides(idx + 1)
    }
    // For optimization purposes, we are reimplementing some of indexIterator.
    val ndarrayIndices = Iterator.tabulate(resultShape.product) { elementIdx =>
      val indexArray = Array.fill(shape.length)(0)
      var remainder = elementIdx
      indexStrides.indices.foreach { strideIdx =>
        if (strideIdx == indexStrides.length - 1)
          indexArray(strideIdx) = dimensionIndices(strideIdx)(remainder)
        else {
          indexArray(strideIdx) =
            dimensionIndices(strideIdx)(remainder / indexStrides(strideIdx))
          remainder = remainder % indexStrides(strideIdx)
        }
      }
      indexArray
    }
    val sliceElements =
      ndarrayIndices.map(elementIndices => apply(elementIndices))
    NDArray(sliceElements.toArray).reshape(resultShape)
  }

  /** Returns the result of matrix multiplication of 2D arrays.
    *
    * @param other
    *   The array to multiply. Must be 2D.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    * @return
    *   The result of matrix multiplication of 2D arrays. If this array is of
    *   shape (m, n) and the other array is of shape (n, o), the result will be
    *   of shape (m, o).
    */
  def matmul(
      other: NDArray[T]
  )(implicit num: Numeric[T]): NDArray[T] =
    if (shape.length != 2 || other.shape.length != 2)
      throw new ShapeException("matmul inputs must be 2D arrays")
    else if (shape(1) != other.shape(0))
      throw new ShapeException("Array 1 columns do not match array 2 rows")
    else {
      val numRows = shape(0)
      val numCols = other.shape(1)
      var newElementsReversed = List.empty[T]
      (0 until numRows).foreach { r =>
        (0 until numCols).foreach { c =>
          val rowVector = slice(Array(Some(Array(r)), None)).squeeze()
          val colVector = other.slice(Array(None, Some(Array(c)))).squeeze()
          // The dot product of 1D arrays is a scalar.
          val vectorDotProduct = rowVector dot colVector
          newElementsReversed =
            vectorDotProduct(Array(0)) +: newElementsReversed
        }
      }
      NDArray[T](newElementsReversed.reverse.toArray)
        .reshape(Array(numRows, numCols))
    }

  /** Returns the dot product of this array with another array.
    *
    * If both this and other are 1-D arrays, it is inner product of vectors. If
    * both this and other are 2-D arrays, it is matrix multiplication. If this
    * is an N-D array and other is a 1-D array, it is an inner product over the
    * last axis of this and other; e.g., if this.shape is (m, n) and other.shape
    * is (n,), the result is an NDArray of shape (m,) which is the inner product
    * of all m length n 1D arrays in this with other. If this is an N-D array
    * and other is an M-D array (where M>=2), it is an inner product over the
    * last axis of this and the second-to-last axis of other: dot(this,
    * other)[i, j, k, m] = sum(this[i, j, :] * other[k, :, m]).
    *
    * @param other
    *   The array to dot.
    * @param num
    *   An implicit parameter defining a set of numeric operations.
    */
  def dot(
      other: NDArray[T]
  )(implicit num: Numeric[T]): NDArray[T] = {
    def vectorInnerProduct(): NDArray[T] = {
      val thisFlat = flatten()
      val otherFlat = other.flatten()
      if (thisFlat.length != otherFlat.length)
        throw new ShapeException(
          "1D arrays must be of the same length to compute dot product"
        )
      else
        NDArray(
          List(
            thisFlat
              .zip(otherFlat)
              .map(tup => num.times(tup._1, tup._2))
              .reduce(num.plus)
          ).toArray
        )
    }
    def lastAxisInnerProduct(): NDArray[T] =
      if (shape.last != other.shape.head)
        throw new ShapeException(
          "Last axes must match for last axis inner product"
        )
      else {
        val resultShape = shape.dropRight(1)
        val ndarrayIndices = indexIterator(resultShape)
        val newElementsArrays = ndarrayIndices.map { ndarrayIndex =>
          val sliceIndices = ndarrayIndex.map(idx => Some(Array(idx))) :+ None
          slice(sliceIndices).squeeze() dot other
        }
        // Because this is a 1D vector inner product, each array holds a scalar.
        val newElements = newElementsArrays.map(_.flatten().head)
        NDArray[T](newElements.toArray).reshape(resultShape)
      }
    def multidimensionalInnerProduct(): NDArray[T] =
      if (shape.last != other.shape(other.shape.length - 2))
        throw new ShapeException(
          "Last axes must match second to last axis for multidimensional inner product"
        )
      else {
        val resultShape =
          shape.dropRight(1) ++ other.shape.dropRight(2) :+ other.shape.last
        val ndarrayIndicesThis = indexIterator(shape.dropRight(1))
        // Because this is a 1D vector inner product, each array holds a scalar.
        val newElements = ndarrayIndicesThis.flatMap { ndarrayIndexThis =>
          val sliceIndicesThis =
            ndarrayIndexThis.map(idx => Some(Array(idx))) :+ None
          val ndarrayIndicesOther = indexIterator(
            other.shape.dropRight(
              2
            ) :+ other.shape.last
          )
          ndarrayIndicesOther.map { ndarrayIndexOther =>
            val sliceIndicesOtherIntermediate =
              ndarrayIndexOther.map(idx => Some(Array(idx)))
            val sliceIndicesOther = sliceIndicesOtherIntermediate.slice(
              0,
              sliceIndicesOtherIntermediate.length - 2
            ) ++ List(None, sliceIndicesOtherIntermediate.last)
            (slice(sliceIndicesThis).squeeze() dot other
              .slice(sliceIndicesOther)
              .squeeze()).flatten().head
          }
        }
        NDArray[T](newElements.toArray).reshape(resultShape)
      }
    if (shape.length == 1 && other.shape.length == 1) vectorInnerProduct()
    else if (shape.length == 2 && other.shape.length == 2) matmul(other)
    else if (other.shape.length == 1) lastAxisInnerProduct()
    else if (shape.length > 1 && other.shape.length > 1)
      multidimensionalInnerProduct()
    else throw new ShapeException("dot undefined for these shapes")
  }

  /** Maps a function to every element in the NDArray, preserving the shape.
    *
    * @param f
    *   The function to apply to every element in the array.
    * @tparam B
    *   The return type of the map function.
    */
  def map[B: ClassTag](f: T => B): NDArray[B] =
    NDArray(flatten().map(f)).reshape(shape)

  /** Returns a new NDArray by reducing slices on the given axis.
    *
    * @param f
    *   A function that takes a 1D array as input and produces a single output.
    * @param axis
    *   The axis along which to take slices of the array. These slices are
    *   passed to the reduction function f. If axis is 0, the reduction function
    *   is applied on slices (None, i, j, ...) for all dimensions i, j, ...
    * @param keepDims
    *   If true, do not eliminate the reduced axis; keep it with size 1.
    * @tparam B
    *   The return type of the reduce function.
    * @return
    *   The reduced array. The axis dimension will be eliminated in the
    *   reduction. Reducing with a summation function would collapse the
    *   dimension by summing all elements in each slice.
    */
  def reduce[B: ClassTag](
      f: NDArray[T] => B,
      axis: Int,
      keepDims: Boolean = false
  ): NDArray[B] = {
    val ndarrayIndices = indexIterator(
      shape.indices
        .map(idx =>
          if (idx == axis) 1
          else shape(idx)
        )
        .toArray
    )
    val sliceIndices = ndarrayIndices.map(ndarrayIndex =>
      ndarrayIndex.indices
        .map(idx =>
          if (idx == axis) None
          else Some(Array(ndarrayIndex(idx)))
        )
        .toArray
    )
    val slices = sliceIndices.map(slice)
    val newElements = slices.map(f)
    val newShape = shape.indices
      .flatMap(idx =>
        if (idx == axis && (keepDims || shape.length == 1)) Some(1)
        else if (idx == axis) None
        else Some(shape(idx))
      )
      .toArray
    NDArray[B](newElements.toArray).reshape(newShape)
  }

  /** Converts an Int array of classes to one-hot encoded binary vectors.
    *
    * If this array is not NDArray[Int], it is first converted to that type.
    *
    * @param numClasses
    *   The number of classes in the dataset. If None, this function assumes it
    *   is the max of the array plus 1.
    * @return
    *   An array of one-hot encoded binary vectors. The output has one greater
    *   rank than the input. This last dimension is for the one-hot vectors.
    */
  def toCategorical(
      numClasses: Option[Int] = None
  )(implicit num: Numeric[T]): NDArray[Int] = classTag[T] match {
    case _ if classTag[T] == classTag[Int] =>
      val oneHotLength =
        numClasses.getOrElse(elements.asInstanceOf[Array[Int]].max + 1)
      NDArray(
        elements.flatMap(classIdx =>
          Array.tabulate(oneHotLength)(arrIdx =>
            if (arrIdx == classIdx.asInstanceOf[Int]) 1 else 0
          )
        )
      ).reshape(shape :+ oneHotLength)
    case _ => toInt.toCategorical(numClasses = numClasses)
  }

  /** Returns the string representation of the NDArray. */
  override def toString: String =
    flatten().mkString("[", ", ", "]") + shape.mkString("(", " x ", ")")
}
