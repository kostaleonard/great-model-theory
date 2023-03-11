package ndarray

import exceptions.ShapeException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.scalatest.concurrent.TimeLimits
import org.scalatest.time.SpanSugar._

import scala.language.postfixOps

class NDArraySpec extends AnyFlatSpec with Matchers with TimeLimits {

  /** Returns the mean execution time of the function over a number of trials.
    *
    * @param f
    *   The function to profile.
    * @param trials
    *   The number of times to execute the function. The result will be the mean
    *   execution time over all trials.
    * @tparam T
    *   The returns type of the function. Unused.
    */
  private def getMeanExecutionTimeMilliseconds[T](f: () => T, trials: Int = 1000): Double = {
    val t0 = System.nanoTime()
    (0 until trials).foreach(_ => f())
    val t1 = System.nanoTime()
    val elapsed = (t1 - t0) / (trials * 1e6)
    elapsed
  }

  "An N-dimensional array" should "have the correct number of elements (rank 2)" in {
    val arr = NDArray.zeros[Int](Array(2, 3))
    assert(arr.flatten().length == 6)
  }

  it should "have the correct number of elements (rank 5)" in {
    val arr = NDArray.zeros[Int](Array(2, 3, 4, 5, 6))
    assert(arr.flatten().length == 2 * 3 * 4 * 5 * 6)
  }

  it should "return the elements as a 1D array when flattened" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    assert(arr.flatten() sameElements Array(0, 1, 2, 3, 4, 5))
  }

  it should "return that it is empty" in {
    val arr = NDArray.empty[Int]
    assert(arr.isEmpty)
    assert(!arr.nonEmpty)
  }

  it should "return that it is empty when created with empty shape" in {
    val arr = NDArray.arange[Int](Array(2, 0))
    assert(arr.isEmpty)
    assert(!arr.nonEmpty)
  }

  it should "return that it is not empty" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    assert(!arr.isEmpty)
    assert(arr.nonEmpty)
  }

  it should "return the element at the given indices" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    assert(arr1(Array(0, 0)) == 0)
    assert(arr1(Array(0, 1)) == 1)
    assert(arr1(Array(1, 0)) == 3)
    assert(arr1(Array(1, 2)) == 5)
    val arr2 = NDArray.arange[Int](Array(2, 3, 4))
    assert(arr2(Array(0, 2, 1)) == 9)
    assert(arr2(Array(1, 1, 0)) == 16)
  }

  it should "return an NDArray with the same elements but new shape when reshaped" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = arr1.reshape(Array(3, 2))
    assert(arr2.shape sameElements Array(3, 2))
    assert(arr1.flatten() sameElements arr2.flatten())
  }

  it should "throw an error if reshaped to a size that does not match the number of elements" in {
    val arr = NDArray.zeros[Int](Array(2))
    assertThrows[ShapeException](arr.reshape(Array(2, 2)))
  }

  it should "convert a numeric data type to Float" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    val expected = NDArray.arange[Float](Array(2, 3))
    assert(arr.toFloat arrayApproximatelyEquals expected)
  }

  it should "convert a numeric data type to Double" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    val expected = NDArray.arange[Double](Array(2, 3))
    assert(arr.toDouble arrayApproximatelyEquals expected)
  }

  it should "convert a numeric data type to Int" in {
    val arr = NDArray.arange[Float](Array(2, 3))
    val expected = NDArray.arange[Int](Array(2, 3))
    assert(arr.toInt arrayEquals expected)
  }

  it should "convert a numeric data type to Long" in {
    val arr = NDArray.arange[Float](Array(2, 3))
    val expected = NDArray.arange[Long](Array(2, 3))
    assert(arr.toLong arrayEquals expected)
  }

  it should "be able to contain arbitrary data types" in {
    val arr1 = NDArray.ofValue[String](Array(2, 3), "hello")
    assert(arr1.flatten().forall(_.equals("hello")))
    case class Hello(x: Int)
    val arr2 = NDArray[Hello](List(Hello(0), Hello(1)))
    assert(arr2.flatten().forall(_.isInstanceOf[Hello]))
  }

  it should "return an array of all indices in order" in {
    val arr = NDArray.zeros[Int](Array(2, 3, 2))
    val indices = arr.indices.toArray
    val expected = Array(
      Array(0, 0, 0),
      Array(0, 0, 1),
      Array(0, 1, 0),
      Array(0, 1, 1),
      Array(0, 2, 0),
      Array(0, 2, 1),
      Array(1, 0, 0),
      Array(1, 0, 1),
      Array(1, 1, 0),
      Array(1, 1, 1),
      Array(1, 2, 0),
      Array(1, 2, 1)
    )
    assert(
      indices.indices.forall(elementIdx =>
        indices(elementIdx) sameElements expected(elementIdx)
      )
    )
  }

  it should "return the indices in near-constant time" in {
    val executionTimeDifferenceBuffer = 10
    val arr1 = NDArray.zeros[Int](Array(2, 3, 2))
    val meanTimeArr1 = getMeanExecutionTimeMilliseconds(() => arr1.indices)
    val arr2 = NDArray.zeros[Int](Array(60000, 28, 28))
    //TODO scalatest failAfter doesn't terminate the test after the time limit--make issue
    failAfter(100 millis) {
      val meanTimeArr2 = getMeanExecutionTimeMilliseconds(() => arr2.indices)
      assert(meanTimeArr2 < meanTimeArr1 * executionTimeDifferenceBuffer)
    }
  }

  it should "return an array with one element updated" in {
    val arr1 = NDArray.zeros[Int](Array(2, 3, 2))
    val arr2 = arr1.updated(Array(0, 1, 0), 1)
    assert(arr2.flatten().count(_ == 0) == arr2.shape.product - 1)
    assert(arr2(Array(0, 1, 0)) == 1)
  }

  it should "be equal in comparison with an array of the same shape and elements" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(2, 3))
    assert(arr1 arrayEquals arr2)
  }

  it should "not be equal in comparison with an array of different elements" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.zeros[Int](Array(2, 3))
    assert(arr1 arrayNotEquals arr2)
  }

  it should "not be equal in comparison with an array of different shape" in {
    val arr1 = NDArray.zeros[Int](Array(2, 3))
    val arr2 = NDArray.zeros[Int](Array(3, 2))
    assert(arr1 arrayNotEquals arr2)
  }

  it should "produce a mask of all true on == comparison with an array of the same shape and elements" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(2, 3))
    val mask = arr1 == arr2
    assert(mask.shape sameElements arr1.shape)
    assert(mask.flatten().forall(identity))
  }

  it should "produce a mask of only the equal elements on == comparison with an array of the same shape and different elements" in {
    val arr1 = NDArray(List(0, 1, 2, 3, 4, 5))
    val arr2 = NDArray(List(0, 1, 9, 9, 4, 9))
    val mask = arr1 == arr2
    assert(mask.shape sameElements arr1.shape)
    assert(mask(Array(0)))
    assert(mask(Array(1)))
    assert(!mask(Array(2)))
    assert(!mask(Array(3)))
    assert(mask(Array(4)))
    assert(!mask(Array(5)))
  }

  it should "produce a mask on == comparison with arrays that can be broadcast together (3 x 1, 3)" in {
    val arr1 = NDArray(List(0, 1, 2)).reshape(Array(3, 1))
    val arr2 = NDArray(List(1, 1, 1))
    val mask = arr1 == arr2
    assert(mask.shape sameElements Array(3, 3))
    assert(!mask(Array(0, 0)))
    assert(!mask(Array(0, 1)))
    assert(!mask(Array(0, 2)))
    assert(mask(Array(1, 0)))
    assert(mask(Array(1, 1)))
    assert(mask(Array(1, 2)))
    assert(!mask(Array(2, 0)))
    assert(!mask(Array(2, 1)))
    assert(!mask(Array(2, 2)))
  }

  it should "produce a 1-element array on == comparison with arrays that cannot be broadcast together" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    assertThrows[ShapeException](arr1 == arr2)
  }

  it should "be approximately equal in comparison with an array of the same shape and similar elements" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.000001, 0.9999999, 2.0, 3.00000001, 4))
    assert(arr1 arrayApproximatelyEquals arr2)
  }

  it should "be approximately equal in comparison with an array of the same shape and large epsilon" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.49, 0.8, 2.2, 2.95, 4))
    assert(arr1.arrayApproximatelyEquals(arr2, epsilon = 0.5))
  }

  it should "not be approximately equal in comparison with an array of the same shape and dissimilar elements" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.1, 0.9999999, 2.0, 3.00000001, 4))
    assert(arr1 arrayNotApproximatelyEquals arr2)
  }

  it should "not be approximately equal in comparison with an array of different shape" in {
    val arr1 = NDArray.zeros[Double](Array(2, 3))
    val arr2 = NDArray.zeros[Double](Array(3, 2))
    assert(arr1 arrayNotApproximatelyEquals arr2)
  }

  it should "produce a mask of all true on ~= comparison with an array of the same shape and similar elements" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.000001, 0.9999999, 2.0, 3.00000001, 4))
    val mask = arr1 ~= arr2
    assert(mask.shape sameElements arr1.shape)
    assert(mask.flatten().forall(identity))
  }

  it should "produce a mask of all true on ~= comparison with an array of the same shape and large epsilon" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.49, 0.8, 2.2, 2.95, 4))
    val mask = arr1.~=(arr2, epsilon = 0.5)
    assert(mask.shape sameElements arr1.shape)
    assert(mask.flatten().forall(identity))
  }

  it should "produce a mask of only the similar elements on ~= comparison with an array of the same shape and different elements" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.1, 0.9999999, 2.0, 3.00000001, 5))
    val mask = arr1 ~= arr2
    assert(mask.shape sameElements arr1.shape)
    assert(!mask(Array(0)))
    assert(mask(Array(1)))
    assert(mask(Array(2)))
    assert(mask(Array(3)))
    assert(!mask(Array(4)))
  }

  it should "produce a mask on ~= comparison with arrays that can be broadcast together (3 x 1, 3)" in {
    val arr1 = NDArray[Double](List(0, 1, 2)).reshape(Array(3, 1))
    val arr2 = NDArray[Double](List(0.1, 0.9999999, 2.0))
    val mask = arr1 ~= arr2
    assert(mask.shape sameElements Array(3, 3))
    assert(!mask(Array(0, 0)))
    assert(!mask(Array(0, 1)))
    assert(!mask(Array(0, 2)))
    assert(!mask(Array(1, 0)))
    assert(mask(Array(1, 1)))
    assert(!mask(Array(1, 2)))
    assert(!mask(Array(2, 0)))
    assert(!mask(Array(2, 1)))
    assert(mask(Array(2, 2)))
  }

  it should "produce a mask on ~= comparison with arrays that can be broadcast together and large epsilon (2 x 1, 2)" in {
    val arr1 = NDArray[Double](List(0, 1)).reshape(Array(2, 1))
    val arr2 = NDArray[Double](List(0.1, 0.9999999))
    val mask = arr1.~=(arr2, epsilon = 0.5)
    assert(mask.shape sameElements Array(2, 2))
    assert(mask(Array(0, 0)))
    assert(!mask(Array(0, 1)))
    assert(!mask(Array(1, 0)))
    assert(mask(Array(1, 1)))
  }

  it should "fail on ~= comparison with arrays that cannot be broadcast together" in {
    val arr1 = NDArray.arange[Double](Array(2, 3))
    val arr2 = NDArray.arange[Double](Array(3, 2))
    assertThrows[ShapeException](arr1 ~= arr2)
  }

  it should "be approximately equal when using Floats" in {
    val arr1 = NDArray[Float](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Float](List(0.000001f, 0.9999999f, 2.0f, 3.00000001f, 4))
    assert(arr1 arrayApproximatelyEquals arr2)
  }

  it should "be approximately equal when using Ints" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Int](List(0, 1, 3, 2, 4))
    assert(arr1.arrayApproximatelyEquals(arr2, epsilon = 2.0))
  }

  it should "be approximately equal when using Longs" in {
    val arr1 = NDArray[Long](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Long](List(0, 1, 3, 2, 4))
    assert(arr1.arrayApproximatelyEquals(arr2, epsilon = 2.0))
  }

  it should "broadcast an array to the target shape (1 => 2)" in {
    val arr = NDArray[Int](List(0))
    val targetShape = Array(2)
    val broadcast = arr.broadcastTo(targetShape)
    val expectedArrBroadcast = NDArray[Int](List(0, 0))
    assert(broadcast arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (1 => 2 x 2)" in {
    val arr = NDArray[Int](List(0))
    val targetShape = Array(2, 2)
    val broadcast = arr.broadcastTo(targetShape)
    val expectedArrBroadcast =
      NDArray[Int](List(0, 0, 0, 0)).reshape(targetShape)
    assert(broadcast arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (2 => 3 x 2)" in {
    val arr = NDArray[Int](List(0, 1))
    val targetShape = Array(3, 2)
    val broadcast = arr.broadcastTo(targetShape)
    val expectedArrBroadcast =
      NDArray[Int](List(0, 1, 0, 1, 0, 1)).reshape(targetShape)
    assert(broadcast arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (3 x 2 => 2 x 3 x 2)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(3, 2))
    val targetShape = Array(2, 3, 2)
    val broadcast = arr.broadcastTo(targetShape)
    val expectedArrBroadcast = NDArray[Int](
      List(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5)
    ).reshape(targetShape)
    assert(broadcast arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (3 x 1 => 3 x 3)" in {
    val arr = NDArray[Int](List(0, 1, 2)).reshape(Array(3, 1))
    val targetShape = Array(3, 3)
    val broadcast = arr.broadcastTo(targetShape)
    val expectedArrBroadcast =
      NDArray[Int](List(0, 0, 0, 1, 1, 1, 2, 2, 2)).reshape(targetShape)
    assert(broadcast arrayEquals expectedArrBroadcast)
  }

  it should "fail to broadcast an array to an invalid target shape (2 => 2 x 3)" in {
    val arr = NDArray[Int](List(0, 1))
    val targetShape = Array(2, 3)
    assertThrows[ShapeException](arr.broadcastTo(targetShape))
  }

  it should "fail to broadcast an array to an invalid target shape (2 x 3 => 3)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val targetShape = Array(3)
    assertThrows[ShapeException](arr.broadcastTo(targetShape))
  }

  it should "fail to broadcast an array to an invalid target shape (2 x 3 => 2 x 1)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val targetShape = Array(2, 1)
    assertThrows[ShapeException](arr.broadcastTo(targetShape))
  }

  it should "fail to broadcast an array to an invalid target shape (1 x 2 x 3 => 5 x 1 x 3)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(1, 2, 3))
    val targetShape = Array(5, 1, 3)
    assertThrows[ShapeException](arr.broadcastTo(targetShape))
  }

  it should "broadcast two arrays to matching shape (2 x 3, 1)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val arr2 = NDArray[Int](List(0))
    val broadcast = arr1 broadcastWith arr2
    val expectedArr2Broadcast =
      NDArray[Int](List(0, 0, 0, 0, 0, 0)).reshape(Array(2, 3))
    assert(broadcast._1 arrayEquals arr1)
    assert(broadcast._2 arrayEquals expectedArr2Broadcast)
  }

  it should "broadcast two arrays to matching shape (2 x 3, 3)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val arr2 = NDArray[Int](List(0, 1, 2))
    val broadcast = arr1 broadcastWith arr2
    val expectedArr2Broadcast =
      NDArray[Int](List(0, 1, 2, 0, 1, 2)).reshape(Array(2, 3))
    assert(broadcast._1 arrayEquals arr1)
    assert(broadcast._2 arrayEquals expectedArr2Broadcast)
  }

  it should "broadcast two arrays to matching shape (3 x 2, 2)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(3, 2))
    val arr2 = NDArray[Int](List(0, 1))
    val broadcast = arr1 broadcastWith arr2
    val expectedArr2Broadcast =
      NDArray[Int](List(0, 1, 0, 1, 0, 1)).reshape(Array(3, 2))
    assert(broadcast._1 arrayEquals arr1)
    assert(broadcast._2 arrayEquals expectedArr2Broadcast)
  }

  it should "broadcast two arrays to matching shape (3, 2 x 3)" in {
    val arr1 = NDArray[Int](List(0, 1, 2))
    val arr2 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val broadcast = arr1 broadcastWith arr2
    val expectedArr1Broadcast =
      NDArray[Int](List(0, 1, 2, 0, 1, 2)).reshape(Array(2, 3))
    assert(broadcast._1 arrayEquals expectedArr1Broadcast)
    assert(broadcast._2 arrayEquals arr2)
  }

  it should "broadcast two arrays to matching shape (3 x 1, 1 x 3)" in {
    // Example broadcast taken from https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val broadcast = arr1 broadcastWith arr2
    val expectedArr1Broadcast =
      NDArray[Int](List(1, 1, 1, 2, 2, 2, 3, 3, 3)).reshape(Array(3, 3))
    val expectedArr2Broadcast =
      NDArray[Int](List(4, 5, 6, 4, 5, 6, 4, 5, 6)).reshape(Array(3, 3))
    assert(broadcast._1 arrayEquals expectedArr1Broadcast)
    assert(broadcast._2 arrayEquals expectedArr2Broadcast)
    val broadcastSum = broadcast._1 + broadcast._2
    val expectedSum =
      NDArray[Int](List(5, 6, 7, 6, 7, 8, 7, 8, 9)).reshape(Array(3, 3))
    assert(broadcastSum arrayEquals expectedSum)
  }

  it should "broadcast two arrays to matching shape (8 x 1 x 6 x 1, 7 x 1 x 5)" in {
    val arr1 = NDArray.arange[Int](Array(8, 1, 6, 1))
    val arr2 = NDArray.arange[Int](Array(7, 1, 5))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast._1.shape sameElements Array(8, 7, 6, 5))
    assert(broadcast._2.shape sameElements Array(8, 7, 6, 5))
    // Computed this sum with NumPy.
    val broadcastSum = broadcast._1 + broadcast._2
    val sumSlice = broadcastSum
      .slice(Array(Some(Array(1)), Some(Array(2)), Some(Array(3)), None))
      .squeeze()
    val expectedSlice = NDArray[Int](List(19, 20, 21, 22, 23))
    assert(sumSlice arrayEquals expectedSlice)
  }

  it should "fail to broadcast two incompatible arrays (3, 4)" in {
    val arr1 = NDArray.arange[Int](Array(3))
    val arr2 = NDArray.arange[Int](Array(4))
    assertThrows[ShapeException](arr1 broadcastWith arr2)
  }

  it should "fail to broadcast two incompatible arrays (2 x 1, 8 x 4 x 3)" in {
    val arr1 = NDArray.arange[Int](Array(2, 1))
    val arr2 = NDArray.arange[Int](Array(8, 4, 3))
    assertThrows[ShapeException](arr1 broadcastWith arr2)
  }

  it should "define + for element-wise addition (Int)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Int](List(1, 1, 3, 2, 4))
    val addition = arr1 + arr2
    assert(addition.flatten() sameElements Array(1, 2, 5, 5, 8))
  }

  it should "define + for element-wise addition (Double)" in {
    val arr1 = NDArray[Double](List(0.0, 1.0, 2.0, 3.0, 4.0))
    val arr2 = NDArray[Double](List(1.0, 1.1, 3.0, 2.7, 4.5))
    val addition = arr1 + arr2
    assert(
      addition arrayApproximatelyEquals NDArray[Double](
        List(1.0, 2.1, 5.0, 5.7, 8.5)
      )
    )
  }

  it should "retain the same shape in element-wise addition" in {
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.arange[Int](Array(2, 3, 4))
    val addition = arr1 + arr2
    assert(addition.shape sameElements Array(2, 3, 4))
  }

  it should "broadcast arrays in element-wise addition (1, 2 x 2)" in {
    val arr1 = NDArray[Int](List(1))
    val arr2 = NDArray[Int](List(0, 1, 2, 3)).reshape(Array(2, 2))
    val addition = arr1 + arr2
    val expectedSum =
      NDArray[Int](List(1, 2, 3, 4)).reshape(Array(2, 2))
    assert(addition arrayEquals expectedSum)
  }

  it should "broadcast arrays in element-wise addition (3 x 1, 1 x 3)" in {
    // Example broadcast taken from https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val addition = arr1 + arr2
    val expectedSum =
      NDArray[Int](List(5, 6, 7, 6, 7, 8, 7, 8, 9)).reshape(Array(3, 3))
    assert(addition arrayEquals expectedSum)
  }

  it should "broadcast arrays in element-wise addition (8 x 1 x 6 x 1, 7 x 1 x 5)" in {
    val arr1 = NDArray.arange[Int](Array(8, 1, 6, 1))
    val arr2 = NDArray.arange[Int](Array(7, 1, 5))
    val addition = arr1 + arr2
    // Computed this sum with NumPy.
    val sumSlice = addition
      .slice(Array(Some(Array(1)), Some(Array(2)), Some(Array(3)), None))
      .squeeze()
    val expectedSlice = NDArray[Int](List(19, 20, 21, 22, 23))
    assert(sumSlice arrayEquals expectedSlice)
  }

  it should "fail to perform element-wise addition on arrays with different shape" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    assertThrows[ShapeException](arr1 + arr2)
  }

  it should "define - for element-wise subtraction (Int)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Int](List(1, 1, 3, 2, 4))
    val subtraction = arr1 - arr2
    assert(subtraction.flatten() sameElements Array(-1, 0, -1, 1, 0))
  }

  it should "define - for element-wise subtraction (Double)" in {
    val arr1 = NDArray[Double](List(0.0, 1.0, 2.0, 3.0, 4.0))
    val arr2 = NDArray[Double](List(1.0, 1.1, 3.0, 2.7, 4.5))
    val subtraction = arr1 - arr2
    assert(
      subtraction arrayApproximatelyEquals NDArray[Double](
        List(-1.0, -0.1, -1.0, 0.3, -0.5)
      )
    )
  }

  it should "retain the same shape in element-wise subtraction" in {
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.arange[Int](Array(2, 3, 4))
    val subtraction = arr1 - arr2
    assert(subtraction.shape sameElements Array(2, 3, 4))
  }

  it should "broadcast arrays in element-wise subtraction (1, 2 x 2)" in {
    val arr1 = NDArray[Int](List(1))
    val arr2 = NDArray[Int](List(0, 1, 2, 3)).reshape(Array(2, 2))
    val subtraction = arr1 - arr2
    val expectedSum =
      NDArray[Int](List(1, 0, -1, -2)).reshape(Array(2, 2))
    assert(subtraction arrayEquals expectedSum)
  }

  it should "broadcast arrays in element-wise subtraction (3 x 1, 1 x 3)" in {
    // Example broadcast taken from https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val subtraction = arr1 - arr2
    val expectedSum =
      NDArray[Int](List(-3, -4, -5, -2, -3, -4, -1, -2, -3)).reshape(
        Array(3, 3)
      )
    assert(subtraction arrayEquals expectedSum)
  }

  it should "fail to perform element-wise subtraction on arrays with different shape" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    assertThrows[ShapeException](arr1 - arr2)
  }

  it should "define * for element-wise multiplication" in {
    val arr1 = NDArray[Int](List(2, 3, 4))
    val arr2 = NDArray[Int](List(2, 3, 4))
    val multiplication = arr1 * arr2
    assert(multiplication arrayEquals NDArray(List(4, 9, 16)))
  }

  it should "broadcast arrays in element-wise multiplication (3 x 1, 1 x 3)" in {
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val multiplication = arr1 * arr2
    val expected =
      NDArray[Int](List(4, 5, 6, 8, 10, 12, 12, 15, 18)).reshape(
        Array(3, 3)
      )
    assert(multiplication arrayEquals expected)
  }

  it should "fail to perform element-wise multiplication on arrays with mismatching shape" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    assertThrows[ShapeException](arr1 * arr2)
  }

  it should "define / for element-wise division" in {
    val arr1 = NDArray[Float](List(3, 6, 7))
    val arr2 = NDArray[Float](List(2, 3, 4))
    val division = arr1 / arr2
    assert(
      division arrayApproximatelyEquals NDArray[Float](List(1.5f, 2, 1.75f))
    )
  }

  it should "broadcast arrays in element-wise division (3 x 1, 1 x 3)" in {
    val arr1 = NDArray[Float](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Float](List(1, 2, 3)).reshape(Array(1, 3))
    val division = arr1 / arr2
    val expected =
      NDArray[Float](List(1, 0.5f, 0.33333f, 2, 1, 0.66667f, 3, 1.5f, 1))
        .reshape(
          Array(3, 3)
        )
    assert(division arrayApproximatelyEquals expected)
  }

  it should "fail to perform element-wise division on arrays with mismatching shape" in {
    val arr1 = NDArray.arange[Float](Array(2, 3))
    val arr2 = NDArray.arange[Float](Array(3, 2))
    assertThrows[ShapeException](arr1 / arr2)
  }

  it should "overload / to allow division with the array element type" in {
    val arr = NDArray[Float](List(2, 4, 6))
    val division = arr / 2
    assert(division arrayApproximatelyEquals NDArray[Float](List(1, 2, 3)))
  }

  it should "return the sum of all elements" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4))
    assert(arr.sum == 10)
  }

  it should "return the sum along an axis" in {
    val arr = NDArray.arange[Int](Array(2, 3, 2))
    val sumAxis0 = arr.sumAxis(0)
    val expectedSumAxis0 =
      NDArray(List(6, 8, 10, 12, 14, 16)).reshape(Array(3, 2))
    assert(sumAxis0 arrayEquals expectedSumAxis0)
    val sumAxis1 = arr.sumAxis(1)
    val expectedSumAxis1 = NDArray(List(6, 9, 24, 27)).reshape(Array(2, 2))
    assert(sumAxis1 arrayEquals expectedSumAxis1)
    val sumAxis2 = arr.sumAxis(2)
    val expectedSumAxis2 =
      NDArray(List(1, 5, 9, 13, 17, 21)).reshape(Array(2, 3))
    assert(sumAxis2 arrayEquals expectedSumAxis2)
  }

  it should "return the sum along an axis, preserving dimensions" in {
    val arr = NDArray.arange[Int](Array(2, 3, 2))
    val sumAxis0 = arr.sumAxis(0, keepDims = true)
    val expectedSumAxis0 =
      NDArray(List(6, 8, 10, 12, 14, 16)).reshape(Array(1, 3, 2))
    assert(sumAxis0 arrayEquals expectedSumAxis0)
    val sumAxis1 = arr.sumAxis(1, keepDims = true)
    val expectedSumAxis1 = NDArray(List(6, 9, 24, 27)).reshape(Array(2, 1, 2))
    assert(sumAxis1 arrayEquals expectedSumAxis1)
    val sumAxis2 = arr.sumAxis(2, keepDims = true)
    val expectedSumAxis2 =
      NDArray(List(1, 5, 9, 13, 17, 21)).reshape(Array(2, 3, 1))
    assert(sumAxis2 arrayEquals expectedSumAxis2)
  }

  it should "return the mean of all elements" in {
    val arr = NDArray[Float](List(0, 1, 2, 3, 4))
    assert(arr.mean == 2)
  }

  it should "return 0 for the mean of an empty array" in {
    val arr = NDArray.empty[Float]
    assert(arr.mean == 0)
  }

  it should "return the square of all elements" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val expected = NDArray[Int](List(0, 1, 4, 9, 16, 25)).reshape(Array(2, 3))
    assert(arr.square arrayEquals expected)
  }

  it should "return the reciprocal of all elements" in {
    val arr = NDArray[Double](List(1, 2, 3, 4, 5, 6, 7, 8)).reshape(Array(2, 4))
    val expected = NDArray[Double](
      List(1, 0.5, 0.3333333333333333, 0.25, 0.2, 0.1666666,
        0.14285714285714285, 0.125)
    ).reshape(Array(2, 4))
    assert(arr.reciprocal arrayApproximatelyEquals expected)
  }

  it should "return infinity for the reciprocal of 0" in {
    val arr = NDArray[Double](List(0, 1))
    val expected = NDArray[Double](List(Double.PositiveInfinity, 1))
    assert(arr.reciprocal arrayEquals expected)
  }

  it should "return the negation of all elements" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4))
    val expected = NDArray[Int](List(0, -1, -2, -3, -4))
    assert(arr.negate arrayEquals expected)
  }

  it should "return the exponentiation of all elements" in {
    val arr = NDArray[Double](List(0, 1, 2, -3, 4))
    val expected = NDArray[Double](
      List(1, Math.exp(1.0), Math.exp(2.0), Math.exp(-3.0), Math.exp(4.0))
    )
    assert(arr.exp arrayApproximatelyEquals expected)
  }

  it should "return the same 1D array when transposed" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4))
    assert(arr.transpose arrayEquals arr)
  }

  it should "return the transposed array (2D)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val expected = NDArray[Int](List(0, 3, 1, 4, 2, 5)).reshape(Array(3, 2))
    assert(arr.transpose arrayEquals expected)
  }

  it should "return the transposed array (3D)" in {
    val arr = NDArray.arange[Int](Array(2, 3, 2))
    val expected = NDArray[Int](List(0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11))
      .reshape(Array(2, 3, 2))
    assert(arr.transpose arrayEquals expected)
  }

  it should "remove length 1 dimensions when squeezed (rank 3)" in {
    val arr = NDArray.arange[Int](Array(2, 1, 3))
    val squeezed = arr.squeeze()
    assert(squeezed.shape sameElements Array(2, 3))
    assert(squeezed.flatten() sameElements arr.flatten())
  }

  it should "remove length 1 dimensions when squeezed (rank 5)" in {
    val arr = NDArray.arange[Int](Array(2, 1, 1, 3, 1))
    val squeezed = arr.squeeze()
    assert(squeezed.shape sameElements Array(2, 3))
    assert(squeezed.flatten() sameElements arr.flatten())
  }

  it should "leave arrays with no length 1 dimensions unchanged when squeezed" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    val squeezed = arr.squeeze()
    assert(arr arrayEquals squeezed)
  }

  it should "return all elements when provided None for each dimension in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(Array(None, None, None))
    assert(sliced arrayEquals arr)
  }

  it should "return an array with the rows requested in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(Array(Some(Array(0)), None, None))
    assert(sliced.shape sameElements Array(1, 3, 4))
    assert(sliced.flatten() sameElements (0 until 12))
  }

  it should "return an array with the columns requested in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(Array(None, Some(Array(1, 2)), None))
    assert(sliced.shape sameElements Array(2, 2, 4))
    assert(
      sliced.flatten() sameElements Array(4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18,
        19, 20, 21, 22, 23)
    )
  }

  it should "return an array with the elements requested in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(Array(None, Some(Array(1, 2)), Some(Array(0))))
    assert(sliced.shape sameElements Array(2, 2, 1))
    assert(sliced.flatten() sameElements Array(4, 8, 16, 20))
  }

  it should "return the matrix multiplication of two 2D arrays" in {
    // Example multiplication taken from https://en.wikipedia.org/wiki/Matrix_multiplication
    val arr1 =
      NDArray[Int](List(1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2)).reshape(
        Array(4, 3)
      )
    val arr2 =
      NDArray[Int](List(1, 2, 1, 2, 3, 1, 4, 2, 2)).reshape(Array(3, 3))
    val expectedResult = NDArray[Int](List(5, 4, 3, 8, 9, 5, 6, 5, 3, 11, 9, 6))
      .reshape(Array(4, 3))
    val matmulResult = arr1 matmul arr2
    assert(matmulResult arrayEquals expectedResult)
  }

  it should "fail to matrix multiply two 2D arrays with mismatching shapes" in {
    val arr1 = NDArray.ones[Int](Array(3, 2))
    val arr2 = NDArray.ones[Int](Array(3, 2))
    assertThrows[ShapeException](arr1 matmul arr2)
  }

  it should "fail to matrix multiply non-2D arrays" in {
    val arr1 = NDArray.ones[Int](Array(3, 2))
    val arr2 = NDArray.ones[Int](Array(2, 2, 2))
    assertThrows[ShapeException](arr1 matmul arr2)
  }

  it should "return the dot product of two 1D arrays" in {
    val arr1 = NDArray.arange[Int](Array(5))
    val arr2 = NDArray.ones[Int](Array(5))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.shape sameElements Array(1))
    assert(dotProduct(Array(0)) == 10)
  }

  it should "fail to return the dot product of two 1D arrays of different lengths" in {
    val arr1 = NDArray.arange[Int](Array(6))
    val arr2 = NDArray.ones[Int](Array(5))
    assertThrows[ShapeException](arr1 dot arr2)
  }

  it should "return the matrix multiplication of two 2D arrays using dot" in {
    // Example multiplication taken from https://en.wikipedia.org/wiki/Matrix_multiplication
    val arr1 =
      NDArray[Int](List(1, 0, 1, 2, 1, 1, 0, 1, 1, 1, 1, 2)).reshape(
        Array(4, 3)
      )
    val arr2 =
      NDArray[Int](List(1, 2, 1, 2, 3, 1, 4, 2, 2)).reshape(Array(3, 3))
    val expectedResult = NDArray[Int](List(5, 4, 3, 8, 9, 5, 6, 5, 3, 11, 9, 6))
      .reshape(Array(4, 3))
    val matmulResult = arr1 dot arr2
    assert(matmulResult arrayEquals expectedResult)
  }

  it should "return the inner products over the last axis of a multidimensional array (2D) and a 1D array using dot" in {
    // Example multiplication computed with np.dot.
    val arr1 = NDArray.arange[Int](Array(3, 4))
    val arr2 = NDArray.ones[Int](Array(4))
    val expectedResult = NDArray[Int](List(6, 22, 38))
    val dotProduct = arr1 dot arr2
    assert(dotProduct arrayEquals expectedResult)
  }

  it should "return the inner products over the last axis of a multidimensional array (3D) and a 1D array using dot" in {
    // Example multiplication computed with np.dot.
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.ones[Int](Array(4))
    val expectedResult =
      NDArray[Int](List(6, 22, 38, 54, 70, 86)).reshape(Array(2, 3))
    val dotProduct = arr1 dot arr2
    assert(dotProduct arrayEquals expectedResult)
  }

  it should "fail to return the dot product over the last axis of a multidimensional array and a 1D array using dot when the last axis shape does not match" in {
    val arr1 = NDArray.arange[Int](Array(4, 3))
    val arr2 = NDArray.ones[Int](Array(4))
    assertThrows[ShapeException](arr1 dot arr2)
  }

  it should "return the inner products of two multidimensional arrays using dot" in {
    // Example multiplication computed with np.dot.
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.arange[Int](Array(4, 2))
    val expectedResult =
      NDArray[Int](List(28, 34, 76, 98, 124, 162, 172, 226, 220, 290, 268, 354))
        .reshape(Array(2, 3, 2))
    val dotProduct = arr1 dot arr2
    assert(dotProduct arrayEquals expectedResult)
  }

  it should "fail to return the inner products of two multidimensional arrays using dot when shapes don't match" in {
    val arr1 = NDArray.arange[Int](Array(2, 3, 2))
    val arr2 = NDArray.ones[Int](Array(4, 2))
    assertThrows[ShapeException](arr1 dot arr2)
  }

  it should "fail to return the dot product on for shapes with no defined operation" in {
    val arr1 = NDArray.arange[Int](Array(3))
    val arr2 = NDArray.ones[Int](Array(3, 2))
    assertThrows[ShapeException](arr1 dot arr2)
  }

  it should "map a function to every element" in {
    val arr = NDArray.ones[Int](Array(2, 3))
    val mapped = arr.map(_ * 2)
    assert(mapped.shape sameElements arr.shape)
    assert(mapped.flatten().forall(_ == 2))
  }

  it should "reduce an array along an axis (axis 0)" in {
    val arr = NDArray.ones[Int](Array(2, 3))
    val reduced = arr.reduce(slice => slice.flatten().sum, 0)
    assert(reduced.shape sameElements Array(3))
    assert(reduced arrayEquals NDArray[Int](List(2, 2, 2)))
  }

  it should "reduce an array along an axis (axis 1)" in {
    val arr = NDArray.ones[Int](Array(2, 3))
    val reduced = arr.reduce(slice => slice.flatten().sum, 1)
    assert(reduced.shape sameElements Array(2))
    assert(reduced arrayEquals NDArray[Int](List(3, 3)))
  }

  it should "apply the reduction in order" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    val reduced = arr.reduce(slice => slice.flatten().head, 0)
    assert(reduced.shape sameElements Array(3))
    assert(reduced arrayEquals NDArray[Int](List(0, 1, 2)))
  }

  it should "preserve dimensions in reduction if specified" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    val reduced = arr.reduce(_.sum, 0, keepDims = true)
    assert(reduced.shape sameElements Array(1, 3))
    assert(reduced arrayEquals NDArray[Int](List(3, 5, 7)).reshape(Array(1, 3)))
  }

  it should "preserve dimensions in reduction if the array has only 1 dimension" in {
    val arr = NDArray.ones[Int](Array(3))
    val reduced = arr.reduce(_.sum, 0)
    assert(reduced arrayEquals NDArray[Int](List(3)))
  }

  it should "represent its elements in string form" in {
    val arr = NDArray.arange[Int](Array(2, 3))
    assert(arr.toString == "[0, 1, 2, 3, 4, 5](2 x 3)")
  }

  "An NDArray.empty array" should "have no elements" in {
    val arr = NDArray.empty[Int]
    assert(arr.flatten().isEmpty)
  }

  "An NDArray.ofValue array" should "contain only the given value" in {
    val arr = NDArray.ofValue[Int](Array(2, 3), 5)
    assert(arr.flatten().forall(_ == 5))
  }

  "An NDArray.zeros array" should "contain only zeros" in {
    val arr = NDArray.zeros[Int](Array(2, 3))
    assert(arr.flatten().forall(_ == 0))
  }

  it should "pass type parameter information correctly" in {
    val arr = NDArray.zeros[Float](Array(2, 3))
    assert(arr.flatten().forall(_.isInstanceOf[Float]))
  }

  "An NDArray.ones array" should "contain only ones" in {
    val arr = NDArray.ones[Int](Array(2, 3))
    assert(arr.flatten().forall(_ == 1))
  }

  it should "pass type parameter information correctly" in {
    val arr = NDArray.ones[Float](Array(2, 3))
    assert(arr.flatten().forall(_.isInstanceOf[Float]))
  }

  "An NDArray.apply array" should "convert a flat sequence into a rank 1 NDArray" in {
    val values = List(1, 2, 3, 4)
    val arr = NDArray[Int](values)
    assert(arr.shape sameElements Array(4))
    val elements = arr.flatten()
    assert(elements.indices.forall(idx => elements(idx) == values(idx)))
  }

  it should "convert an empty sequence into an empty NDArray" in {
    val arr = NDArray[Int](List.empty)
    assert(arr.flatten().isEmpty)
  }

  "An NDArray.arange array" should "contain elements (0, 1, 2, ...) as index increments" in {
    val arr = NDArray.arange[Int](Array(2, 3, 2))
    assert(arr(Array(0, 0, 0)) == 0)
    assert(arr(Array(0, 0, 1)) == 1)
    assert(arr(Array(0, 1, 0)) == 2)
    assert(arr(Array(0, 1, 1)) == 3)
    assert(arr(Array(0, 2, 0)) == 4)
    assert(arr(Array(0, 2, 1)) == 5)
    assert(arr(Array(1, 0, 0)) == 6)
    assert(arr(Array(1, 0, 1)) == 7)
    assert(arr(Array(1, 1, 0)) == 8)
    assert(arr(Array(1, 1, 1)) == 9)
    assert(arr(Array(1, 2, 0)) == 10)
    assert(arr(Array(1, 2, 1)) == 11)
  }

  it should "contain elements (0, 1, 2, ...) when flattened" in {
    val arr = NDArray.arange[Int](Array(2, 3, 2))
    val elements = arr.flatten()
    assert(elements.indices.forall(idx => elements(idx) == idx))
  }

  it should "pass type parameter information correctly" in {
    val arr = NDArray.arange[Float](Array(2, 3, 2))
    assert(arr.flatten().forall(_.isInstanceOf[Float]))
  }

  "An NDArray.random[Float] array" should "contain different elements in [0, 1)" in {
    val arr = NDArray.random[Float](Array(2, 3))
    assert(arr.flatten().forall(element => 0 <= element && element < 1))
    val head = arr(Array(0, 0))
    assert(!arr.flatten().forall(_ == head))
  }

  "An NDArray.random[Int] array" should "contain different elements in [-2 ^ 31, 2 ^ 31 - 1]" in {
    val arr = NDArray.random[Int](Array(2, 3))
    assert(
      arr
        .flatten()
        .forall(element => Int.MinValue <= element && element <= Int.MaxValue)
    )
    val head = arr(Array(0, 0))
    assert(!arr.flatten().forall(_ == head))
  }
}
