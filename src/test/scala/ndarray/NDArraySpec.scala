package ndarray

import exceptions.ShapeException
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NDArraySpec extends AnyFlatSpec with Matchers {

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

  it should "return the element at the given indices" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    assert(arr1(List(0, 0)) == 0)
    assert(arr1(List(0, 1)) == 1)
    assert(arr1(List(1, 0)) == 3)
    assert(arr1(List(1, 2)) == 5)
    val arr2 = NDArray.arange[Int](Array(2, 3, 4))
    assert(arr2(List(0, 2, 1)) == 9)
    assert(arr2(List(1, 1, 0)) == 16)
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

  it should "be able to contain arbitrary data types" in {
    val arr1 = NDArray.ofValue[String](Array(2, 3), "hello")
    assert(arr1.flatten().forall(_.equals("hello")))
    case class Hello(x: Int)
    val arr2 = NDArray[Hello](List(Hello(0), Hello(1)))
    assert(arr2.flatten().forall(_.isInstanceOf[Hello]))
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
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements arr1.shape)
    assert(mask.get.flatten().forall(identity))
  }

  it should "produce a mask of only the equal elements on == comparison with an array of the same shape and different elements" in {
    val arr1 = NDArray(List(0, 1, 2, 3, 4, 5))
    val arr2 = NDArray(List(0, 1, 9, 9, 4, 9))
    val mask = arr1 == arr2
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements arr1.shape)
    assert(mask.get.apply(List(0)))
    assert(mask.get.apply(List(1)))
    assert(!mask.get.apply(List(2)))
    assert(!mask.get.apply(List(3)))
    assert(mask.get.apply(List(4)))
    assert(!mask.get.apply(List(5)))
  }

  it should "produce a mask on == comparison with arrays that can be broadcast together (3 x 1, 3)" in {
    val arr1 = NDArray(List(0, 1, 2)).reshape(Array(3, 1))
    val arr2 = NDArray(List(1, 1, 1))
    val mask = arr1 == arr2
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements Array(3, 3))
    assert(!mask.get.apply(List(0, 0)))
    assert(!mask.get.apply(List(0, 1)))
    assert(!mask.get.apply(List(0, 2)))
    assert(mask.get.apply(List(1, 0)))
    assert(mask.get.apply(List(1, 1)))
    assert(mask.get.apply(List(1, 2)))
    assert(!mask.get.apply(List(2, 0)))
    assert(!mask.get.apply(List(2, 1)))
    assert(!mask.get.apply(List(2, 2)))
  }

  it should "produce a 1-element array on == comparison with arrays that cannot be broadcast together" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    val mask = arr1 == arr2
    assert(mask.isFailure)
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
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements arr1.shape)
    assert(mask.get.flatten().forall(identity))
  }

  it should "produce a mask of all true on ~= comparison with an array of the same shape and large epsilon" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.49, 0.8, 2.2, 2.95, 4))
    val mask = arr1.~=(arr2, epsilon = 0.5)
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements arr1.shape)
    assert(mask.get.flatten().forall(identity))
  }

  it should "produce a mask of only the similar elements on ~= comparison with an array of the same shape and different elements" in {
    val arr1 = NDArray[Double](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Double](List(0.1, 0.9999999, 2.0, 3.00000001, 5))
    val mask = arr1 ~= arr2
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements arr1.shape)
    assert(!mask.get.apply(List(0)))
    assert(mask.get.apply(List(1)))
    assert(mask.get.apply(List(2)))
    assert(mask.get.apply(List(3)))
    assert(!mask.get.apply(List(4)))
  }

  it should "produce a mask on ~= comparison with arrays that can be broadcast together (3 x 1, 3)" in {
    val arr1 = NDArray[Double](List(0, 1, 2)).reshape(Array(3, 1))
    val arr2 = NDArray[Double](List(0.1, 0.9999999, 2.0))
    val mask = arr1 ~= arr2
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements Array(3, 3))
    assert(!mask.get.apply(List(0, 0)))
    assert(!mask.get.apply(List(0, 1)))
    assert(!mask.get.apply(List(0, 2)))
    assert(!mask.get.apply(List(1, 0)))
    assert(mask.get.apply(List(1, 1)))
    assert(!mask.get.apply(List(1, 2)))
    assert(!mask.get.apply(List(2, 0)))
    assert(!mask.get.apply(List(2, 1)))
    assert(mask.get.apply(List(2, 2)))
  }

  it should "produce a mask on ~= comparison with arrays that can be broadcast together and large epsilon (2 x 1, 2)" in {
    val arr1 = NDArray[Double](List(0, 1)).reshape(Array(2, 1))
    val arr2 = NDArray[Double](List(0.1, 0.9999999))
    val mask = arr1.~=(arr2, epsilon = 0.5)
    assert(mask.isSuccess)
    assert(mask.get.shape sameElements Array(2, 2))
    assert(mask.get.apply(List(0, 0)))
    assert(!mask.get.apply(List(0, 1)))
    assert(!mask.get.apply(List(1, 0)))
    assert(mask.get.apply(List(1, 1)))
  }

  it should "fail on ~= comparison with arrays that cannot be broadcast together" in {
    val arr1 = NDArray.arange[Double](Array(2, 3))
    val arr2 = NDArray.arange[Double](Array(3, 2))
    val mask = arr1 ~= arr2
    assert(mask.isFailure)
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
    assert(broadcast.isSuccess)
    val expectedArrBroadcast = NDArray[Int](List(0, 0))
    assert(broadcast.get arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (1 => 2 x 2)" in {
    val arr = NDArray[Int](List(0))
    val targetShape = Array(2, 2)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isSuccess)
    val expectedArrBroadcast =
      NDArray[Int](List(0, 0, 0, 0)).reshape(targetShape)
    assert(broadcast.get arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (2 => 3 x 2)" in {
    val arr = NDArray[Int](List(0, 1))
    val targetShape = Array(3, 2)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isSuccess)
    val expectedArrBroadcast =
      NDArray[Int](List(0, 1, 0, 1, 0, 1)).reshape(targetShape)
    assert(broadcast.get arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (3 x 2 => 2 x 3 x 2)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(3, 2))
    val targetShape = Array(2, 3, 2)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isSuccess)
    val expectedArrBroadcast = NDArray[Int](
      List(0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5)
    ).reshape(targetShape)
    assert(broadcast.get arrayEquals expectedArrBroadcast)
  }

  it should "broadcast an array to the target shape (3 x 1 => 3 x 3)" in {
    val arr = NDArray[Int](List(0, 1, 2)).reshape(Array(3, 1))
    val targetShape = Array(3, 3)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isSuccess)
    val expectedArrBroadcast =
      NDArray[Int](List(0, 0, 0, 1, 1, 1, 2, 2, 2)).reshape(targetShape)
    assert(broadcast.get arrayEquals expectedArrBroadcast)
  }

  it should "fail to broadcast an array to an invalid target shape (2 => 2 x 3)" in {
    val arr = NDArray[Int](List(0, 1))
    val targetShape = Array(2, 3)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isFailure)
  }

  it should "fail to broadcast an array to an invalid target shape (2 x 3 => 3)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val targetShape = Array(3)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isFailure)
  }

  it should "fail to broadcast an array to an invalid target shape (2 x 3 => 2 x 1)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val targetShape = Array(2, 1)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isFailure)
  }

  it should "fail to broadcast an array to an invalid target shape (1 x 2 x 3 => 5 x 1 x 3)" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(1, 2, 3))
    val targetShape = Array(5, 1, 3)
    val broadcast = arr.broadcastTo(targetShape)
    assert(broadcast.isFailure)
  }

  it should "broadcast two arrays to matching shape (2 x 3, 1)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val arr2 = NDArray[Int](List(0))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isSuccess)
    val expectedArr2Broadcast =
      NDArray[Int](List(0, 0, 0, 0, 0, 0)).reshape(Array(2, 3))
    assert(broadcast.get._1 arrayEquals arr1)
    assert(broadcast.get._2 arrayEquals expectedArr2Broadcast)
  }

  it should "broadcast two arrays to matching shape (2 x 3, 3)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val arr2 = NDArray[Int](List(0, 1, 2))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isSuccess)
    val expectedArr2Broadcast =
      NDArray[Int](List(0, 1, 2, 0, 1, 2)).reshape(Array(2, 3))
    assert(broadcast.get._1 arrayEquals arr1)
    assert(broadcast.get._2 arrayEquals expectedArr2Broadcast)
  }

  it should "broadcast two arrays to matching shape (3 x 2, 2)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(3, 2))
    val arr2 = NDArray[Int](List(0, 1))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isSuccess)
    val expectedArr2Broadcast =
      NDArray[Int](List(0, 1, 0, 1, 0, 1)).reshape(Array(3, 2))
    assert(broadcast.get._1 arrayEquals arr1)
    assert(broadcast.get._2 arrayEquals expectedArr2Broadcast)
  }

  it should "broadcast two arrays to matching shape (3, 2 x 3)" in {
    val arr1 = NDArray[Int](List(0, 1, 2))
    val arr2 = NDArray[Int](List(0, 1, 2, 3, 4, 5)).reshape(Array(2, 3))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isSuccess)
    val expectedArr1Broadcast =
      NDArray[Int](List(0, 1, 2, 0, 1, 2)).reshape(Array(2, 3))
    assert(broadcast.get._1 arrayEquals expectedArr1Broadcast)
    assert(broadcast.get._2 arrayEquals arr2)
  }

  it should "broadcast two arrays to matching shape (3 x 1, 1 x 3)" in {
    // Example broadcast taken from https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isSuccess)
    val expectedArr1Broadcast =
      NDArray[Int](List(1, 1, 1, 2, 2, 2, 3, 3, 3)).reshape(Array(3, 3))
    val expectedArr2Broadcast =
      NDArray[Int](List(4, 5, 6, 4, 5, 6, 4, 5, 6)).reshape(Array(3, 3))
    assert(broadcast.get._1 arrayEquals expectedArr1Broadcast)
    assert(broadcast.get._2 arrayEquals expectedArr2Broadcast)
    val broadcastSum = broadcast.get._1 + broadcast.get._2
    assert(broadcastSum.isSuccess)
    val expectedSum =
      NDArray[Int](List(5, 6, 7, 6, 7, 8, 7, 8, 9)).reshape(Array(3, 3))
    assert(broadcastSum.get arrayEquals expectedSum)
  }

  it should "broadcast two arrays to matching shape (8 x 1 x 6 x 1, 7 x 1 x 5)" in {
    val arr1 = NDArray.arange[Int](Array(8, 1, 6, 1))
    val arr2 = NDArray.arange[Int](Array(7, 1, 5))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isSuccess)
    assert(broadcast.get._1.shape sameElements Array(8, 7, 6, 5))
    assert(broadcast.get._2.shape sameElements Array(8, 7, 6, 5))
    // Computed this sum with NumPy.
    val broadcastSum = broadcast.get._1 + broadcast.get._2
    assert(broadcastSum.isSuccess)
    val sumSlice = broadcastSum.get
      .slice(List(Some(List(1)), Some(List(2)), Some(List(3)), None))
      .squeeze()
    val expectedSlice = NDArray[Int](List(19, 20, 21, 22, 23))
    assert(sumSlice arrayEquals expectedSlice)
  }

  it should "fail to broadcast two incompatible arrays (3, 4)" in {
    val arr1 = NDArray.arange[Int](Array(3))
    val arr2 = NDArray.arange[Int](Array(4))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isFailure)
  }

  it should "fail to broadcast two incompatible arrays (2 x 1, 8 x 4 x 3)" in {
    val arr1 = NDArray.arange[Int](Array(2, 1))
    val arr2 = NDArray.arange[Int](Array(8, 4, 3))
    val broadcast = arr1 broadcastWith arr2
    assert(broadcast.isFailure)
  }

  it should "define + for element-wise addition (Int)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Int](List(1, 1, 3, 2, 4))
    val addition = arr1 + arr2
    assert(addition.isSuccess)
    assert(addition.get.flatten() sameElements Array(1, 2, 5, 5, 8))
  }

  it should "define + for element-wise addition (Double)" in {
    val arr1 = NDArray[Double](List(0.0, 1.0, 2.0, 3.0, 4.0))
    val arr2 = NDArray[Double](List(1.0, 1.1, 3.0, 2.7, 4.5))
    val addition = arr1 + arr2
    assert(addition.isSuccess)
    assert(
      addition.get arrayApproximatelyEquals NDArray[Double](
        List(1.0, 2.1, 5.0, 5.7, 8.5)
      )
    )
  }

  it should "retain the same shape in element-wise addition" in {
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.arange[Int](Array(2, 3, 4))
    val addition = arr1 + arr2
    assert(addition.isSuccess)
    assert(addition.get.shape sameElements Array(2, 3, 4))
  }

  it should "broadcast arrays in element-wise addition (1, 2 x 2)" in {
    val arr1 = NDArray[Int](List(1))
    val arr2 = NDArray[Int](List(0, 1, 2, 3)).reshape(Array(2, 2))
    val addition = arr1 + arr2
    assert(addition.isSuccess)
    val expectedSum =
      NDArray[Int](List(1, 2, 3, 4)).reshape(Array(2, 2))
    assert(addition.get arrayEquals expectedSum)
  }

  it should "broadcast arrays in element-wise addition (3 x 1, 1 x 3)" in {
    // Example broadcast taken from https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val addition = arr1 + arr2
    assert(addition.isSuccess)
    val expectedSum =
      NDArray[Int](List(5, 6, 7, 6, 7, 8, 7, 8, 9)).reshape(Array(3, 3))
    assert(addition.get arrayEquals expectedSum)
  }

  it should "broadcast arrays in element-wise addition (8 x 1 x 6 x 1, 7 x 1 x 5)" in {
    val arr1 = NDArray.arange[Int](Array(8, 1, 6, 1))
    val arr2 = NDArray.arange[Int](Array(7, 1, 5))
    val addition = arr1 + arr2
    assert(addition.isSuccess)
    // Computed this sum with NumPy.
    val sumSlice = addition.get
      .slice(List(Some(List(1)), Some(List(2)), Some(List(3)), None))
      .squeeze()
    val expectedSlice = NDArray[Int](List(19, 20, 21, 22, 23))
    assert(sumSlice arrayEquals expectedSlice)
  }

  it should "fail to perform element-wise addition on arrays with different shape" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    assert((arr1 + arr2).isFailure)
  }

  it should "define - for element-wise subtraction (Int)" in {
    val arr1 = NDArray[Int](List(0, 1, 2, 3, 4))
    val arr2 = NDArray[Int](List(1, 1, 3, 2, 4))
    val subtraction = arr1 - arr2
    assert(subtraction.isSuccess)
    assert(subtraction.get.flatten() sameElements Array(-1, 0, -1, 1, 0))
  }

  it should "define - for element-wise subtraction (Double)" in {
    val arr1 = NDArray[Double](List(0.0, 1.0, 2.0, 3.0, 4.0))
    val arr2 = NDArray[Double](List(1.0, 1.1, 3.0, 2.7, 4.5))
    val subtraction = arr1 - arr2
    assert(subtraction.isSuccess)
    assert(
      subtraction.get arrayApproximatelyEquals NDArray[Double](
        List(-1.0, -0.1, -1.0, 0.3, -0.5)
      )
    )
  }

  it should "retain the same shape in element-wise subtraction" in {
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.arange[Int](Array(2, 3, 4))
    val subtraction = arr1 - arr2
    assert(subtraction.isSuccess)
    assert(subtraction.get.shape sameElements Array(2, 3, 4))
  }

  it should "broadcast arrays in element-wise subtraction (1, 2 x 2)" in {
    val arr1 = NDArray[Int](List(1))
    val arr2 = NDArray[Int](List(0, 1, 2, 3)).reshape(Array(2, 2))
    val subtraction = arr1 - arr2
    assert(subtraction.isSuccess)
    val expectedSum =
      NDArray[Int](List(1, 0, -1, -2)).reshape(Array(2, 2))
    assert(subtraction.get arrayEquals expectedSum)
  }

  it should "broadcast arrays in element-wise subtraction (3 x 1, 1 x 3)" in {
    // Example broadcast taken from https://numpy.org/doc/stable/reference/generated/numpy.broadcast.html
    val arr1 = NDArray[Int](List(1, 2, 3)).reshape(Array(3, 1))
    val arr2 = NDArray[Int](List(4, 5, 6)).reshape(Array(1, 3))
    val subtraction = arr1 - arr2
    assert(subtraction.isSuccess)
    val expectedSum =
      NDArray[Int](List(-3, -4, -5, -2, -3, -4, -1, -2, -3)).reshape(
        Array(3, 3)
      )
    assert(subtraction.get arrayEquals expectedSum)
  }

  it should "fail to perform element-wise subtraction on arrays with different shape" in {
    val arr1 = NDArray.arange[Int](Array(2, 3))
    val arr2 = NDArray.arange[Int](Array(3, 2))
    assert((arr1 - arr2).isFailure)
  }

  it should "return the sum of all elements" in {
    val arr = NDArray[Int](List(0, 1, 2, 3, 4))
    assert(arr.sum == 10)
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
    val sliced = arr.slice(List(None, None, None))
    assert(sliced arrayEquals arr)
  }

  it should "return an array with the rows requested in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(List(Some(List(0)), None, None))
    assert(sliced.shape sameElements Array(1, 3, 4))
    assert(sliced.flatten() sameElements (0 until 12))
  }

  it should "return an array with the columns requested in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(List(None, Some(List(1, 2)), None))
    assert(sliced.shape sameElements Array(2, 2, 4))
    assert(
      sliced.flatten() sameElements Array(4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18,
        19, 20, 21, 22, 23)
    )
  }

  it should "return an array with the elements requested in a slice" in {
    val arr = NDArray.arange[Int](Array(2, 3, 4))
    val sliced = arr.slice(List(None, Some(List(1, 2)), Some(List(0))))
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
    assert(matmulResult.isSuccess)
    assert(matmulResult.get arrayEquals expectedResult)
  }

  it should "fail to matrix multiply two 2D arrays with mismatching shapes" in {
    val arr1 = NDArray.ones[Int](Array(3, 2))
    val arr2 = NDArray.ones[Int](Array(3, 2))
    val matmulResult = arr1 matmul arr2
    assert(matmulResult.isFailure)
  }

  it should "fail to matrix multiply non-2D arrays" in {
    val arr1 = NDArray.ones[Int](Array(3, 2))
    val arr2 = NDArray.ones[Int](Array(2, 2, 2))
    val matmulResult = arr1 matmul arr2
    assert(matmulResult.isFailure)
  }

  it should "return the dot product of two 1D arrays" in {
    val arr1 = NDArray.arange[Int](Array(5))
    val arr2 = NDArray.ones[Int](Array(5))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isSuccess)
    assert(dotProduct.get.shape sameElements Array(1))
    assert(dotProduct.get.apply(List(0)) == 10)
  }

  it should "fail to return the dot product of two 1D arrays of different lengths" in {
    val arr1 = NDArray.arange[Int](Array(6))
    val arr2 = NDArray.ones[Int](Array(5))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isFailure)
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
    assert(matmulResult.isSuccess)
    assert(matmulResult.get arrayEquals expectedResult)
  }

  it should "return the inner products over the last axis of a multidimensional array (2D) and a 1D array using dot" in {
    // Example multiplication computed with np.dot.
    val arr1 = NDArray.arange[Int](Array(3, 4))
    val arr2 = NDArray.ones[Int](Array(4))
    val expectedResult = NDArray[Int](List(6, 22, 38))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isSuccess)
    assert(dotProduct.get arrayEquals expectedResult)
  }

  it should "return the inner products over the last axis of a multidimensional array (3D) and a 1D array using dot" in {
    // Example multiplication computed with np.dot.
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.ones[Int](Array(4))
    val expectedResult =
      NDArray[Int](List(6, 22, 38, 54, 70, 86)).reshape(Array(2, 3))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isSuccess)
    assert(dotProduct.get arrayEquals expectedResult)
  }

  it should "fail to return the dot product over the last axis of a multidimensional array and a 1D array using dot when the last axis shape does not match" in {
    val arr1 = NDArray.arange[Int](Array(4, 3))
    val arr2 = NDArray.ones[Int](Array(4))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isFailure)
  }

  it should "return the inner products of two multidimensional arrays using dot" in {
    // Example multiplication computed with np.dot.
    val arr1 = NDArray.arange[Int](Array(2, 3, 4))
    val arr2 = NDArray.arange[Int](Array(4, 2))
    val expectedResult =
      NDArray[Int](List(28, 34, 76, 98, 124, 162, 172, 226, 220, 290, 268, 354))
        .reshape(Array(2, 3, 2))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isSuccess)
    assert(dotProduct.get arrayEquals expectedResult)
  }

  it should "fail to return the inner products of two multidimensional arrays using dot when shapes don't match" in {
    val arr1 = NDArray.arange[Int](Array(2, 3, 2))
    val arr2 = NDArray.ones[Int](Array(4, 2))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isFailure)
  }

  it should "fail to return the dot product on for shapes with no defined operation" in {
    val arr1 = NDArray.arange[Int](Array(3))
    val arr2 = NDArray.ones[Int](Array(3, 2))
    val dotProduct = arr1 dot arr2
    assert(dotProduct.isFailure)
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
    assert(arr(List(0, 0, 0)) == 0)
    assert(arr(List(0, 0, 1)) == 1)
    assert(arr(List(0, 1, 0)) == 2)
    assert(arr(List(0, 1, 1)) == 3)
    assert(arr(List(0, 2, 0)) == 4)
    assert(arr(List(0, 2, 1)) == 5)
    assert(arr(List(1, 0, 0)) == 6)
    assert(arr(List(1, 0, 1)) == 7)
    assert(arr(List(1, 1, 0)) == 8)
    assert(arr(List(1, 1, 1)) == 9)
    assert(arr(List(1, 2, 0)) == 10)
    assert(arr(List(1, 2, 1)) == 11)
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
    val head = arr(List(0, 0))
    assert(!arr.flatten().forall(_ == head))
  }

  "An NDArray.random[Int] array" should "contain different elements in [-2 ^ 31, 2 ^ 31 - 1]" in {
    val arr = NDArray.random[Int](Array(2, 3))
    assert(
      arr
        .flatten()
        .forall(element => Int.MinValue <= element && element <= Int.MaxValue)
    )
    val head = arr(List(0, 0))
    assert(!arr.flatten().forall(_ == head))
  }
}
