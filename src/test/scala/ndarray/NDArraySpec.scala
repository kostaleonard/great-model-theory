package ndarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NDArraySpec extends AnyFlatSpec with Matchers {
  "An N-dimensional array" should "have the correct number of elements (rank 2)" in {
    val arr = NDArray.zeros[Int](List(2, 3))
    assert(arr.flatten().length == 6)
  }

  it should "have the correct number of elements (rank 5)" in {
    val arr = NDArray.zeros[Int](List(2, 3, 4, 5, 6))
    assert(arr.flatten().length == 2 * 3 * 4 * 5 * 6)
  }

  it should "return the elements as a 1D array when flattened" in {
    val arr = NDArray.arange[Int](List(2, 3))
    assert(arr.flatten() sameElements Array(0, 1, 2, 3, 4, 5))
  }

  it should "return the element at the given indices" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    assert(arr1(List(0, 0)) == 0)
    assert(arr1(List(0, 1)) == 1)
    assert(arr1(List(1, 0)) == 3)
    assert(arr1(List(1, 2)) == 5)
    val arr2 = NDArray.arange[Int](List(2, 3, 4))
    assert(arr2(List(0, 2, 1)) == 9)
    assert(arr2(List(1, 1, 0)) == 16)
  }

  it should "return an NDArray with the same elements but new shape when reshaped" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    val arr2 = arr1.reshape(List(3, 2))
    assert(arr2.shape sameElements Array(3, 2))
    assert(arr1.flatten() sameElements arr2.flatten())
  }

  it should "be able to contain arbitrary data types" in {
    val arr1 = NDArray.ofValue[String](List(2, 3), "hello")
    assert(arr1.flatten().forall(_.equals("hello")))
    case class Hello(x: Int)
    val arr2 = NDArray[Hello](List(Hello(0), Hello(1)))
    assert(arr2.flatten().forall(_.isInstanceOf[Hello]))
  }

  it should "be equal in comparison with an array of the same shape and elements" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    val arr2 = NDArray.arange[Int](List(2, 3))
    assert(arr1 arrayEquals arr2)
  }

  it should "not be equal in comparison with an array of different elements" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    val arr2 = NDArray.zeros[Int](List(2, 3))
    assert(arr1 arrayNotEquals arr2)
  }

  it should "not be equal in comparison with an array of different shape" in {
    val arr1 = NDArray.zeros[Int](List(2, 3))
    val arr2 = NDArray.zeros[Int](List(3, 2))
    assert(arr1 arrayNotEquals arr2)
  }

  it should "produce a mask of all true on == comparison with an array of the same shape and elements" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    val arr2 = NDArray.arange[Int](List(2, 3))
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

  it should "fail on == comparison with an array of different shape" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    val arr2 = NDArray.arange[Int](List(3, 2))
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
    val arr1 = NDArray.zeros[Double](List(2, 3))
    val arr2 = NDArray.zeros[Double](List(3, 2))
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

  it should "fail on ~= comparison with an array of different shape" in {
    val arr1 = NDArray.arange[Double](List(2, 3))
    val arr2 = NDArray.arange[Double](List(3, 2))
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

  "An NDArray.empty array" should "have no elements" in {
    val arr = NDArray.empty[Int]
    assert(arr.flatten().isEmpty)
  }

  "An NDArray.ofValue array" should "contain only the given value" in {
    val arr = NDArray.ofValue[Int](List(2, 3), 5)
    assert(arr.flatten().forall(_ == 5))
  }

  "An NDArray.zeros array" should "contain only zeros" in {
    val arr = NDArray.zeros[Int](List(2, 3))
    assert(arr.flatten().forall(_ == 0))
  }

  it should "pass type parameter information correctly" in {
    val arr = NDArray.zeros[Float](List(2, 3))
    assert(arr.flatten().forall(_.isInstanceOf[Float]))
  }

  "An NDArray.ones array" should "contain only ones" in {
    val arr = NDArray.ones[Int](List(2, 3))
    assert(arr.flatten().forall(_ == 1))
  }

  it should "pass type parameter information correctly" in {
    val arr = NDArray.ones[Float](List(2, 3))
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
    val arr = NDArray.arange[Int](List(2, 3, 2))
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
    val arr = NDArray.arange[Int](List(2, 3, 2))
    val elements = arr.flatten()
    assert(elements.indices.forall(idx => elements(idx) == idx))
  }

  it should "pass type parameter information correctly" in {
    val arr = NDArray.arange[Float](List(2, 3, 2))
    assert(arr.flatten().forall(_.isInstanceOf[Float]))
  }

  "An NDArray.random[Float] array" should "contain different elements in [0, 1)" in {
    val arr = NDArray.random[Float](List(2, 3))
    assert(arr.flatten().forall(element => 0 <= element && element < 1))
    val head = arr(List(0, 0))
    assert(!arr.flatten().forall(_ == head))
  }

  "An NDArray.random[Int] array" should "contain different elements in [-2 ^ 31, 2 ^ 31 - 1]" in {
    val arr = NDArray.random[Int](List(2, 3))
    assert(
      arr
        .flatten()
        .forall(element => Int.MinValue <= element && element <= Int.MaxValue)
    )
    val head = arr(List(0, 0))
    assert(!arr.flatten().forall(_ == head))
  }
}
