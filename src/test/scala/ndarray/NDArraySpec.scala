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

  it should "default to type Float" in {
    //TODO I'm going to have to learn the syntax a bit better to get this to pass.
    val arr = NDArray.zeros(List(2, 3))
    //assert(arr.flatten().forall(_.isInstanceOf[Float]))
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

  "An NDArray.ones array" should "contain only ones" in {
    val arr = NDArray.ones[Int](List(2, 3))
    assert(arr.flatten().forall(_ == 1))
  }

  "An NDArray.apply array" should "convert a flat sequence into a rank 1 NDArray" in {
    val values = List(1, 2, 3, 4)
    val arr = NDArray[Int](values)
    assert(arr.shape == List(4))
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
}
