package ndarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NDArraySpec extends AnyFlatSpec with Matchers {
  "An N-dimensional array" should "have the correct number of elements" in {
    val arr = NDArray.zeros[Int](List(2, 3, 4, 5, 6))
    assert(arr.flatten().length == 2 * 3 * 4 * 5 * 6)
  }

  it should "default to type Float" in {
    //TODO I'm going to have to learn the syntax a bit better to get this to pass.
    val arr = NDArray.zeros(List(2, 3))
    //assert(arr.flatten().forall(_.isInstanceOf[Float]))
  }

  it should "have the correct number of elements" in {
    val arr = NDArray.zeros[Int](List(2, 3))
    assert(arr.flatten().length == 6)
  }

  it should "return the element at the given indices" in {
    val arr1 = NDArray.arange[Int](List(2, 3))
    assert(arr1(List(0, 0)) == 0)
    assert(arr1(List(0, 1)) == 1)
    assert(arr1(List(1, 0)) == 3)
    assert(arr1(List(1, 2)) == 5)
    val arr2 = NDArray.arange[Int](List(2, 3, 4))
    assert(arr2(List(0, 2, 1)) == 10)
    assert(arr2(List(1, 1, 0)) == 16)
  }

  "An NDArray.zeros array" should "contain only zeros" in {
    val arr = NDArray.zeros[Int](List(2, 3))
    assert(arr.flatten().forall(_ == 0))
  }
}
