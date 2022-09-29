package ndarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NDArraySpec extends AnyFlatSpec with Matchers {
  "An NDArray.zeros array" should "contain only zeros" in {
    val arr = NDArray.zeros(List(2, 3))
    assert(arr.flatten().forall(_ == 0))
  }
}
