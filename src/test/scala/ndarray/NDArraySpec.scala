package ndarray

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NDArraySpec extends AnyFlatSpec with Matchers {
  "An NDArray.zeros array" should "contain only zeros" in {
    //TODO we want NDArray to default to float32 if we don't specify the type
    val arr = NDArray.zeros[Int](List(2, 3))
    assert(arr.flatten().forall(_ == 0))
  }
}
