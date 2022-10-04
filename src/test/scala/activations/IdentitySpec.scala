package activations

import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class IdentitySpec extends AnyFlatSpec with Matchers {

  "An Identity activation" should "make no change to the inputs" in {
    val identity = Identity[Float]()
    val arr = NDArray.arange[Float](List(2, 3))
    assert(arr.flatten() sameElements identity.activation(arr).flatten())
  }
}
