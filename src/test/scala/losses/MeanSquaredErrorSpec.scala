package losses

import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MeanSquaredErrorSpec extends AnyFlatSpec with Matchers {

  "A MeanSquaredError loss function" should "compute the mean squared error on the input" in {
    val y_true = NDArray[Float](Array(2.0f)).reshape(List(1, 1))
    val y_pred = NDArray[Float](Array(0.0f)).reshape(List(1, 1))
    val mse = new MeanSquaredError[Float]
    val loss = mse.compute_loss(y_true, y_pred)
    assert(loss.shape sameElements Array(1))
    assert(loss arrayApproximatelyEquals NDArray[Float](Array(4.0f)))
  }

  it should "reduce by the mean on the last dimension (2D)" in {
    val y_true = NDArray[Float](Array(2.0f, -2.0f)).reshape(List(1, 2))
    val y_pred = NDArray[Float](Array(0.0f, 1.0f)).reshape(List(1, 2))
    val mse = new MeanSquaredError[Float]
    val loss = mse.compute_loss(y_true, y_pred)
    assert(loss.shape sameElements Array(1))
    assert(loss arrayApproximatelyEquals NDArray[Float](Array(6.5f)))
  }

  it should "reduce by the mean on the last dimension (3D)" in {
    //TODO
  }

  it should "apply the loss on each element in the batch" in {
    //TODO
  }
}
