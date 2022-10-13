package losses

import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MeanSquaredErrorSpec extends AnyFlatSpec with Matchers {

  "A MeanSquaredError loss function" should "compute the mean squared error on the input" in {
    val y_true = NDArray[Float](List(2.0f)).reshape(List(1, 1))
    val y_pred = NDArray[Float](List(0.0f)).reshape(List(1, 1))
    val mse = new MeanSquaredError[Float]
    val loss = mse.compute_loss(y_true, y_pred)
    assert(loss.shape sameElements Array(1))
    assert(loss arrayApproximatelyEquals NDArray[Float](List(4.0f)))
  }

  it should "apply the loss on each element in the batch" in {
    val y_true = NDArray[Float](List(2.0f, 1.0f, -3.0f)).reshape(List(3, 1))
    val y_pred = NDArray[Float](List(0.0f, 1.0f, 1.0f)).reshape(List(3, 1))
    val mse = new MeanSquaredError[Float]
    val loss = mse.compute_loss(y_true, y_pred)
    assert(loss.shape sameElements Array(3))
    assert(
      loss arrayApproximatelyEquals NDArray[Float](List(4.0f, 0.0f, 16.0f))
    )
  }

  it should "reduce by the mean on the last dimension (2D)" in {
    val y_true =
      NDArray[Float](List(2.0f, -2.0f, 0.0f, 1.0f)).reshape(List(2, 2))
    val y_pred =
      NDArray[Float](List(0.0f, 1.0f, 0.0f, 2.0f)).reshape(List(2, 2))
    val mse = new MeanSquaredError[Float]
    val loss = mse.compute_loss(y_true, y_pred)
    assert(loss.shape sameElements Array(2))
    assert(loss arrayApproximatelyEquals NDArray[Float](List(6.5f, 0.5f)))
  }

  it should "reduce by the mean on the last dimension (3D)" in {
    val y_true = NDArray[Float](
      List(2.0f, -2.0f, 0.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
        1.0f)
    ).reshape(List(2, 3, 2))
    val y_pred = NDArray[Float](
      List(0.0f, 1.0f, 0.0f, 2.0f, 3.0f, -1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 1.0f,
        0.0f)
    ).reshape(List(2, 3, 2))
    val mse = new MeanSquaredError[Float]
    val loss = mse.compute_loss(y_true, y_pred)
    assert(loss.shape sameElements Array(2, 3))
    assert(
      loss arrayApproximatelyEquals NDArray[Float](
        List(2.5f, 6.5f, 0.0f, 0.5f, 0.0f, 1.0f)
      ).reshape(List(2, 3))
    )
  }
}
