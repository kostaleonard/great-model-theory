package applications

import ndarray.NDArray

import scala.io.Source

/** Gets the MNIST dataset. */
case object MNIST {
  val trainImagesUrl =
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  val trainLabelsUrl =
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

  /** Returns the MNIST dataset as a pair of arrays (features, labels).
    *
    * The MNIST dataset contains 60,000 grayscale images and labels. Each image
    * is 28 x 28 and contains pixel values from 0 to 255. Each label is an
    * integer from 0 to 9 indicating the number shown in the image.
    */
  def getDataset: (NDArray[Int], NDArray[Int]) = {
    val trainImagesContent = Source.fromURL(trainImagesUrl)
    println(trainImagesContent.iter.next())
    ???
  }
}
