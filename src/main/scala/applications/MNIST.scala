package applications

import ndarray.NDArray

import java.io.{BufferedInputStream, ByteArrayInputStream, File, FileInputStream}
import java.net.URL
import java.util.zip.GZIPInputStream
import scala.io.Source
import sys.process._
import scala.language.postfixOps

/** Gets the MNIST dataset. */
case object MNIST {
  private val trainImagesUrl =
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  private val trainLabelsUrl =
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  private val urlEncoding = "ISO-8859-1"

  /** Returns the MNIST dataset as a pair of arrays (features, labels).
    *
    * The MNIST dataset contains 60,000 grayscale images and labels. Each image
    * is 28 x 28 and contains pixel values from 0 to 255. Each label is an
    * integer from 0 to 9 indicating the number shown in the image.
    */
  def getDataset: (NDArray[Int], NDArray[Int]) = {
    val trainImagesGzipContent = Source.fromURL(trainImagesUrl)(urlEncoding)
    val bytes = trainImagesGzipContent.mkString.getBytes(urlEncoding)
    //TODO remove debugging
    println(bytes.take(20).map(b => String.format("%02x", Byte.box(b))).mkString("Array(", ", ", ")"))
    val trainImagesContent = Source.fromInputStream(new GZIPInputStream(new ByteArrayInputStream(bytes)))(urlEncoding)
    //The magic number is 0x00000803(2051)
    println(trainImagesContent.mkString.getBytes(urlEncoding).take(20).map(b => String.format("%02x", b)).mkString("Array(", ", ", ")"))
    ???
  }
}
