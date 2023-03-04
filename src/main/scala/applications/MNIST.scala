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
    val bytes = trainImagesGzipContent.map(_.toByte).toArray
    //val bytes = trainImagesGzipContent.mkString.getBytes
    //println(bytes.take(20).map(b => String.format("%02x", Byte.box(b))).mkString("Array(", ", ", ")"))
    //println(bytes.length)
    //TODO want to replace FileInputStream with something that just reads the gzip contents string as bytes
    val trainImagesContent = Source.fromInputStream(new GZIPInputStream(new ByteArrayInputStream(bytes)))(urlEncoding)
    //println(trainImagesGzipContent.getLines().next())
    //println(trainImagesContent.mkString)

    //TODO this uses a subprocess, which I don't like.
    //new URL(trainImagesUrl) #> new File("/Users/leo/Downloads/train_images.gz") !!
    //val trainImagesContent = Source.fromInputStream(new GZIPInputStream(new BufferedInputStream(new FileInputStream("/Users/leo/Downloads/train_images.gz"))))(urlEncoding)
    //val trainImagesContent = Source.fromInputStream(new BufferedInputStream(new FileInputStream("/Users/leo/Downloads/train_images.gz")))(urlEncoding)
    //The magic number is 0x00000803(2051)
    println(trainImagesContent.mkString.getBytes.take(20).map(b => String.format("%02x", Byte.box(b))).mkString("Array(", ", ", ")"))
    ???
  }
}
