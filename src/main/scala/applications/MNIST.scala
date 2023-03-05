package applications

import ndarray.NDArray

import java.io.{ByteArrayInputStream, IOException}
import java.util.zip.GZIPInputStream
import scala.io.Source
import scala.language.postfixOps

/** Gets the MNIST dataset. */
case object MNIST {
  private val trainImagesUrl =
    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
  private val trainLabelsUrl =
    "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
  private val urlEncoding = "ISO-8859-1"
  private val imageFileMagicNumber = Array(0, 0, 8, 3)

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
    val trainImagesBytes = trainImagesContent.mkString.getBytes(urlEncoding)
    val trainImages = parseImageFileBytes(trainImagesBytes)
    println(trainImagesContent.mkString.getBytes(urlEncoding).take(20).map(b => String.format("%02x", b)).mkString("Array(", ", ", ")"))
    ???
  }

  /** Returns the image array from the decompressed IDX-formatted file.
    *
    * The following site contains both the download links for the MNIST dataset
    * and the parsing instructions for IDX files:
    * http://yann.lecun.com/exdb/mnist/
    *
    * @param bytes
    *   The binary contents of the decompressed IDX-formatted file.
    */
  private def parseImageFileBytes(bytes: Array[Byte]): NDArray[Int] = {
    val magicNumber = bytes.take(4)
    if(!(magicNumber sameElements imageFileMagicNumber)) throw new IOException(s"Images file has incorrect magic number: ${magicNumber.mkString("Array(", ", ", ")")}")
    val numImages = BigInt(bytes.slice(4, 8)).toInt
    val numRows = BigInt(bytes.slice(8, 12)).toInt
    val numCols = BigInt(bytes.slice(12, 16)).toInt
    var images = NDArray.zeros[Int](Array(numImages, numRows, numCols))
    //TODO producing the indices takes way too long and we don't need it.
    //val imageIndices = images.indices
    val numPixelValues = numImages * numRows * numCols
    (0 until numPixelValues).foreach{ pixelIdx =>
      // This bitwise AND causes pixelValue to be an unsigned int in [0, 255].
      val pixelValue = bytes(pixelIdx + 16) & 0xff
      //val imageIdx = imageIndices(pixelIdx)

    }
    ???
  }
}
