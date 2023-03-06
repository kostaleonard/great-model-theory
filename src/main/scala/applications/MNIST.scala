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

  //TODO probably want train and test datasets
  /** Returns the MNIST dataset as a pair of arrays (features, labels).
    *
    * The MNIST dataset contains 60,000 grayscale images and labels. Each image
    * is 28 x 28 and contains pixel values from 0 to 255. Each label is an
    * integer from 0 to 9 indicating the number shown in the image.
    */
  def getDataset: (NDArray[Int], NDArray[Int]) = {
    val trainImagesBytes = getDecompressedBytesFromUrl(trainImagesUrl)
    val trainImages = parseImageFileBytes(trainImagesBytes)

    ???
  }

  private def getDecompressedBytesFromUrl(url: String): Array[Byte] = {
    val fileGzipContent = Source.fromURL(url)(urlEncoding)
    val fileGzipBytes = fileGzipContent.mkString.getBytes(urlEncoding)
    val fileDecompressedContent = Source.fromInputStream(new GZIPInputStream(new ByteArrayInputStream(fileGzipBytes)))(urlEncoding)
    val fileDecompressedBytes = fileDecompressedContent.mkString.getBytes(urlEncoding)
    fileDecompressedBytes
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
    var pixelsReversed = List.empty[Int]
    val numPixelValues = numImages * numRows * numCols
    (0 until numPixelValues).foreach{ pixelIdx =>
      // This bitwise AND causes pixelValue to be an unsigned int in [0, 255].
      val pixelValue = bytes(pixelIdx + 16) & 0xff
      pixelsReversed +:= pixelValue
    }
    val images = NDArray(pixelsReversed.reverse).reshape(Array(numImages, numRows, numCols))
    images
  }
}
