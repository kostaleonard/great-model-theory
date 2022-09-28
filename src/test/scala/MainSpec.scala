import main.Main
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

//TODO this test is outdated
class MainSpec extends AnyFlatSpec with Matchers {
  "The Hello object" should "say hello" in {
    Main.main shouldNot "hello"
  }
}