import org.imgscalr.Scalr
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import kotlin.coroutines.experimental.buildSequence

fun File.prepareImage(): DoubleArray = ImageIO.read(this)
    .makeSquare()
    .resize() // (200, 200)
    .imageAsGreyVec()

fun BufferedImage.makeSquare(): BufferedImage {
    val w = width
    val h = height
    val dim = listOf(w, h).min()!!

    return when {
        w == h -> this
        w > h -> Scalr.crop(this, (w - h) / 2, 0, dim, dim)
        w < h -> Scalr.crop(this, 0, (h - w) / 2, dim, dim)
        else -> TODO()
    }
}

fun BufferedImage.resize(width: Int = 64, height: Int = 64) =
    Scalr.resize(this, Scalr.Method.BALANCED, width, height)

fun BufferedImage.imageAsGreyVec(): DoubleArray =
    buildSequence {
        for (w in 0 until width) {
            for (h in 0 until height) {
                // https://stackoverflow.com/questions/2615522/java-bufferedimage-getting-red-green-and-blue-individually
                val col = Color(getRGB(w, h))
                yield((col.red + col.green + col.blue).toDouble() / 3)
            }
        }
    }.toList().toDoubleArray()


/** From https://stackoverflow.com/questions/26918675/removing-transparency-in-png-bufferedimage **/
internal fun BufferedImage.removeAlphaChannel(): BufferedImage {
    val copy = BufferedImage(getWidth(), getHeight(), BufferedImage.TYPE_INT_RGB);
    val g2d = copy.createGraphics()
    g2d.setColor(Color.WHITE) // Or what ever fill color you want...
    g2d.fillRect(0, 0, copy.getWidth(), copy.getHeight())
    g2d.drawImage(this, 0, 0, null)
    g2d.dispose()

    return copy
}