import me.tongfei.progressbar.ProgressBar
import org.datavec.image.loader.Java2DNativeImageLoader
import org.imgscalr.Scalr
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.logging.Level
import java.util.logging.LogManager
import javax.imageio.ImageIO
import kotlin.coroutines.experimental.buildSequence

/**
 * @author Holger Brandl
 */
fun readPhoto2BusinessModel(mappingFile: File): Map<String, String> {
    return mappingFile
        .readLines()
        .drop(1)
        .map { it.split(",").let { it[0] to it[1] } }.toMap()
}

internal fun configureLogger() {
    // http://saltnlight5.blogspot.de/2013/08/how-to-configure-slf4j-with-different.html

    //    https://stackoverflow.com/questions/4311026/how-to-get-slf4j-hello-world-working-with-log4j
    //    The log levels are ERROR > WARN > INFO > DEBUG > TRACE.
    //    System.setProperty(org.slf4j.impl.SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "TRACE");
    //    val root = LoggerFactory.getLogger(ROOT_LOGGER_NAME) as SimpleLogger

    // https://stackoverflow.com/questions/194765/how-do-i-get-java-logging-output-to-appear-on-a-single-line
    System.setProperty(
        "java.util.logging.SimpleFormatter.format",
        """%1${'$'}tY-%1${'$'}tm-%1${'$'}td %1${'$'}tH:%1${'$'}tM:%1${'$'}tS %4${'$'}s %2${'$'}s %5${'$'}s%6${'$'}s%n"""
    )

    // https://nozaki.me/roller/kyle/entry/java-util-logging-programmatic-configuration
    //    java.util.logging.LogManager.getLogManager().getLogger(ZooModel::class.java.simpleName).level = Level.FINEST
    LogManager.getLogManager().getLogger("").level = Level.INFO

    //    LoggerFactory.getLogger("foo").info("hello info")
    //    LoggerFactory.getLogger("foo").warn("hello warn")
    //    LoggerFactory.getLogger("foo").debug("hello debug")
}

internal val now get() = LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME).replace(":", "_")


//
// Image Processing
//


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

/**
 * @author Holger Brandl
 */


fun loadImagesVGG(path: File): Pair<List<File>, List<INDArray>> {
    val images = path.listFiles({ file -> file.extension == "jpg" })
        .toList()
    //        .take(50)


    val imageData = ProgressBar.wrap(images, "Preparing Images").map {
        ImageIO.read(it)
            .makeSquare()
            // todo it seems that vgg is using 224x224 as input dim
            .resize(224, 224)
            .removeAlphaChannel()
            .let {
                Java2DNativeImageLoader(it.height, it.width, 3).asMatrix(it)
            }
    }

    return Pair(images, imageData)
}

fun prepareTrainDataVG(path: File): ImageDataSet {
    val (images, imageData) = loadImagesVGG(path)

    val businessLabels = File(DATA_ROOT, "train.csv").readLines().drop(1).map {
        val splitLine = it.split(",")
        val businessId = splitLine[0]
        // 0: good_for_lunch
        // 1: good_for_dinner
        // 2: takes_reservations
        // 3: outdoor_seating
        // 4: restaurant_is_expensive
        // 5: has_alcohol
        // 6: has_table_service
        // 7: ambience_is_classy
        // 8: good_for_kids
        val labels = splitLine[1].trim().split(" ").filterNot { it.isBlank() }.map { it.toInt() }
        businessId to labels

    }.toMap()


    val photo2business = readPhoto2BusinessModel(File(DATA_ROOT, "train_photo_to_biz_ids.csv"))


    val labels = images.map { it.nameWithoutExtension }.map {
        val businessID = photo2business[it]!!

        var imageLabels = businessLabels[businessID]!!

        // simplify problem to use a single category only
        imageLabels = listOf(1) - imageLabels

        // convert to indicator vector
        IntArray(NUM_CLASSES, { if (imageLabels.contains(it)) 1 else 0 }).toList()
    }


    // https://nd4j.org/userguide#creating
    // val data =listOf(Nd4j.create(doubleArrayOf(12.0,3.0)), Nd4j.create(doubleArrayOf(12.0,3.0)))

    val ndFeatures = Nd4j.vstack(imageData.map { it })
    val ndLabels = Nd4j.vstack(labels.map { Nd4j.create(it) })

    return ImageDataSet(images, DataSet(ndFeatures, ndLabels))
}