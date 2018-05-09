import archive.loadImagesVGG
import org.bytedeco.javacv.Java2DFrameUtils
import org.datavec.api.io.labels.PathMultiLabelGenerator
import org.datavec.api.split.CollectionInputSplit
import org.datavec.api.writable.IntWritable
import org.datavec.image.data.ImageWritable
import org.datavec.image.recordreader.ImageRecordReader
import org.datavec.image.transform.BaseImageTransform
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.imgscalr.Scalr
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.slf4j.LoggerFactory
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.*
import java.util.logging.Level
import java.util.logging.LogManager
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

    LoggerFactory.getLogger("org.jline.utils.Log").debug("diabling jline logger")
    LogManager.getLogManager().getLogger("org.jline.utils.Log").level = Level.OFF

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

open class File2LabelConverter() {

    val businessLabels by lazy {
        File(DATA_ROOT, "train.csv").readLines().drop(1).map {
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
    }

    protected val photo2business by lazy { readPhoto2BusinessModel(File(DATA_ROOT, "train_photo_to_biz_ids.csv")) }

    open fun getLabels(it: String): List<Int> {
        val businessID = photo2business[it]!!

        var imageLabels = businessLabels[businessID]!!

        // simplify problem to use a single category only
        //        if(NUM_CLASSES==2) {
        //            imageLabels = listOf(1) - imageLabels
        //        }

        // convert to indicator vector
        //        return IntArray(allLabels.size, { if (imageLabels.contains(it)) 1 else 0 }).toList()
        return imageLabels
    }

    fun getLabels(imageFile: File): List<Int> = getLabels(imageFile.nameWithoutExtension)

    open val allLabels by lazy { businessLabels.values.flatten().distinct().map { it.toString() } }
}


class TwoClassLabelConverter(val classIndex: Int) : File2LabelConverter() {
    override fun getLabels(it: String): List<Int> {
        val businessID = photo2business[it]!!

        var imageLabels = businessLabels[businessID]!!

        return if (imageLabels.contains(classIndex)) listOf(1) else listOf(0)
    }

    override val allLabels: List<String>
        get() = listOf("0", "1")
}


fun createRecReaderDataIterator(
    path: File,
    batchSize: Int = 256,
    maxExamples: Int = Int.MAX_VALUE,
    labelConverter: File2LabelConverter = File2LabelConverter()
): RecordReaderDataSetIterator {

    // todo simplify to just use PathLabelGenerator in case of 2 classes
    //    val labelConverter = TwoClassLabelConverter(3)


    val trafo = object : BaseImageTransform<Double>() {
        override fun doTransform(image: ImageWritable?, random: Random?): ImageWritable {
            val bufferedImage = Java2DFrameUtils.toBufferedImage(image!!.frame)
            //            val bufferedImage = Java2DFrameConverter.cloneBufferedImage(image.frame)

            val transformed = bufferedImage.removeAlphaChannel()

            return ImageWritable(Java2DFrameUtils.toFrame(transformed))
            //            return image
        }
    }

    val pathLabelGen = PathMultiLabelGenerator { uriPath ->
        //            return listOf(LongWritable(1))
        labelConverter.getLabels(File(uriPath)).map { IntWritable(it) }
    }


    val recordReader = object : ImageRecordReader(224, 224, 3, pathLabelGen) {
        init {
            imageTransform = trafo

            val jpgFiles = path.listFiles({ file -> file.extension == "jpg" }).take(maxExamples)
            //            val splitTrainNum = ceil(jpgFiles.size * 0.8).toInt() // 80/20 training/test split

            initialize(CollectionInputSplit(jpgFiles.map { it.toURI() }))
        }
    }


    return RecordReaderDataSetIterator(recordReader, batchSize, 1, 2)
}


fun main(args: Array<String>) {

    //    val dsIterator = createRecReaderDataIterator(File(DATA_ROOT, "train_photos"), batchSize = 10)
    val dsIterator = playground.buildDataSetIterator(File(DATA_ROOT, "train_photos"), batchSize = 10)

    for (dataSet in dsIterator) {
        val featureMatrix = dataSet.featureMatrix
        val labels = dataSet.labels
        print("features: ${featureMatrix.shapeInfoToString()}")
        print("labels ${labels.shapeInfoToString()}")
    }
}


fun prepareTestData(path: File, numExamples: Int = Int.MAX_VALUE, batchSize: Int = 200): Sequence<Pair<List<File>, INDArray>> {
    val testImages = path.listFiles({ file -> file.extension == "jpg" }).toList()
        .take(numExamples)
        .chunked(batchSize)
        .iterator()

    return buildSequence {

        while (testImages.hasNext()) {
            val (imageFiles, imageData) = testImages.next().loadImagesVGG()

            val ndFeatures = Nd4j.vstack(imageData.map { it })

            yield(imageFiles to ndFeatures)
        }
    }
}
