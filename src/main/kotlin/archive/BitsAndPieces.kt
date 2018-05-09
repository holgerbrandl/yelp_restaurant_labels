package archive

import File2LabelConverter
import ImageDataSet
import makeSquare
import me.tongfei.progressbar.ProgressBar
import org.datavec.image.loader.Java2DNativeImageLoader
import org.deeplearning4j.datasets.iterator.BaseDatasetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher
import org.nd4j.linalg.factory.Nd4j
import removeAlphaChannel
import resize
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.ceil

/**
 * @author Holger Brandl
 */

fun createDataIterator(
    path: File,
    isTrain: Boolean = true,
    batchSize: Int = 256,
    maxExamples: Int = Int.MAX_VALUE
): BaseDatasetIterator {

    val dataFetcher = object : BaseDataFetcher() {

        val numExamples: Int

        val labelConverter by lazy { File2LabelConverter() }

        val batches: Iterator<List<File>>

        // https://kotlinlang.org/docs/reference/classes.html
        init {
            val jpgFiles = path.listFiles({ file -> file.extension == "jpg" }).take(maxExamples)
            val splitTrainNum = ceil(jpgFiles.size * 0.8).toInt() // 80/20 training/test split


            val splitPart = if (isTrain) jpgFiles.take(splitTrainNum) else jpgFiles.drop(splitTrainNum)

            numExamples = splitPart.size
            batches = splitPart.chunked(size = batchSize).iterator()

            //            val it = seq.iterator()
            //            it.asSequence().take(3).toList() // [1, 2, 3]
            //            it.asSequence().take(3).toList() // [4, 5, 6]
        }

        override fun next(): DataSet {
            // for a reference example see org.deeplearning4j.datasets.fetchers.MnistDataFetcher.fetch
            val (images, imageData) = batches.next().loadImagesVGG()


            val labelIndexVectors = images.map { it.nameWithoutExtension }.map {
                val imageLabels = labelConverter.getLabels(it)
                IntArray(labelConverter.allLabels.size, { if (imageLabels.contains(it)) 1 else 0 }).toList()
            }

            // https://nd4j.org/userguide#creating
            // val data =listOf(Nd4j.create(doubleArrayOf(12.0,3.0)), Nd4j.create(doubleArrayOf(12.0,3.0)))

            val ndFeatures = Nd4j.vstack(imageData.map { it })
            val ndLabels = Nd4j.vstack(labelIndexVectors.map { Nd4j.create(it) })

            return DataSet(ndFeatures, ndLabels)
        }

        override fun fetch(numExamples: Int) {
            // not needed since we have the data already but we could actually call kaggle api here
        }

        override fun hasMore(): Boolean {
            return batches.hasNext()
        }

        override fun totalExamples(): Int = numExamples
    }


    return BaseDatasetIterator(batchSize, -1, dataFetcher)
}


// bulk load all data in one piece
fun prepareTrainDataVG(path: File): ImageDataSet {
    val (images, imageData) = path.listFiles({ file -> file.extension == "jpg" })
        .toList().loadImagesVGG()

    val labelConverter by lazy { File2LabelConverter() }

    val labels = images.map { it.nameWithoutExtension }.map { labelConverter.getLabels(it) }

    // https://nd4j.org/userguide#creating
    // val data =listOf(Nd4j.create(doubleArrayOf(12.0,3.0)), Nd4j.create(doubleArrayOf(12.0,3.0)))

    val ndFeatures = Nd4j.vstack(imageData.map { it })
    val ndLabels = Nd4j.vstack(labels.map { Nd4j.create(it) })

    return ImageDataSet(images, DataSet(ndFeatures, ndLabels))
}


fun List<File>.loadImagesVGG(): Pair<List<File>, List<INDArray>> {
    //        .take(50)


    val imageData = ProgressBar.wrap(this, "Preparing Images").map {
        ImageIO.read(it)
            .makeSquare()
            // todo it seems that vgg is using 224x224 as input dim
            .resize(224, 224)
            .removeAlphaChannel()
            .let {
                Java2DNativeImageLoader(it.height, it.width, 3).asMatrix(it)
            }
    }

    return Pair(this, imageData)
}


