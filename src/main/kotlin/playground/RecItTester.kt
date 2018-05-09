package playground

import DATA_ROOT
import org.datavec.api.io.labels.ParentPathLabelGenerator
import org.datavec.api.split.CollectionInputSplit
import org.datavec.api.writable.IntWritable
import org.datavec.api.writable.Writable
import org.datavec.image.recordreader.ImageRecordReader
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import java.io.File

/**
 * @author Holger Brandl
 */

fun buildDataSetIterator(path: File, batchSize: Int = 64): RecordReaderDataSetIterator {

    val pathLabelGen = object : ParentPathLabelGenerator() {

        override fun getLabelForPath(path: String?): Writable {
            val label = if (File(path).nameWithoutExtension.endsWith("1")) 1 else 0
            return IntWritable(label)
        }
    }

    val recordReader = object : ImageRecordReader(224, 224, 3, pathLabelGen) {
        init {
            val jpgFiles = path.listFiles({ file -> file.extension == "jpg" })
            initialize(CollectionInputSplit(jpgFiles.map { it.toURI() }))
        }

    }

    val next = recordReader.next()
    return RecordReaderDataSetIterator(recordReader, batchSize, 1, 2)
}


fun main(args: Array<String>) {

    //    val dsIterator = createRecReaderDataIterator(File(DATA_ROOT, "train_photos"), batchSize = 10)
    val dsIterator = buildDataSetIterator(File(DATA_ROOT, "train_photos"), batchSize = 10)

    for (dataSet in dsIterator) {
        val featureMatrix = dataSet.featureMatrix
        val labels = dataSet.labels
        print("features: ${featureMatrix.shapeInfoToString()}")
        print("labels ${labels.shapeInfoToString()}")
    }
}
