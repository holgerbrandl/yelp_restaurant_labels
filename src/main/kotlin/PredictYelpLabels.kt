import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import java.io.File
import java.io.FileFilter
import kotlin.coroutines.experimental.buildSequence

/**
 * @author Holger Brandl
 */

fun main(args: Array<String>) {
    configureLogger()

    // read and pre-process the images in one big bulk
    //    val allTrainDS = prepareTrainDataVG(File(DATA_ROOT, "train_photos")).ds
    //    // bullk input data
    //    println("train shape  is " + allTrainDS.featureMatrix.shape().joinToString())
    //    println("label shape  is " + allTrainDS.labels.shape().joinToString())
    //
    //    val splitTrainNum = ceil(allTrainDS.numExamples() * 0.8).toInt() // 80/20 training/test split
    //    val dataSplit: SplitTestAndTrain = allTrainDS.splitTestAndTrain(splitTrainNum, Random(42))
    //
    //    val batchSize = 128
    //    val trainData = ListDataSetIterator(dataSplit.train.asList(), batchSize)
    //    val testData = ListDataSetIterator(dataSplit.test.asList(), batchSize)


    // https://nd4j.org/userguide#serialization
    // https://stackoverflow.com/questions/4841340/what-is-the-use-of-bytebuffer-in-java
    //    allTrainDS.save(File("yelp_train.dat"))
    //    DataSet().load(File("yelp_train.dat"))


    // no need to do this manually, since its done by splitTestAndTrain
    //    val someSeed=42.toLong()
    //    Nd4j.shuffle(allTrainDS.featureMatrix,  Random(someSeed), 1) // this changes ds.  Shuffles rows
    //    Nd4j.shuffle(allTrainDS.labels,  Random(someSeed), 1) // this changes ds.  Shuffles labels accordingly


    // rather use train iterator
    // train total=234842; test total=237152
    val trainData = createDataIterator(File(DATA_ROOT, "train_photos"), maxExamples = 50000)
    val validationData = createDataIterator(File(DATA_ROOT, "train_photos"), isTrain = false, maxExamples = 5000)


    println("Building model....")


    //        val (model, modelName)  = buildVggTransferModel(allTrainDS)!! to "vgg_transfer"

    //    val (model, modelName) = customConfModel(trainData, validationData) to "custom_cnn";
    //    model.save(File("dense_model.${modelName}.${now}.dat"))
    //    val model =     MultiLayerNetwork.load(File("dense_model.modelName.2018-05-02T09_41_20.898.dat"),false)
    val model = MultiLayerNetwork.load(mostRecent("dense_model.custom_cnn"), false)


    println("Evaluate model....")


    val testDataIt = prepareTestData(File(DATA_ROOT, "test_photos"), numExamples = 5000)

    testDataIt.forEach { evalData ->
        // subset INDArray directly
        //evalData.second.get(interval(0, 10))

        println("shape of eval data is ${evalData.second.shapeInfoToString()}")

        val modelOut = model.output(evalData.second)

        //    // peak into actual predictions
        //    for(i in 1 ..10){
        //        modelOut.getRow(i).toDoubleVector().joinToString().let(::println)
        //    }

        // recast output depending on model type (multilayer vs computation graph
        val testOuput = when (modelOut) {
            is INDArray -> modelOut
            is Array<*> -> modelOut.first() as INDArray
            else -> TODO()
        }

        val submissionLabels = buildSequence {
            for (i in 0 until testOuput.rows())
                yield(testOuput.getRow(i).toDoubleVector().map { it.toInt() }.filter { it != 1 })
        }.toList()

        // todo should be more fun with krangl

        // https://www.kaggle.com/c/yelp-restaurant-photo-classification#evaluation

        val photo2business = readPhoto2BusinessModel(File(DATA_ROOT, "test_photo_to_biz.csv"))


        require(evalData.first.size == submissionLabels.size)

        val submissionData = evalData.first
            .zip(submissionLabels)
            .map { photo2business[it.first.nameWithoutExtension]!! to it.second }
            .groupBy { it.first }
            .mapValues { (_, labels) -> labels.flatMap { it.second }.distinct() }

        File("kaggle_submission.${now}.txt").printWriter().use { pw ->
            // the submission format should be business_id to labels
            pw.write("business_id\tlabels\n")

            submissionData.forEach {
                pw.write("${it.key}, ${it.value.joinToString { " " }}\n")
            }
        }
    }


}


fun mostRecent(prefix: String): File? = File(".")
    .listFiles(FileFilter { it.name.startsWith(prefix) })
    .sortedBy { it.lastModified() }
    .last()
    .also { println("most recent model is ${it}") }

data class ImageInfo(val rows: Int, val columns: Int, val channels: Int)
