import org.datavec.api.records.metadata.RecordMetaDataURI
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.slf4j.LoggerFactory
import java.io.File
import java.io.FileFilter
import java.io.IOException
import kotlin.coroutines.experimental.buildSequence
import kotlin.math.roundToInt

/**
 * @author Holger Brandl
 */

fun main(args: Array<String>) {
    configureLogger()

    val file2Label = TwoClassLabelConverter(YelpLabel.GOOD_FOR_KIDS)

    val (trainData, validationData) = prepareTrainingData(file2Label)


    println("Building model....")


    //        val (model, modelName)  = buildVggTransferModel(allTrainDS)!! to "vgg_transfer"

    //    val (model, modelName) = customConfModel(trainData, validationData, file2Label.numClasses) to "custom_cnn";
    //        model.save(File("${modelName}.${now}.dat"))

    //    val model =     MultiLayerNetwork.load(File("dense_model.modelName.2018-05-02T09_41_20.898.dat"),false)
    //        val model =     MultiLayerNetwork.load(File("dense_model.custom_cnn.2018-05-02T16_19_27.471.dat"),false)
    val model = MultiLayerNetwork.load(mostRecent("custom_cnn"), false)


    println("Evaluating model....")
    //    println(model.summary())
    //    println(model.conf().toJson())


    val testDataIterator = createTestRecReaderDataIterator(File(DATA_ROOT, "test_photos"), maxExamples = 500)

    //        val exampleMetaData = dataSet.exampleMetaData
    for (testData in testDataIterator) {
        val testPrediction = model.output(testData.featureMatrix)

        val files = testData
            .getExampleMetaData(RecordMetaDataURI::class.java)
            .map { File(it.uri) }


        // for multi-class detection
        //        val testPrediction = model.output(testData.featureMatrix)
        val submissionLabels = buildSequence {
            with(testPrediction) {
                for (i in 0 until rows())
                    yield(getRow(i).toDoubleVector().withIndex().filter { it.value > 0.5 }.indices.toList())
            }
        }.toList()



        val photo2business = readPhoto2BusinessModel(File(DATA_ROOT, "test_photo_to_biz.csv"))


        val submissionData = files
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

@Throws(IOException::class)
fun multiClassCNN(trainData: DataSetIterator, validationData: DataSetIterator, numClasses: Int): MultiLayerNetwork {

    val firstBatch = trainData.next()
    trainData.reset()

    val nFeatures = firstBatch.featureMatrix.getRow(0).length().toDouble() // hyper, hyper parameter
    val nLabels = firstBatch.labels.getRow(0).length()

    println("number of features is ${nFeatures} and the number of labels is ${nLabels}")

    val numRows = firstBatch.featureMatrix.shape()[2]
    val numColumns = firstBatch.featureMatrix.shape()[3] // numRows * numColumns must equal columns in initial data * channels
    val nChannels = 3 // would be 3 if color image w R,G,B


    // todo later: run multiple epochs over the data
    //    val numEpochs = 15 // number of epochs to perform
    val epochitTr = MultipleEpochsIterator(4, trainData)


    val log = LoggerFactory.getLogger("conv_model_trainer")

    val outputNum = numClasses // # of classes (# of columns in output)


    val builder = NeuralNetConfiguration.Builder()
        .seed(12345.toLong())
        .updater(Nesterovs(0.01, 0.9)) //specify the rate of change of the learning rate.
        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
        .list()

        //First convolution layer with ReLU as activation function
        .layer(ConvolutionLayer.Builder(6, 6)
            .nIn(nChannels)
            .stride(2, 2) // default stride(2,2)
            .nOut(20) // # of feature maps/ filters
            .dropOut(0.5)
            .activation(Activation.RELU) // rectified linear units
            .weightInit(WeightInit.XAVIER)
            .build())

        //First subsampling layer
        .layer(SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())

        //Second convolution layer with ReLU as activation function
        .layer(ConvolutionLayer.Builder(6, 6)
            .stride(2, 2)
            .nOut(50)
            .activation(Activation.RELU)
            .build())

        //Second subsampling layer
        .layer(SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
            .kernelSize(2, 2)
            .stride(2, 2)
            .build())

        //Dense layer
        .layer(DenseLayer.Builder()
            .activation(Activation.RELU)
            .nOut(500)
            .build())

        // Final and fully connected layer with Softmax as activation function
        // multi-label
        .layer(OutputLayer.Builder(LossFunctions.LossFunction.XENT) // aka binary cross entropy
            .nOut(outputNum)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SIGMOID)
            .build())


        .backprop(true).pretrain(false)
        .setInputType(InputType.convolutional(numRows, numColumns, nChannels))
    // InputType.convolutional for normal image
    //        .setInputType(InputType.convolutionalFlat(28, 28, 1))

    val conf = builder.build()

    log.info("Build model....")

    val model = MultiLayerNetwork(conf).apply {
        init()
        addListeners(ScoreIterationListener(1))

        addListeners(object : BaseTrainingListener() {
            override fun onEpochEnd(model: Model?) {
                (model as MultiLayerNetwork).evaluateOn(validationData).let {
                    // tood maybe log epoch with layers.last().epochCount
                    println("model metrics after epoch, are ${it.stats()}")
                }
            }
        })

        println(summary())
    }

    log.info("Train model....")


    //    model.fit(trainData)
    model.fit(epochitTr)

    // do final evaluation
    println("final model performance is ${model.evaluateOn(validationData).stats()}")

    return model;
}


fun prepareTrainingData(file2Label: TwoClassLabelConverter, validatitionProp: Double = 0.2, maxExamples: Int = Int.MAX_VALUE, batchSize: Int = 255): Pair<RecordReaderDataSetIterator, RecordReaderDataSetIterator> {
    val allFiles = File(DATA_ROOT, "train_photos")
        .listFiles({ file -> file.extension == "jpg" }).toList()
        .take(maxExamples)

    val iterator = allFiles.shuffled().iterator()

    val trainData = createTrainRecReaderDataIterator(
        iterator.asSequence().take((allFiles.size * (1 - validatitionProp)).roundToInt()).toList(),
        batchSize = batchSize,
        labelConverter = file2Label
    )

    val validationData = createTrainRecReaderDataIterator(
        iterator.asSequence().toList(),
        batchSize = batchSize,
        labelConverter = file2Label
    )

    return trainData to validationData
}

fun mostRecent(prefix: String): File? = File(".")
    .listFiles(FileFilter { it.name.startsWith(prefix) })
    .sortedBy { it.lastModified() }
    .last()
    .also { println("most recent model is ${it}") }

data class ImageInfo(val rows: Int, val columns: Int, val channels: Int)
