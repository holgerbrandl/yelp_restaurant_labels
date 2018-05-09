import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD
import java.io.File
import java.util.*
import javax.imageio.ImageIO
import kotlin.coroutines.experimental.buildSequence
import kotlin.math.ceil


/**
 * @author Holger Brandl
 */

//const val NUM_CLASSES = 10 // number of output classes
const val NUM_CLASSES = 2 // number of output classes

//val DATA_ROOT = File("/Users/brandl/projects/deep_learning/dl4j/kaggle_yelp/data")
//val DATA_ROOT = File(System.getProperty("user.home"), "projects/data/yelp-restaurant-photos")
val DATA_ROOT by lazy {
    if (System.getProperty("os.name").contains("Mac OS"))
    //        File("/Volumes/talisker/projects/data/yelp-restaurant-photos")
    //    File(System.getProperty("user.home"), "projects/data/yelp-restaurant-photos")
        File("/Users/brandl/projects/deep_learning/kaggle_yelp_rest_pics/data")

    else {
        File(System.getProperty("user.home"), "projects/data/yelp-restaurant-photos")
    }
}


fun loadImages(path: File): Pair<List<File>, List<DoubleArray>> {
    val images = path.listFiles({ file -> file.extension == "jpg" }).toList()

    val imageData = images.map {
        //        println("reading image ${it}")
        ImageIO.read(it)
            .makeSquare()
            .resize() // (200, 200)
            .imageAsGreyVec()
    }

    return Pair(images, imageData)
}


data class ImageDataSet(val imagesFiles: List<File>, val ds: DataSet)


fun prepareTrainData(path: File, resizeImgDim: Int = 64): ImageDataSet {
    val (images, imageData) = loadImages(path)

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

    val ndFeatures = Nd4j.vstack(imageData.map { Nd4j.create(it) })
    val ndLabels = Nd4j.vstack(labels.map { Nd4j.create(it) })

    return ImageDataSet(images, DataSet(ndFeatures, ndLabels))
}


fun main(args: Array<String>) {
    configureLogger()

    // read and pre-process the images
    val allTrainDS = prepareTrainData(File(DATA_ROOT, "train_photos")).ds

    // https://nd4j.org/userguide#serialization
    // https://stackoverflow.com/questions/4841340/what-is-the-use-of-bytebuffer-in-java
    //    allTrainDS.save(File("yelp_train.dat"))
    //    DataSet().load(File("yelp_train.dat"))


    // no need to do this manually, since its done by splitTestAndTrain
    //    val someSeed=42.toLong()
    //    Nd4j.shuffle(allTrainDS.featureMatrix,  Random(someSeed), 1) // this changes ds.  Shuffles rows
    //    Nd4j.shuffle(allTrainDS.labels,  Random(someSeed), 1) // this changes ds.  Shuffles labels accordingly


    println("Building model....")

    val model = buildDenseModel(allTrainDS)


    model.save(File("dense_model.${now}.dat"))
    //    MultiLayerNetwork.load()

    println("Evaluate model....")


    fun prepareTestData(path: File): Pair<List<File>, INDArray> {
        val (images, imageData) = loadImages(path)

        val ndFeatures = Nd4j.vstack(imageData.map { Nd4j.create(it) })

        return images to ndFeatures
    }

    val evalData = prepareTestData(File(DATA_ROOT, "test_photos"))

    val eval = Evaluation(NUM_CLASSES) //create an evaluation object with 10 possible classes
    val testOuput = model.output(evalData.second)  // no first in case of MultiLayerNetwork

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

    print(eval.stats())

}

internal fun buildDenseModel(allTrainDS: DataSet): MultiLayerNetwork {

    val splitTrainNum = ceil(allTrainDS.numExamples() * 0.8).toInt() // 80/20 training/test split
    val dataSplit: SplitTestAndTrain = allTrainDS.splitTestAndTrain(splitTrainNum, Random(42))

    // use a mini-batch iterator
    val batchSize = 128
    val trainData = ListDataSetIterator(dataSplit.train.asList(), batchSize)

    // todo later: run multiple epochs over the data
    val numEpochs = 15 // number of epochs to perform
    val epochitTr = MultipleEpochsIterator(numEpochs, dataSplit.train)



    //number of rows and columns in the input pictures
    val numRows = 64
    val numColumns = 64
    val rngSeed = 123 // random number seed for reproducibility
    val rate = 0.001 // learning rate


    // more or less the same as MLPMnistTwoLayerExample

    val conf = NeuralNetConfiguration.Builder()
        .seed(rngSeed.toLong()) //include a random seed for reproducibility
        // use stochastic gradient descent as an optimization algorithm

        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .updater(Nesterovs(rate, 0.98)) //specify the rate of change of the learning rate.
        .l2(rate * 0.05) // regularize learning model
        .list()
        .layer(0, DenseLayer.Builder() //create the first input layer.
            .nIn(numRows * numColumns)
            .nOut(500)
            .build())
        .layer(1, DenseLayer.Builder() //create the second input layer
            .nIn(500)
            .nOut(100)
            .build())
        .layer(2, OutputLayer.Builder(NEGATIVELOGLIKELIHOOD) //create hidden layer
            .activation(Activation.SOFTMAX)
            .nIn(100)
            .nOut(NUM_CLASSES)
            .build())
        .pretrain(false).backprop(true) //use backpropagation to adjust weights
        .build()

    val model = MultiLayerNetwork(conf)
    model.init()


    model.setListeners(ScoreIterationListener(1))  //print the score with every iteration

    model.summary()
    model.setListeners(object : BaseTrainingListener() {
        override fun iterationDone(model: Model?, iteration: Int, epoch: Int) {
            val validationScore = (model as MultiLayerNetwork).score(dataSplit.test)
            println("neg log likelihood after ${iteration} iterations on validation set is ${validationScore} ")
        }
    })


    model.fit(trainData)

    return model
}



