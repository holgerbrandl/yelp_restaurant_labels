import org.datavec.image.loader.Java2DNativeImageLoader
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
import org.deeplearning4j.eval.Evaluation
import org.deeplearning4j.nn.api.Model
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration
import org.deeplearning4j.nn.transferlearning.TransferLearning
import org.deeplearning4j.optimize.api.BaseTrainingListener
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.zoo.model.VGG19
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.io.File
import java.util.*
import javax.imageio.ImageIO
import kotlin.coroutines.experimental.buildSequence
import kotlin.math.ceil

/**
 * @author Holger Brandl
 */


fun loadImagesVGG(path: File): Pair<List<File>, List<INDArray>> {
    val images = path.listFiles({ file -> file.extension == "jpg" }).toList()


    val imageData = images.map {
        //        println("reading image ${it}")
        ImageIO.read(it)
            .makeSquare()
            // todo it seems that vgg is using 224x224 as input dim
            .resize()
            .removeAlphaChannel()
            .let {
                Java2DNativeImageLoader(it.height, it.width, 3).asMatrix(it)
            }
        //            .imageAsGreyVec()
    }

    return Pair(images, imageData)
}


fun prepareTrainDataVG(path: File, resizeImgDim: Int = 64): ImageDataSet {
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




internal fun buildVggTransferModel(allTrainDS: DataSet): ComputationGraph? {

    val splitTrainNum = ceil(allTrainDS.numExamples() * 0.8).toInt() // 80/20 training/test split
    val dataSplit: SplitTestAndTrain = allTrainDS.splitTestAndTrain(splitTrainNum, Random(42))

    val batchSize = 128
    val trainData = ListDataSetIterator(dataSplit.train.asList(), batchSize)
    //    val testData = ListDataSetIterator(dataSplit.test.asList(), batchSize)

    // todo later: run multiple epochs over the data
        val numEpochs = 15 // number of epochs to perform
    //    val epochitTr = MultipleEpochsIterator(4, dataSplit.train)


    val vgg = VGG19().initPretrained() as ComputationGraph


    //number of rows and columns in the input pictures
    val numRows = 64
    val numColumns = 64





    print("structure of vgg model is: $vgg")


    //
    // fine-tune (not what we want here
    //

    // from /Users/brandl/projects/deep_learning/dl4j-examples/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/transferlearning/vgg16/README.md
    val fineTuneConf = FineTuneConfiguration.Builder()
        .updater(Nesterovs(5e-5))
        .seed(23)
        .build()
    //
    //
    //    val model = TransferLearning.Builder(vgg)
    //        .fineTuneConfiguration(fineTuneConf)
    //        .setFeatureExtractor(vgg.layerNames.indexOf("block4_pool"))
    //        .build();

    //
    // new output layer
    //
    //    LogManager.getRootLogger().setLevel(Level.DEBUG);
    //    LogManager.getLogManager().getLogger("com.my.company").setLevel(whateverloglevel)

    //    println(vgg.configuration.toJson())

    // see /Users/brandl/projects/deep_learning/dl4j-examples/dl4j-spark-examples/dl4j-spark/src/main/java/org/deeplearning4j/transferlearning/vgg16/TransferLearning.md
    val model = TransferLearning.GraphBuilder(vgg)
        .fineTuneConfiguration(fineTuneConf)
        .setFeatureExtractor("fc2")
        .removeVertexKeepConnections("predictions")
        .addLayer("predictions", OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
            .activation(Activation.SOFTMAX)
            .nIn(4096)
            .nOut(NUM_CLASSES)
            .build(), "fc2")
        .build()

    println(model.summary())

    model.setListeners(ScoreIterationListener(1))  //print the score with every iteration

    model.setListeners(object : BaseTrainingListener() {
        override fun iterationDone(model: Model?, iteration: Int, epoch: Int) {
            val validationScore = (model as MultiLayerNetwork).score(dataSplit.test)
            println("neg log likelihood after ${iteration} iterations on validation set is ${validationScore} ")
        }
    })


    for (i in 0..numEpochs - 1) {
        println("Epoch " + i)
        model.fit(allTrainDS.iterator().next())
    }

    return model
}

fun main(args: Array<String>) {
    configureLogger()

    // read and pre-process the images
    val allTrainDS = prepareTrainDataVG(File(DATA_ROOT, "train_photos")).ds

    // https://nd4j.org/userguide#serialization
    // https://stackoverflow.com/questions/4841340/what-is-the-use-of-bytebuffer-in-java
    //    allTrainDS.save(File("yelp_train.dat"))
    //    DataSet().load(File("yelp_train.dat"))


    // no need to do this manually, since its done by splitTestAndTrain
    //    val someSeed=42.toLong()
    //    Nd4j.shuffle(allTrainDS.featureMatrix,  Random(someSeed), 1) // this changes ds.  Shuffles rows
    //    Nd4j.shuffle(allTrainDS.labels,  Random(someSeed), 1) // this changes ds.  Shuffles labels accordingly


    println("Building model....")

    //    val model = buildDenseModel(allTrainDS)
    //    val model = buildVggTransferModel(allTrainDS)!!
    val model = customConfModel(allTrainDS);
    //

    model.save(File("dense_model.${now}.dat"))
    //    MultiLayerNetwork.load()

    println("Evaluate model....")


    fun prepareTestData(path: File): Pair<List<File>, INDArray> {
        val (images, imageData) = loadImagesVGG(path)

        val ndFeatures = Nd4j.vstack(imageData.map { it })

        return images to ndFeatures
    }

    val evalData = prepareTestData(File(DATA_ROOT, "test_photos"))

    val eval = Evaluation(NUM_CLASSES) //create an evaluation object with 10 possible classes
    val modelOut = model.output(evalData.second)

    // recast output depending on model type (multilayer vs computation graph
    val testOuput =  when(modelOut){
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

    print(eval.stats())

}

