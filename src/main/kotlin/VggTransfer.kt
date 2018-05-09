import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator
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
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.SplitTestAndTrain
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions
import java.util.*
import kotlin.math.ceil


internal fun buildVggTransferModel(allTrainDS: DataSet): ComputationGraph? {

    val splitTrainNum = ceil(allTrainDS.numExamples() * 0.8).toInt() // 80/20 training/test split
    val dataSplit: SplitTestAndTrain = allTrainDS.splitTestAndTrain(splitTrainNum, Random(42))

    val batchSize = 128
    val trainData = ListDataSetIterator(dataSplit.train.asList(), batchSize)
    //    val testData = ListDataSetIterator(dataSplit.test.asList(), batchSize)

    // todo later: run multiple epochs over the data
    //        val numEpochs = 15 // number of epochs to perform
    val trainDataEpochs = MultipleEpochsIterator(4, trainData)
    //

    val vgg = VGG19().initPretrained() as ComputationGraph


    //number of rows and columns in the input pictures
    print("structure of vgg model is: ${vgg.summary()}")

    // from /Users/brandl/projects/deep_learning/dl4j-examples/dl4j-spark-examples/dl4j-spark/src/playground.playground.main/java/org/deeplearning4j/transferlearning/vgg16/README.md
    val fineTuneConf = FineTuneConfiguration.Builder()
        .updater(Nesterovs(5e-5))
        .seed(23)
        .build()

    //
    // // fine-tune (not what we want here
    //
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

    // see /Users/brandl/projects/deep_learning/dl4j-examples/dl4j-spark-examples/dl4j-spark/src/playground.playground.main/java/org/deeplearning4j/transferlearning/vgg16/TransferLearning.md
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

    model.setListeners(ScoreIterationListener(1))  //print the score with every iteration

    model.setListeners(object : BaseTrainingListener() {
        override fun iterationDone(model: Model?, iteration: Int, epoch: Int) {
            val validationScore = (model as MultiLayerNetwork).score(dataSplit.test)
            println("neg log likelihood after ${iteration} iterations on validation set is ${validationScore} ")
        }
    })

    //     other vgg16 transfer learning example
    //    /Users/brandl/projects/deep_learning/dl4j-examples/dl4j-examples/src/playground.playground.main/java/org/deeplearning4j/examples/transferlearning/vgg16/EditLastLayerOthersFrozen.java
    //    // train shape
    //    result = {int[4]@2386}
    //    0 = 15
    //    1 = 3
    //    2 = 224
    //    3 = 224


    // but we get
    //train data shape is 50, 3, 64, 64
    //Invalid input array: expected shape [minibatch, channels, height, width] = [minibatch, 512, 7, 7] - got [50, 512, 2, 2]
    println("train data shape is ${allTrainDS.featureMatrix.shape().joinToString()}")
    //    model.fit(allTrainDS)
    model.fit(trainData)
    //    model.fit(trainDataEpochs)

    return model
}


