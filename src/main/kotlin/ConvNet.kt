import org.deeplearning4j.eval.Evaluation
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
import java.io.IOException


@Throws(IOException::class)
fun customConfModel(trainData: DataSetIterator, validationData: DataSetIterator, numClasses: Int): MultiLayerNetwork {

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
    //    val epochitTr = MultipleEpochsIterator(4, dataSplit.train)


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
        //        .layer(OutputLayer.Builder(LossFunctions.LossFunction.XENT) // aka binary cross entropy
        //            .nOut(outputNum)
        //            .weightInit(WeightInit.XAVIER)
        //            .activation(Activation.SIGMOID)
        //            .build())
        // single lable
        .layer(OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) // aka binary cross entropy
            .nOut(outputNum)
            .weightInit(WeightInit.XAVIER)
            .activation(Activation.SOFTMAX)
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


    model.fit(trainData)

    // do final evaluation
    println("final model performance is ${model.evaluateOn(validationData).stats()}")

    return model;
}


private fun MultiLayerNetwork.evaluateOn(testData: DataSetIterator): Evaluation {
    // note this way to detect the number of output classes may not work depending on model topology
    //    testData.train.labels.getRow(0).length()
    val numClasses = layers.last().paramTable().get("b")!!.shape()[1]

    val eval = Evaluation(numClasses)

    for (ds in testData) {
        val output = output(ds.featureMatrix, false)
        eval.eval(ds.getLabels(), output)
    }

    return eval
}
