import smile.classification.NaiveBayes
import smile.classification.NaiveBayes.Model.MULTINOMIAL
import smile.math.SparseArray
import kotlin.coroutines.experimental.buildSequence

/**
 * @author Holger Brandl
 */

fun main(args: Array<String>) {
    val multiNomialNB = NaiveBayes(MULTINOMIAL, doubleArrayOf(0.4, 0.1, 0.5), 2)

    //    multiNomialNB.lea

    val classGenerator = buildSequence {
        yield(multiNomialNB.predict(SparseArray().apply { append(2, 1.0) }))
    }

    println("predictions are ${classGenerator.take(20).joinToString()}")
}