import org.jetbrains.kotlin.config.AnalysisFlag.Flags.experimental
import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import org.jetbrains.kotlin.gradle.dsl.Coroutines
import org.jetbrains.kotlin.js.translate.context.Namer.kotlin
import kotlin.reflect.KClass

buildscript {
    var kotlin_version: String by extra
    kotlin_version = "1.2.40"

    repositories {
        mavenCentral()
    }
    dependencies {
        classpath(kotlinModule("gradle-plugin", kotlin_version))
    }
}

group = "com.github.holgerbrandl.dl"
version = "1.0-SNAPSHOT"

plugins{
    java
    application
//    id("com.bmuschko.docker-java-application") version "3.1.0"
    id("org.jetbrains.kotlin.jvm")version "1.2.40"
}



// https://stackoverflow.com/a/50045271/590437
application {
//    mainClassName = "com.github.holgerbrandl.Tester"
    //    mainClassName = "VggTransferKt"
    mainClassName = "PredictYelpLabelsKt"
}

//http://www.gubatron.com/blog/2017/07/20/how-to-run-your-kotlin-gradle-built-app-from-the-command-line/
//// DO notice the "Kt" suffix on the class name below, if you don't use the Kt generated class you will get errors
//mainClassName = 'com.myapp.MyKotlinAppKt'
//project.convention.findByType(ApplicationPluginConvention::class.java)!!.mainClassName = "com.github.holgerbrandl.Tester"

//// optional: add one string per argument you want as the default JVM args
//applicationDefaultJvmArgs = ["-Xms512m", "-Xmx1g"]

val kotlin_version: String by extra

repositories {
    mavenCentral()
    maven("https://oss.sonatype.org/content/repositories/snapshots")
}

val nd4jVersion = "1.0.0-alpha"
// https://deeplearning4j.org/snapshots
//val nd4jVersion = "1.0.0-SNAPSHOT"

dependencies {
    compile(kotlinModule("stdlib-jdk8", kotlin_version))

    compile("org.nd4j","nd4j-native-platform","1.0.0-alpha")
    //    compile("org.nd4j","nd4j-cuda-8.0-platform","1.0.0-alpha")
    compile("org.nd4j","nd4s_2.11","0.7.2")
    compile("org.deeplearning4j", "deeplearning4j-core", nd4jVersion)
    compile("org.deeplearning4j", "deeplearning4j-zoo", nd4jVersion)
    compile("org.deeplearning4j", "deeplearning4j-nn", nd4jVersion)

    // http://saltnlight5.blogspot.de/2013/08/how-to-configure-slf4j-with-different.html
    // compile("org.slf4j:slf4j-simple:1.7.25")
    compile("org.slf4j", "slf4j-jdk14", "1.7.5")

    //    compile("org.deeplearning4j" , "deeplearning4j-ui","1.0.0-alpha")
    compile(  "org.imgscalr" , "imgscalr-lib" , "4.2")


    compile("me.tongfei", "progressbar", "0.6.0")

    compile("com.github.haifengl", "smile-core", "1.5.1")

    testCompile("junit", "junit", "4.12")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}

tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}



// https://android.jlelse.eu/custom-gradle-tasks-with-kotlin-e0f8659628a6
// https://stackoverflow.com/questions/39576170/proper-way-to-run-kotlin-application-from-gradle-task

//task MLPMnistSingleLayerExample(type: JavaExec) {
//    classpath = sourceSets.main.runtimeClasspath
//    main = "org.deeplearning4j.examples.feedforward.mnist.MLPMnistSingleLayerExample"
//}