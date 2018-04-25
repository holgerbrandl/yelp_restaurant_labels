import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

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

plugins {
    java
}

group = "com.github.holgerbrandl.dl"
version = "1.0-SNAPSHOT"

apply {
    plugin("kotlin")
}

val kotlin_version: String by extra

repositories {
    mavenCentral()
}

dependencies {
    compile(kotlinModule("stdlib-jdk8", kotlin_version))

    compile("org.nd4j","nd4j-native-platform","1.0.0-alpha")
    compile("org.nd4j","nd4s_2.11","0.7.2")
    compile("org.deeplearning4j","deeplearning4j-core","1.0.0-alpha")
//    compile("org.deeplearning4j" , "deeplearning4j-ui","1.0.0-alpha")

    testCompile("junit", "junit", "4.12")
}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}
tasks.withType<KotlinCompile> {
    kotlinOptions.jvmTarget = "1.8"
}