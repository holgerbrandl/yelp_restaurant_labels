import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.logging.Level
import java.util.logging.LogManager

/**
 * @author Holger Brandl
 */
fun readPhoto2BusinessModel(mappingFile: File): Map<String, String> {
    return mappingFile
        .readLines()
        .drop(1)
        .map { it.split(",").let { it[0] to it[1] } }.toMap()
}

internal fun configureLogger() {
    // http://saltnlight5.blogspot.de/2013/08/how-to-configure-slf4j-with-different.html

    //    https://stackoverflow.com/questions/4311026/how-to-get-slf4j-hello-world-working-with-log4j
    //    The log levels are ERROR > WARN > INFO > DEBUG > TRACE.
    //    System.setProperty(org.slf4j.impl.SimpleLogger.DEFAULT_LOG_LEVEL_KEY, "TRACE");
    //    val root = LoggerFactory.getLogger(ROOT_LOGGER_NAME) as SimpleLogger

    // https://stackoverflow.com/questions/194765/how-do-i-get-java-logging-output-to-appear-on-a-single-line
    System.setProperty(
        "java.util.logging.SimpleFormatter.format",
        """%1${'$'}tY-%1${'$'}tm-%1${'$'}td %1${'$'}tH:%1${'$'}tM:%1${'$'}tS %4${'$'}s %2${'$'}s %5${'$'}s%6${'$'}s%n"""
    )

    // https://nozaki.me/roller/kyle/entry/java-util-logging-programmatic-configuration
    //    java.util.logging.LogManager.getLogManager().getLogger(ZooModel::class.java.simpleName).level = Level.FINEST
    LogManager.getLogManager().getLogger("").level = Level.FINEST

//    LoggerFactory.getLogger("foo").info("hello info")
//    LoggerFactory.getLogger("foo").warn("hello warn")
//    LoggerFactory.getLogger("foo").debug("hello debug")
}

internal val now get() = LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME).replace(":", "_")