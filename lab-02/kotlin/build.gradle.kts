import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm") version "2.4.21"
    application
}

repositories { mavenCentral() }

kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }
tasks.withType<JavaCompile>().configureEach { options.release.set(17) }
application { mainClass.set("MainKt") }
