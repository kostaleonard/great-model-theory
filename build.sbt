ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.9"

lazy val root = (project in file("."))
  .settings(
    name := "igneous",
    libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.12" % Test
  )
