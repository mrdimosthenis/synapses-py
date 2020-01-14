# Synapses

A lightweight **Neural Network** library, for **js**, **jvm** and **.net**.

**Documentation**: https://mrdimosthenis.github.io/Synapses

The interface of the library is common across programming languages.
Take a look at the functions:

##### Neural Network

`NeuralNetwork.init` | param1: `layers` | _returns_
---                  | ---              | ---
JavaScript           | `number[]`       | `NeuralNetwork`
Java                 | `int[]`          | `NeuralNetwork`
Scala                | `List[Int]`      | `NeuralNetwork`
F#                   | `List<int>`      | `NeuralNetwork`

`NeuralNetwork.prediction` | param1: `neuralNetwork` | param2: `inputValues` | _returns_
---                        | ---                     | ---                   | ---
JavaScript                 | `NeuralNetwork`         | `number[]`            | `number[]`
Java                       | `NeuralNetwork`         | `double[]`            | `double[]`
Scala                      | `NeuralNetwork`         | `List[Double]`        | `List[Double]`
F#                         | `NeuralNetwork`         | `List<float>`         | `List<float>`

`NeuralNetwork.fit` | param1: `neuralNetwork` | param2: `learningRate` | param3: `inputValues` | param4: `expectedOutput` | _returns_
---                 | ---                     | ---                    | ---                   | ---                      | ---
JavaScript          | `NeuralNetwork`         | `number`               | `number[]`            | `number[]`               | `NeuralNetwork`
Java                | `NeuralNetwork`         | `double`               | `double[]`            | `double[]`               | `NeuralNetwork`
Scala               | `NeuralNetwork`         | `Double`               | `List[Double]`        | `List[Double]`           | `NeuralNetwork`
F#                  | `NeuralNetwork`         | `float`                | `List<float>`         | `List<float>`            | `NeuralNetwork`

`NeuralNetwork.customizedInit` | param1: `layers` | param2: `activationF`             | param3: `weightInitF` | _returns_
---                            | ---              | ---                               | ---                   | ---
JavaScript                     | `number[]`       | `(number) => ActivationFunction`  | `(number) => number`  | `NeuralNetwork`
Java                           | `int[]`          | `IntFunction<ActivationFunction>` | `IntFunction<Double>` | `NeuralNetwork`
Scala                          | `List[Int]`      | `Int => ActivationFunction`       | `Int => Double`       | `NeuralNetwork`
F#                             | `List<int>`      | `int -> ActivationFunction`       | `int -> float`        | `NeuralNetwork`

##### Data Preprocessor

`DataPreprocessor.init` | param1: `keysWithDiscreteFlags` | param2: `datapoints`            | _returns_
---                     | ---                             | ---                             | ---
JavaScript              | `any[][]`                       | `iterable`                      | `DataPreprocessor`
Java                    | `Object[][]`                    | `Stream<Map<String,String>>`    | `DataPreprocessor`
Scala                   | `List[(String, Boolean)]`       | `LazyList[Map[String, String]]` | `DataPreprocessor`
F#                      | `List<string * bool>`           | `seq<Map<string, string>>`      | `DataPreprocessor`

`DataPreprocessor.encodedDatapoint` | param1: `dataPreprocessor` | param2: `datapoint`   | _returns_
---                                 | ---                        | ---                   | ---
JavaScript                          | `DataPreprocessor`         | `object`              | `number[]`
Java                                | `DataPreprocessor`         | `Map<String,String>`  | `double[]`
Scala                               | `DataPreprocessor`         | `Map[String, String]` | `List[Double]`
F#                                  | `DataPreprocessor`         | `Map<string, string>` | `List<float>`

`DataPreprocessor.decodedDatapoint` | param1: `dataPreprocessor` | param2: `encodedDatapoint` | _returns_
---                                 | ---                        | ---                        | ---
JavaScript                          | `DataPreprocessor`         | `number[]`                 | `object`
Java                                | `DataPreprocessor`         | `double[]`                 | `Map<String,String>`
Scala                               | `DataPreprocessor`         | `List[Double]`             | `Map[String, String]`
F#                                  | `DataPreprocessor`         | `List<float>`              | `Map<string, string>`

##### Statistics

`Statistics.rootMeanSquareError` | param1: `expectedWithOutputValues`       | _returns_
---                              | ---                                      | ---
JavaScript                       | `iterable`                               | `number`
Java                             | `Stream<double[][]>`                     | `double`
Scala                            | `LazyList[(List[Double], List[Double])]` | `Double`
F#                               | `seq<List<float> * List<float>>`         | `float`
