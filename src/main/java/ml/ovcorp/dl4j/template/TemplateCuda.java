package ml.ovcorp.dl4j.template;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Hello world!
 *
 */
public class TemplateCuda
{
    private static final Logger log = LoggerFactory.getLogger(TemplateCuda.class);

    public static void main(String... args) throws Exception {
        int nChannels = 1; // Number of input channels
        int nClasses = 10; // The number of possible outcomes
        int height = 28;
        int width = 28;
        int batchSize = 50; // Test batch size
        int nEpochs = 5; // Number of training epochs
        int seed = 12345; //

        // Create an iterator using the batch size for one iteration
        log.info("Загрузка данных MNIST для обучения...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize,true,12345);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize,false,12345);

        log.info("Построение модели...");
        ComputationGraphConfiguration config = new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(seed)
                .updater(new Adam())
                .graphBuilder()
                .addInputs("input")
                // Input Image
                .setInputTypes(InputType.convolutionalFlat(height, width, nChannels))
                // 1. Convolutional Layer - Start Encoder
                .addLayer("cnn", new ConvolutionLayer.Builder()
                        .nOut(256)
                        .kernelSize(9, 9)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build(), "input")
                // 2. Primary Capsules
                .addLayer("primary_capsules", new PrimaryCapsules.Builder(8, 32)
                        .kernelSize(9, 9)
                        .stride(2, 2)
                        .build(), "cnn")
                // 3. Digital Capsules - End Encoder
                .addLayer("digit_capsules", new CapsuleLayer.Builder(nClasses, 16, 1).build(), "primary_capsules")
                // 4. Start Decoder
                .addLayer("decoder1", new CapsuleStrengthLayer.Builder().build(), "digit_capsules")
                .addLayer("decoder2", new ActivationLayer.Builder(new ActivationSoftmax()).build(), "decoder1")
                .addLayer("decoder3", new LossLayer.Builder(new LossNegativeLogLikelihood()).build(), "decoder2")
                .setOutputs("decoder3")
                .build();

        ComputationGraph net = new ComputationGraph(config);
        net.init();

        log.info("Старт обучения...");
        net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        net.fit(mnistTrain, nEpochs);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "template_model.zip");

        log.info("Сохранения модели: {}", path);
        net.save(new File(path), true);

        log.info("Завершение работы программы.");
    }
}
