package ml.ovcorp.dl4j.template;

import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

/**
 * Hello world!
 */
public class TemplateCuda {
    private static final Logger log = LoggerFactory.getLogger(TemplateCuda.class);

    public static void main(String... args) throws Exception {
        int batchSize = 50; // Размер пакета выборки
        int nEpochs = 1; // Количество тренеровочных эпох
        int seed = 12345; // Коэффициент рассеивания данных

        // Create an iterator using the batch size for one iteration
        log.info("Загрузка данных MNIST для обучения...");
        DataSetIterator mnistTrain = getTrains(batchSize, seed);
        DataSetIterator mnistTest = getTests(batchSize, seed);

        log.info("Построение модели...");
        ComputationGraph net = new ComputationGraph(getLenetConf(seed));
        net.init();

        log.info("Конфигурация модели:\n{}", net.getConfiguration().toJson());
        log.info("Количество параметров модели: {}", net.numParams());
        log.info("Анализ информации о слоях.");
        int i = 0;
        for (Layer l : net.getLayers()) {
            log.info("{}. Тип слоя: {}. Количество параметров слоя: {}.",
                    ++i,
                    l.type(),
                    l.numParams());
        }

        log.info("Старт обучения...");
        LocalDateTime start = LocalDateTime.now();
        log.info("Время начала обучения: {}", start);
        net.setListeners(new ScoreIterationListener(1), new EvaluativeListener(mnistTest, 1, InvocationType.EPOCH_END)); //Print score every 10 iterations and evaluate on test set every epoch
        net.fit(mnistTrain, nEpochs);
        LocalDateTime end = LocalDateTime.now();
        log.info("Время конца обучения: {}", end);
        diffLocalDateTime(start, end);

        String path = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "template_model.zip");

        log.info("Сохранения модели: {}", path);
        net.save(new File(path), true);

        log.info("Завершение работы программы.");
    }

    private static DataSetIterator getTrains(int batchSize, int seed) throws IOException {
        return new MnistDataSetIterator(batchSize, true, seed);
    }

    private static DataSetIterator getTests(int batchSize, int seed) throws IOException {
        return new MnistDataSetIterator(batchSize, false, seed);
    }

    private static ComputationGraphConfiguration getLenetConf(int seed) {
        int nChannels = 1;
        int nClasses = 10;
        int height = 28;
        int width = 28;

        return new NeuralNetConfiguration.Builder()
                .cudnnAlgoMode(ConvolutionLayer.AlgoMode.NO_WORKSPACE)
                .seed(seed)
                .l2(0.005)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta())
                .graphBuilder()
                .addInputs("input")
                .setInputTypes(InputType.convolutionalFlat(height, width, nChannels))
                .addLayer("cnn1",
                        new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0})
                                .name("cnn1")
                                .nIn(nChannels)
                                .nOut(50)
                                .biasInit(0)
                                .build(), "input")
                .addLayer("maxpool1",
                        new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                                .name("maxpool1")
                                .build(), "cnn1")
                .addLayer("cnn2",
                        new ConvolutionLayer.Builder(new int[]{5, 5}, new int[]{5, 5}, new int[]{1, 1})
                                .name("cnn2")
                                .nOut(100)
                                .biasInit(0)
                                .build(), "maxpool1")
                .addLayer("maxpool2",
                        new SubsamplingLayer.Builder(new int[]{2, 2}, new int[]{2, 2})
                                .name("maxool2")
                                .build(), "cnn2")
                .addLayer("denseLayer",
                        new DenseLayer.Builder()
                                .nOut(500)
                                .build(), "maxpool2")
                .addLayer("outputLayer",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nOut(nClasses)
                                .activation(Activation.SOFTMAX)
                                .build(), "denseLayer")
                .setOutputs("outputLayer")
                .build();
    }

    private static ComputationGraphConfiguration getCapsNetConf(int seed) {
        int nChannels = 1;
        int nClasses = 10;
        int height = 28;
        int width = 28;

        return new NeuralNetConfiguration.Builder()
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
    }

    private static void diffLocalDateTime(LocalDateTime from, LocalDateTime to) {

        LocalDateTime tempDateTime = LocalDateTime.from(from);

        StringBuilder timeString = new StringBuilder();

        long years = tempDateTime.until(to, ChronoUnit.YEARS);
        tempDateTime = tempDateTime.plusYears(years);

        if (years > 0) {
            timeString.append(years).append(" y ");
        }

        long months = tempDateTime.until(to, ChronoUnit.MONTHS);
        tempDateTime = tempDateTime.plusMonths(months);

        if (months > 0) {
            timeString.append(months).append(" mn ");
        }

        long days = tempDateTime.until(to, ChronoUnit.DAYS);
        tempDateTime = tempDateTime.plusDays(days);

        if (days > 0) {
            timeString.append(days).append(" d ");
        }

        long hours = tempDateTime.until(to, ChronoUnit.HOURS);
        tempDateTime = tempDateTime.plusHours(hours);

        if (hours > 0) {
            timeString.append(hours).append(" h ");
        }

        long minutes = tempDateTime.until(to, ChronoUnit.MINUTES);
        tempDateTime = tempDateTime.plusMinutes(minutes);

        if (minutes > 0) {
            timeString.append(minutes).append(" m ");
        }

        long seconds = tempDateTime.until(to, ChronoUnit.SECONDS);
        tempDateTime = tempDateTime.plusSeconds(seconds);

        if (seconds > 0) {
            timeString.append(seconds).append(" s ");
        }

        long milliseconds = tempDateTime.until(to, ChronoUnit.MILLIS);

        if (milliseconds > 0) {
            timeString.append(milliseconds).append(" ms ");
        }

        log.info("Общее время обучения составило: {}", timeString);
    }
}
