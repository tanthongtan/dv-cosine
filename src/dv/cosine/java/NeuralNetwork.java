package dv.cosine.java;

import de.bwaldvogel.liblinear.SolverType;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class NeuralNetwork {

    /* Parameters here */
    private static final int gram = 3;
    private static double lr = 0.001;
    private static final int negSize = 5;
    private static int iter = 120;
    private static final int batchSize = 100;
    private static final int n = 500;
    private static final int a = 6;

    private static final int numThreads = 22;
    private static final boolean saveVecs = true;

    private static final boolean tuning = false;
    /* */

    private static double[][] WV;
    private static double[][] WP;

    private static Random random = new Random();

    public static void main(String[] args) {
        System.out.println("Reading Documents");
        List<Document> allDocs = null;
        allDocs = Dataset.getImdbDataset(gram);
        
        List<Document> docList = new ArrayList<Document>(allDocs);
        List<Document> trainDocs = new ArrayList<Document>();
        List<Document> testDocs = new ArrayList<Document>();
        for (Document doc : allDocs) {
            if (doc.split.equals("train")) {
                trainDocs.add(doc);
            } else if (doc.split.equals("test")) {
                testDocs.add(doc);
            }
        }

        if (!tuning) {
            learnEmbeddingsAndTest(trainDocs, testDocs, allDocs,docList);
        } else if (tuning) {
            Collections.shuffle(trainDocs);
            List<Document> devDocs = trainDocs.subList(0, trainDocs.size()/5);
            List<Document> devTrainDocs = trainDocs.subList(trainDocs.size()/5,trainDocs.size());
            double bestAccuracy = 0;
            int[] iters = {20, 40, 80, 120};
            double[] lrs = {0.25, 0.025, 0.0025, 0.001};
            for (int iterTemp : iters) {
                for (double lrTemp : lrs) {
                    iter = iterTemp;
                    lr = lrTemp;
                    double accuracy = learnEmbeddingsAndTest(devTrainDocs, devDocs, allDocs,docList);
                    System.gc();
                    writeToFileForTuning(false, accuracy);
                    if (accuracy > bestAccuracy) {
                        bestAccuracy = accuracy;
                    }
                }
            }
            writeToFileForTuning(true, bestAccuracy);
        }
    }

    private static void writeToFileForTuning(boolean best, double accuracy) {
        try {
            FileWriter fw;
            if (best) {
                fw = new FileWriter(gram + lr + negSize + iter + batchSize + n + a + "best.txt");
            }
            fw = new FileWriter(gram + lr + negSize + iter + batchSize + n + a + ".txt");
            fw.write("gram = " + gram + "\n");
            fw.write("lr = " + lr + "\n");
            fw.write("negSize = " + negSize + "\n");
            fw.write("iter = " + iter + "\n");
            fw.write("batchSize = " + batchSize + "\n");
            fw.write("n = " + n + "\n");
            fw.write("a = " + a + "\n");
            fw.write("accuracy=" + accuracy + "\n");
            if (best) {
                fw.write("best accuracy\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double learnEmbeddingsAndTest(List<Document> trainDocs, List<Document> testDocs, List<Document> allDocs, List<Document> docList) {
        double accuracy = 0;
        Dataset.initSum();
        System.out.println("Initializing network");
        initNet(allDocs);

        for (int epoch = 0; epoch < iter; epoch++) {
            int startEpoch = (int) System.currentTimeMillis();
            System.out.printf("%d::\n", epoch);
            int p = 0;
            Collections.shuffle(docList);
            ExecutorService pool = Executors.newFixedThreadPool(numThreads);
            while (true) {
                if (p < (docList.size() / batchSize)) {
                    int s = batchSize * p;
                    int e = batchSize * p + batchSize;
                    if (docList.size() < e) {
                        e = docList.size();
                    }
                    pool.execute(new TrainThread(docList.subList(s, e)));
                    p += 1;
                } else {
                    break;
                }
            }
            pool.shutdown();
            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            int endEpoch = (int) System.currentTimeMillis();
            System.out.printf("time: %d seconds\n", (endEpoch - startEpoch) / 1000);

            Classifier binaryClassifier = new Classifier(SolverType.L2R_LR, 1.0, 0.01);
            binaryClassifier.train(WP, trainDocs);
            accuracy = binaryClassifier.score(WP, testDocs);

        }

        if (saveVecs) {
            try {
                FileWriter fw_train = new FileWriter("train_vectors.txt");
                FileWriter fw_test = new FileWriter("test_vectors.txt");

                for (Document doc : allDocs) {
                    if (doc.split.equals("extra")) {
                        continue;
                    }
                    FileWriter fw = fw_train;
                    if (doc.split.equals("test")) {
                        fw = fw_test;
                    }
                    fw.write(doc.sentiment + "\t");
                    for (int i = 0; i < n; i++) {
                        fw.write(WP[doc.tag][i] + "\t");
                    }
                    fw.write("\n");
                }

                fw_train.close();
                fw_test.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        WV = null;
        WP = null;
        return accuracy;
    }

    public static void initNet(List<Document> allDocs) {
        int v = Dataset.wordIdCounts.size();
        int p = allDocs.size();
        System.out.println("v=" + v + " p=" + p);
        WV = new double[v][];
        int realV = 0;
        for (int i = 0; i < v; i++) {
            if (Dataset.wordIdCounts.get(i) >= 2) {
                realV++;
                WV[i] = new double[n];
                for (int j = 0; j < n; j++) {
                    WV[i][j] = (random.nextFloat() - 0.5f) / n;
                }
            }
        }

        System.out.println("realV=" + realV);
        WP = new double[p][n];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < n; j++) {
                WP[i][j] = (random.nextFloat() - 0.5f) / n;
            }
        }
    }

    private static class TrainThread implements Runnable {

        private List<Document> docSubList;

        public TrainThread(List<Document> docSubList) {
            this.docSubList = docSubList;
        }

        public void run() {
            train();
        }

        private void train() {
            double[] temp = new double[n];
            for (Document doc : docSubList) {
                int pi = doc.tag;
                int[] ids = doc.wordIds;
                int[] forIds = getRandomPermutation(ids.length);
                for (int l : forIds) {
                    Arrays.fill(temp, 0);
                    backprop(WP[pi], WV[ids[l]], 1, temp);
                    for (int i = 0; i < negSize; i++) {
                        backprop(WP[pi], WV[Dataset.getRandomWordId()], 0, temp);
                    }
                    for (int i = 0; i < n; i++) {
                        WP[pi][i] += temp[i];
                    }
                }
            }
        }

        private void backprop(double[] h, double[] v, double t, double[] temp) {
            if (h == null || v == null) {
                return;
            }
            double h_dot_v = 0;
            double mag_v = 0;
            double mag_h = 0;
            for (int i = 0; i < n; i++) {
                h_dot_v += h[i] * v[i];
                mag_v += v[i] * v[i];
                mag_h += h[i] * h[i];
            }
            mag_v = sqrt(mag_v);
            mag_h = sqrt(mag_h);
            double cos_theta = h_dot_v / (mag_v * mag_h);
            double y = 1.0 / (1 + exp(-a * cos_theta));
            for (int i = 0; i < n; i++) {
                temp[i] += -(y - t) * a * (v[i] / (mag_v * mag_h) - h[i] * (h_dot_v) / (pow(mag_h, 3) * mag_v)) * lr;
                v[i] += -(y - t) * a * (h[i] / (mag_v * mag_h) - v[i] * (h_dot_v) / (pow(mag_v, 3) * mag_h)) * lr;
            }
        }

        private int[] getRandomPermutation(int length) {
            int[] array = new int[length];
            for (int i = 0; i < array.length; i++) {
                array[i] = i;
            }

            for (int i = 0; i < length; i++) {
                int ran = i + random.nextInt(length - i);
                int temp = array[i];
                array[i] = array[ran];
                array[ran] = temp;
            }
            return array;
        }
    }
}
