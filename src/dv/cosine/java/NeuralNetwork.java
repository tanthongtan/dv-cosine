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
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class NeuralNetwork {

    /* Parameters here */
    private static final int gram = 3;
    private static double lr = 0.001;
    private static final int negSize = 5;
    private static int iter = 120;
    private static final int batchSize = 100; //we use SGD not minibatchGD, but batchSize just determines how many documents are assigned to each thread
    private static final int n = 500;
    private static int a = 6; //only for cosine
    private static boolean lrAnnealing = false;
    private static final String mode = "cosinesimilarity"; //"cosinesimilarity" to use cosine similarity, "dotproduct" to use dot product, and "l2rdotproduct" to use L2 regularized dot product
    private static double lambda = 0.01; //only for L2R dot product

    private static final int numThreads = 22;
    private static final boolean saveVecs = true;

    private static final boolean tuning = false;
    /* */

    private static double[][] WV;
    private static double[][] WP;

    private static Random random = new Random();
    private static double originalLr; //only used in lrAnnealing == true

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
            learnEmbeddingsAndTest(trainDocs, testDocs, allDocs, docList);
        } else if (tuning) {
            Collections.shuffle(trainDocs);
            List<Document> devDocs = trainDocs.subList(0, trainDocs.size() / 5);
            List<Document> devTrainDocs = trainDocs.subList(trainDocs.size() / 5, trainDocs.size());
            double bestAccuracy = 0;
            String[] bestParams = null;
            int[] iters = {10, 20, 40, 80, 120};
            double[] lrs = {0.25, 0.025, 0.0025, 0.001};
            int[] as = {4, 6, 8};
            boolean[] lrAnnealings = {true, false};
            for (boolean lrAnnealingTemp : lrAnnealings) {
                for (int aTemp : as) {
                    for (int iterTemp : iters) {
                        for (double lrTemp : lrs) {
                            iter = iterTemp;
                            lr = lrTemp;
                            lrAnnealing = lrAnnealingTemp;
                            a = aTemp;
                            double accuracy = learnEmbeddingsAndTest(devTrainDocs, devDocs, allDocs, docList);
                            System.gc();
                            writeToFileForTuning(accuracy);
                            if (accuracy > bestAccuracy) {
                                bestAccuracy = accuracy;
                                bestParams = new String[]{gram + "", originalLr + "", negSize + "", iter + "", batchSize + "", n + "", a + "", lrAnnealing + "", mode, lambda + ""};
                            }
                        }
                    }
                }
            }
            writeToFileForTuning(bestAccuracy, bestParams);
        }
    }

    private static void writeToFileForTuning(double accuracy) {
        try {
            FileWriter fw = new FileWriter("" + gram + originalLr + negSize + iter + batchSize + n + a + lrAnnealing + mode + lambda + ".txt");

            fw.write("gram = " + gram + "\n");
            fw.write("lr = " + originalLr + "\n");
            fw.write("negSize = " + negSize + "\n");
            fw.write("iter = " + iter + "\n");
            fw.write("batchSize = " + batchSize + "\n");
            fw.write("n = " + n + "\n");
            fw.write("a = " + a + "\n");
            fw.write("lrAnnealing = " + lrAnnealing + "\n");
            fw.write("mode = " + mode + "\n");
            fw.write("lambda = " + lambda + "\n");
            fw.write("accuracy=" + accuracy + "\n");

            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void writeToFileForTuning(double accuracy, String[] bestParams) {
        try {
            FileWriter fw = new FileWriter("" + bestParams[0] + bestParams[1] + bestParams[2] + bestParams[3] + bestParams[4] + bestParams[5] + bestParams[6] + bestParams[7] + bestParams[8] + bestParams[9] + "best.txt");

            fw.write("gram = " + bestParams[0] + "\n");
            fw.write("lr = " + bestParams[1] + "\n");
            fw.write("negSize = " + bestParams[2] + "\n");
            fw.write("iter = " + bestParams[3] + "\n");
            fw.write("batchSize = " + bestParams[4] + "\n");
            fw.write("n = " + bestParams[5] + "\n");
            fw.write("a = " + bestParams[6] + "\n");
            fw.write("lrAnnealing = " + bestParams[7] + "\n");
            fw.write("mode = " + bestParams[8] + "\n");
            fw.write("lambda = " + bestParams[9] + "\n");
            fw.write("accuracy=" + accuracy + "\n");

            fw.write("best accuracy\n");

            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double learnEmbeddingsAndTest(List<Document> trainDocs, List<Document> testDocs, List<Document> allDocs, List<Document> docList) {
        double accuracy = 0;
        originalLr = lr;
        Dataset.initSum();
        System.out.println("Initializing network");
        initNet(allDocs);

        for (int epoch = 0; epoch < iter; epoch++) {
            int startEpoch = (int) System.currentTimeMillis();
            System.out.printf("%d::\n", epoch);
            if (lrAnnealing) {
                lr = originalLr * (1 - (epoch * 1.0f / iter));
                System.out.printf("lr = %f\n", lr);
            }
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
                    //one iteration of SGD:
                    Arrays.fill(temp, 0);
                    backprop(WP[pi], WV[ids[l]], 1, temp);
                    for (int i = 0; i < negSize; i++) {
                        backprop(WP[pi], WV[Dataset.getRandomWordId()], 0, temp);
                    }
                    if (mode.equals("cosinesimilarity") || mode.equals("dotproduct")) {
                        for (int i = 0; i < n; i++) {
                            WP[pi][i] += temp[i];
                        }
                    } else if (mode.equals("l2rdotproduct")) {
                        for (int i = 0; i < n; i++) {
                            WP[pi][i] += temp[i] - lr * lambda * WP[pi][i];
                        }
                    }
                }
            }
        }

        private void backprop(double[] h, double[] v, double t, double[] temp) {
            if (h == null || v == null) {
                return;
            }
            if (mode.equals("cosinesimilarity")) {
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
            } else if (mode.equals("dotproduct") || mode.equals("l2rdotproduct")) {
                double h_dot_v = 0;
                for (int i = 0; i < n; i++) {
                    h_dot_v += h[i] * v[i];
                }
                double y = 1.0 / (1 + exp(-h_dot_v));
                for (int i = 0; i < n; i++) {
                    temp[i] += -(y - t) * v[i] * lr;
                    if (mode.equals("dotproduct")) {
                        v[i] += -(y - t) * h[i] * lr;
                    } else if (mode.equals("l2rdotproduct")) {
                        v[i] += -(y - t) * h[i] * lr - lr * lambda * v[i];
                    }
                }
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
