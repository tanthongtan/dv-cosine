package dv.cosine.java;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import static java.lang.Math.round;
import static java.lang.Math.pow;
import java.util.Random;

public class Dataset {

    private static final double power = 0.75;

    public static List<Integer> wordIdCounts = new ArrayList<Integer>();
    private static int[] wordIdSums;

    private static Random random = new Random();

    public static List<Document> getImdbDataset(int gram) {
        String filename = "alldata-id_p" + gram + "gram.txt";
        List<Document> allDocs = new ArrayList<Document>();
        Map<String, Integer> word2id = new HashMap<String, Integer>();
        String[] datasetSplit = {"train", "test", "extra", "extra"};
        int[] datasetSentiment = {1, 0, 1, 0, -1, -1, -1, -1};

        try (BufferedReader br = new BufferedReader(new FileReader(new File(filename)))) {
            String line;
            int i = 0;
            while ((line = br.readLine()) != null) {
                //String[] tokens = line.split("\\s+");
                String[] tokens = line.split(" ");
                List<String> words = Arrays.asList(tokens).subList(1, tokens.length);
                int[] wordIds = new int[words.size()];
                int j = 0;
                for (String word : words) {
                    if (word2id.get(word) == null) {
                        word2id.put(word, wordIdCounts.size());
                        wordIdCounts.add(0);
                    }
                    int index = word2id.get(word);
                    wordIdCounts.set(index, wordIdCounts.get(index) + 1);
                    wordIds[j] = index;
                    j++;
                }
                String instanceSplit = datasetSplit[(int) i / 25000];
                int instanceSentiment = datasetSentiment[(int) i / 12500];
                allDocs.add(new Document(wordIds, i, instanceSplit, instanceSentiment));
                i++;
                if (i % 10000 == 0) {
                    System.out.println(i);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return allDocs;
    }

    public static void initSum() {
        wordIdSums = new int[wordIdCounts.size()];
        wordIdSums[0] = (int) round(pow(wordIdCounts.get(0), power));
        for (int i = 1; i < wordIdCounts.size(); i++) {
            wordIdSums[i] = (int) round(pow(wordIdCounts.get(i), power) + wordIdSums[i - 1]);
        }
    }

    public static int getRandomWordId() {
        int i = random.nextInt(wordIdSums[wordIdSums.length - 1]) + 1;
        int l = 0, r = wordIdSums.length - 1;
        while (l != r) {
            int m = (l + r) / 2;
            if (i <= wordIdSums[m]) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;
    }
}
