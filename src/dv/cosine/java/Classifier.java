package dv.cosine.java;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import java.util.List;

public class Classifier {
    
    private Parameter parameter;
    private Model model;
    
    public Classifier(SolverType solver, double C, double eps) {
        parameter = new Parameter(solver, C, eps);
    }
    
    public void train(double[][] WP, List<Document> trainDocs) {
        Problem problem = new Problem();
        int numInstances = trainDocs.size();
        int numFeatures = WP[0].length;
        problem.l = numInstances;
        problem.n = numFeatures;
        
        FeatureNode[][] X_train = new FeatureNode[numInstances][numFeatures];
        double[] Y_train = new double[numInstances];
        for (int i=0; i<numInstances; i++){
            Document doc = trainDocs.get(i);
            Y_train[i] = doc.sentiment;
            for (int j=0; j<numFeatures; j++) {
                X_train[i][j] = new FeatureNode(j+1,WP[doc.tag][j]);
            }
        }
        problem.x = X_train;
        problem.y = Y_train;
        Linear.setDebugOutput(null);
        model = Linear.train(problem, parameter);
    }
    
    public double score(double[][] WP, List<Document> testDocs) {
        int numInstances = testDocs.size();
        int numFeatures = WP[0].length;
        
        int corrects = 0;
        FeatureNode[] X_test = new FeatureNode[numFeatures];
        for (int i=0; i<numInstances; i++){
            Document doc = testDocs.get(i);
            double Y_test = doc.sentiment;
            for (int j=0; j<numFeatures; j++) {
                X_test[j] = new FeatureNode(j+1,WP[doc.tag][j]);
            }
            double prediction = Linear.predict(model, X_test);
            if (Y_test == prediction) {
                corrects++;
            }
        }
        
        double accuracy = ((corrects+0.0)/numInstances)*100;
        System.out.println("Accuracy = "+ accuracy +"% ("+ corrects+"/"+numInstances+")");
        return accuracy;
    }
    
}
