package ch.innuvation.experiments;

import ch.innuvation.linearregression.LinearRegression;
import ch.innuvation.logisticregression.LogisticRegression;
import ch.innuvation.ui.BinaryBoundaryPanel;

import javax.swing.*;
import java.util.Arrays;

public class BinaryClassificationExample {

    public static void main(String[] args) {
        // Training data: 2-class problem (clearly linearly separable)
        double[][] X = {
                // Class 0 (bottom-left region)
                {0.1, 0.2},
                {0.2, 0.3},
                {0.15, 0.4},
                {0.3, 0.2},
                {0.25, 0.35},
                {0.1, 0.3},
                {0.2, 0.1},
                {0.35, 0.25},

                // Class 1 (top-right region)
                {0.7, 0.6},
                {0.8, 0.7},
                {0.75, 0.8},
                {0.9, 0.65},
                {0.85, 0.75},
                {0.7, 0.7},
                {0.8, 0.6},
                {0.9, 0.8},
        };

        // Binary labels: [1,0] for class 0, [0,1] for class 1
        double[][] Y = {
                // Class 0
                {1, 0}, {1, 0}, {1, 0}, {1, 0},
                {1, 0}, {1, 0}, {1, 0}, {1, 0},

                // Class 1
                {0, 1}, {0, 1}, {0, 1}, {0, 1},
                {0, 1}, {0, 1}, {0, 1}, {0, 1},
        };

        System.out.println("=".repeat(70));
        System.out.println("BINARY CLASSIFICATION: LINEAR vs LOGISTIC REGRESSION");
        System.out.println("=".repeat(70));
        System.out.println();

        // Train Linear Regression
        System.out.println("--- LINEAR REGRESSION (Closed-Form) ---");
        LinearRegression lr = new LinearRegression(true);
        lr.train(X, Y);
        System.out.println("Weights (bias, w1, w2): " + weightsToString(lr.getWeights()));
        System.out.println();

        // Train Logistic Regression
        System.out.println("--- LOGISTIC REGRESSION (Gradient Descent) ---");
        LogisticRegression logistic = new LogisticRegression(0.5, 2000);
        logistic.train(X, Y);
        System.out.println("Weights (bias, w1, w2): " + weightsToString(logistic.getWeights()));
        System.out.println();

        // Test predictions
        System.out.println("=".repeat(70));
        System.out.println("TEST PREDICTIONS");
        System.out.println("=".repeat(70));

        double[][] testPoints = {
                {0.2, 0.25},  // Should be class 0
                {0.5, 0.5},   // Boundary region
                {0.75, 0.7},  // Should be class 1
        };

        for (double[] point : testPoints) {
            System.out.println("\nTest point: " + Arrays.toString(point));

            double[] predLR = lr.predict(point);
            int classLR = argMax(predLR);
            System.out.println("  Linear Regression:   " + Arrays.toString(round3(predLR)) + " -> Class " + classLR);

            double[] predLogistic = logistic.predict(point);
            int classLogistic = logistic.predictClass(point);
            System.out.println("  Logistic Regression: " + Arrays.toString(round3(predLogistic)) + " -> Class " + classLogistic);
        }

        System.out.println();
        System.out.println("=".repeat(70));
        System.out.println("DECISION BOUNDARY EQUATION");
        System.out.println("=".repeat(70));
        System.out.println();
        System.out.println("For binary classification, the decision boundary is where:");
        System.out.println("  w₀ + w₁*x₁ + w₂*x₂ = 0");
        System.out.println();
        System.out.println("Rearranging: x₂ = -(w₀ + w₁*x₁) / w₂");
        System.out.println("This is the equation of a STRAIGHT LINE!");
        System.out.println();

        printDecisionBoundary("Linear Regression", lr.getWeights());
        printDecisionBoundary("Logistic Regression", logistic.getWeights());

        // Visualize both side-by-side
        SwingUtilities.invokeLater(() -> {
            JFrame f1 = new JFrame("Linear Regression - Binary Decision Boundary");
            f1.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f1.setContentPane(new BinaryBoundaryPanel(lr, X, Y));
            f1.pack();
            f1.setLocation(50, 50);
            f1.setVisible(true);

            JFrame f2 = new JFrame("Logistic Regression - Binary Decision Boundary");
            f2.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f2.setContentPane(new BinaryBoundaryPanel(logistic, X, Y));
            f2.pack();
            f2.setLocation(900, 50);
            f2.setVisible(true);
        });
    }

    private static void printDecisionBoundary(String name, double[][] weights) {
        double w0 = weights[0][0] - weights[0][1];  // bias difference
        double w1 = weights[1][0] - weights[1][1];  // x1 coefficient difference
        double w2 = weights[2][0] - weights[2][1];  // x2 coefficient difference

        System.out.println(name + ":");
        System.out.printf("  Decision boundary: %.3f + %.3f*x₁ + %.3f*x₂ = 0%n", w0, w1, w2);

        if (Math.abs(w2) > 1e-6) {
            double slope = -w1 / w2;
            double intercept = -w0 / w2;
            System.out.printf("  Or in y=mx+b form: x₂ = %.3f*x₁ + %.3f%n", slope, intercept);
        }
        System.out.println();
    }

    private static String weightsToString(double[][] weights) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < weights.length; i++) {
            sb.append(Arrays.toString(round3(weights[i])));
            if (i < weights.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }

    private static int argMax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }

    private static double[] round3(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Math.round(v[i] * 1000.0) / 1000.0;
        }
        return r;
    }
}