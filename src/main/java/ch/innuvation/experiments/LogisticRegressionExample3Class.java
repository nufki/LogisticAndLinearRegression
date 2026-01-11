package ch.innuvation.experiments;

import ch.innuvation.logisticregression.LogisticRegression;
import ch.innuvation.ui.BoundaryPanelMulti;

import javax.swing.*;
import java.util.Arrays;

public class LogisticRegressionExample3Class {

    public static void main(String[] args) {
        // Create Logistic Regression model
        // Higher learning rate and more iterations for better convergence
        LogisticRegression lr = new LogisticRegression(0.1, 2000);

        // Training data: N samples, each with 2 inputs
        double[][] X = {
                // Class 0 region (near 0,0)
                {0.05, 0.05},
                {0.10, 0.00},
                {0.00, 0.15},
                {0.12, 0.08},
                {0.20, 0.10},

                // Class 1 region (near 1,0)
                {0.90, 0.05},
                {1.00, 0.10},
                {0.85, 0.00},
                {0.95, 0.15},
                {0.80, 0.10},

                // Class 2 region (near 0,1)
                {0.05, 0.90},
                {0.10, 1.00},
                {0.00, 0.85},
                {0.15, 0.95},
                {0.10, 0.80},
        };

        // One-hot labels: N samples, each with 3 outputs
        double[][] Y = {
                // Class 0 -> [1,0,0]
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},
                {1, 0, 0},

                // Class 1 -> [0,1,0]
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},
                {0, 1, 0},

                // Class 2 -> [0,0,1]
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
                {0, 0, 1},
        };

        System.out.println("Training Logistic Regression model...");
        long startTime = System.currentTimeMillis();
        lr.train(X, Y);
        long endTime = System.currentTimeMillis();
        System.out.println("Training completed in " + (endTime - startTime) + " ms");
        System.out.println();

        // Visualize decision boundaries
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("Logistic Regression - 3-Class Decision Boundary");
            f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f.setContentPane(new BoundaryPanelMulti(lr, X, Y, 3));
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });

        // Test predictions on training points
        System.out.println("=== Testing on training data ===");
        test(lr, new double[]{0.08, 0.10}); // class 0-ish
        test(lr, new double[]{0.92, 0.08}); // class 1-ish
        test(lr, new double[]{0.05, 0.95}); // class 2-ish
        System.out.println();

        // Test on in-between points
        System.out.println("=== Testing on intermediate points ===");
        test(lr, new double[]{0.60, 0.10});
        test(lr, new double[]{0.10, 0.60});
        test(lr, new double[]{0.35, 0.35});
        test(lr, new double[]{0.50, 0.50});
    }

    private static void test(LogisticRegression lr, double[] x) {
        double[] probs = lr.predict(x);
        int predicted = lr.predictClass(x);

        System.out.println("x=" + Arrays.toString(x)
                + " -> probs=" + Arrays.toString(round3(probs))
                + " predictedClass=" + predicted);
    }

    private static double[] round3(double[] v) {
        double[] r = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            r[i] = Math.round(v[i] * 1000.0) / 1000.0;
        }
        return r;
    }
}