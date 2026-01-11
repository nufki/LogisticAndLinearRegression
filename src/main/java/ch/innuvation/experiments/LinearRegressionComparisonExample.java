package ch.innuvation.experiments;

import ch.innuvation.linearregression.LinearRegression;
import ch.innuvation.ui.BoundaryPanelMulti;

import javax.swing.*;
import java.util.Arrays;

public class LinearRegressionComparisonExample {

    public static void main(String[] args) {
        // Training data: 3-class classification problem
        double[][] X = {
                // Class 0 region (near 0,0)
                {0.05, 0.05}, {0.10, 0.00}, {0.00, 0.15}, {0.12, 0.08}, {0.20, 0.10},

                // Class 1 region (near 1,0)
                {0.90, 0.05}, {1.00, 0.10}, {0.85, 0.00}, {0.95, 0.15}, {0.80, 0.10},

                // Class 2 region (near 0,1)
                {0.05, 0.90}, {0.10, 1.00}, {0.00, 0.85}, {0.15, 0.95}, {0.10, 0.80},
        };

        double[][] Y = {
                // Class 0 -> [1,0,0]
                {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},

                // Class 1 -> [0,1,0]
                {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},

                // Class 2 -> [0,0,1]
                {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
        };

        System.out.println("=".repeat(60));
        System.out.println("METHOD 1: GRADIENT DESCENT");
        System.out.println("=".repeat(60));

        LinearRegression lrGD = new LinearRegression(0.1, 2000);
        long startGD = System.currentTimeMillis();
        lrGD.train(X, Y);
        long endGD = System.currentTimeMillis();
        System.out.println("Total time: " + (endGD - startGD) + " ms");
        System.out.println();

        System.out.println("=".repeat(60));
        System.out.println("METHOD 2: CLOSED-FORM SOLUTION (Normal Equation)");
        System.out.println("=".repeat(60));

        LinearRegression lrCF = new LinearRegression(true);
        long startCF = System.currentTimeMillis();
        lrCF.train(X, Y);
        long endCF = System.currentTimeMillis();
        System.out.println("Total time: " + (endCF - startCF) + " ms");
        System.out.println();

        System.out.println("=".repeat(60));
        System.out.println("COMPARISON OF PREDICTIONS");
        System.out.println("=".repeat(60));

        double[][] testPoints = {
                {0.08, 0.10},  // class 0-ish
                {0.92, 0.08},  // class 1-ish
                {0.05, 0.95},  // class 2-ish
                {0.50, 0.50},  // center
        };

        for (double[] point : testPoints) {
            System.out.println("\nTest point: " + Arrays.toString(point));

            double[] predGD = lrGD.predict(point);
            int classGD = argMax(predGD);
            System.out.println("  Gradient Descent:  " + Arrays.toString(round3(predGD)) + " -> Class " + classGD);

            double[] predCF = lrCF.predict(point);
            int classCF = argMax(predCF);
            System.out.println("  Closed-form:       " + Arrays.toString(round3(predCF)) + " -> Class " + classCF);
        }

        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("KEY DIFFERENCES:");
        System.out.println("=".repeat(60));
        System.out.println("Gradient Descent:");
        System.out.println("  - Iterative optimization (2000 iterations)");
        System.out.println("  - Requires tuning learning rate");
        System.out.println("  - May not reach exact optimum");
        System.out.println("  - Scales well to large datasets");
        System.out.println();
        System.out.println("Closed-form (Normal Equation):");
        System.out.println("  - Direct computation of optimal solution");
        System.out.println("  - No hyperparameters to tune");
        System.out.println("  - Finds exact optimum (no approximation)");
        System.out.println("  - O(n³) complexity - slow for large features");
        System.out.println("  - May fail if X^T X is singular");
        System.out.println();

        // Visualize both (they should be nearly identical)
        SwingUtilities.invokeLater(() -> {
            JFrame f1 = new JFrame("Gradient Descent - Decision Boundary");
            f1.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f1.setContentPane(new BoundaryPanelMulti(lrGD, X, Y, 3));
            f1.pack();
            f1.setLocation(50, 50);
            f1.setVisible(true);

            JFrame f2 = new JFrame("Closed-Form - Decision Boundary");
            f2.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f2.setContentPane(new BoundaryPanelMulti(lrCF, X, Y, 3));
            f2.pack();
            f2.setLocation(900, 50);
            f2.setVisible(true);
        });
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