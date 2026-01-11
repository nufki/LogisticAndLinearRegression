package ch.innuvation.experiments;

import ch.innuvation.logisticregression.LogisticRegression;
import ch.innuvation.ui.BoundaryPanelMulti;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LogisticRegressionCircularExample {

    public static void main(String[] args) {
        // Create Logistic Regression model
        // Note: Logistic regression is linear, so it won't perfectly separate circular regions
        // This example demonstrates the linear decision boundaries
        LogisticRegression lr = new LogisticRegression(0.1, 3000);

        // Generate circular/spiral training data
        List<double[]> XList = new ArrayList<>();
        List<double[]> YList = new ArrayList<>();
        Random rand = new Random(42);

        // Class 0: Inner circle
        generateCircularData(XList, YList, 0.3, 0.3, 0.15, 30, 0, rand);

        // Class 1: Middle ring
        generateRingData(XList, YList, 0.3, 0.3, 0.25, 0.35, 40, 1, rand);

        // Class 2: Outer region (corners)
        generateCornerData(XList, YList, 25, 2, rand);

        double[][] X = XList.toArray(new double[0][]);
        double[][] Y = YList.toArray(new double[0][]);

        System.out.println("Training Logistic Regression with circular data...");
        System.out.println("Training samples: " + X.length);
        System.out.println("Note: Logistic regression creates LINEAR decision boundaries,");
        System.out.println("      so circular patterns won't be perfectly separated.");
        long startTime = System.currentTimeMillis();
        lr.train(X, Y);
        long endTime = System.currentTimeMillis();
        System.out.println("Training completed in " + (endTime - startTime) + " ms");
        System.out.println();

        // Visualize decision boundaries
        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame("Logistic Regression - Linear Boundaries (Circular Data)");
            f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            f.setContentPane(new BoundaryPanelMulti(lr, X, Y, 3));
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });

        // Test predictions
        System.out.println("=== Testing predictions ===");
        test(lr, new double[]{0.30, 0.30}); // center - class 0
        test(lr, new double[]{0.45, 0.30}); // middle ring - class 1
        test(lr, new double[]{0.10, 0.10}); // corner - class 2
        test(lr, new double[]{0.90, 0.90}); // corner - class 2
    }

    /**
     * Generate points in a circle
     */
    private static void generateCircularData(List<double[]> XList, List<double[]> YList,
                                             double cx, double cy, double radius,
                                             int count, int classIdx, Random rand) {
        for (int i = 0; i < count; i++) {
            double angle = rand.nextDouble() * 2 * Math.PI;
            double r = radius * Math.sqrt(rand.nextDouble()); // uniform distribution in circle
            double x = cx + r * Math.cos(angle);
            double y = cy + r * Math.sin(angle);

            XList.add(new double[]{x, y});
            YList.add(oneHot(classIdx, 3));
        }
    }

    /**
     * Generate points in a ring (annulus)
     */
    private static void generateRingData(List<double[]> XList, List<double[]> YList,
                                         double cx, double cy, double innerRadius,
                                         double outerRadius, int count, int classIdx, Random rand) {
        for (int i = 0; i < count; i++) {
            double angle = rand.nextDouble() * 2 * Math.PI;
            double r = innerRadius + (outerRadius - innerRadius) * Math.sqrt(rand.nextDouble());
            double x = cx + r * Math.cos(angle);
            double y = cy + r * Math.sin(angle);

            XList.add(new double[]{x, y});
            YList.add(oneHot(classIdx, 3));
        }
    }

    /**
     * Generate points in the corners
     */
    private static void generateCornerData(List<double[]> XList, List<double[]> YList,
                                           int countPerCorner, int classIdx, Random rand) {
        // Four corners
        double[][] corners = {
                {0.0, 0.0},   // bottom-left
                {1.0, 0.0},   // bottom-right
                {0.0, 1.0},   // top-left
                {1.0, 1.0}    // top-right
        };

        for (double[] corner : corners) {
            for (int i = 0; i < countPerCorner; i++) {
                double x = corner[0] + (rand.nextDouble() - 0.5) * 0.2;
                double y = corner[1] + (rand.nextDouble() - 0.5) * 0.2;
                x = Math.max(0.0, Math.min(1.0, x)); // clamp to [0,1]
                y = Math.max(0.0, Math.min(1.0, y));

                XList.add(new double[]{x, y});
                YList.add(oneHot(classIdx, 3));
            }
        }
    }

    private static double[] oneHot(int classIdx, int nClasses) {
        double[] y = new double[nClasses];
        y[classIdx] = 1.0;
        return y;
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