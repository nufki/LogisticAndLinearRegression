package ch.innuvation.linearregression;

import java.util.ArrayList;
import java.util.List;

/**
 * Simple Linear Regression for 2D visualization
 * Tracks the gradient descent path for visualization
 */
public class SimpleLinearRegression {
    private double w0;  // bias
    private double w1;  // weight for x1
    private double w2;  // weight for x2
    private double learningRate;
    private int maxIterations;
    private boolean trained = false;

    // Store the path taken during gradient descent
    private List<double[]> optimizationPath;
    private List<Double> errorHistory;

    public SimpleLinearRegression(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.optimizationPath = new ArrayList<>();
        this.errorHistory = new ArrayList<>();
    }

    /**
     * Train using gradient descent for binary classification
     * Simplified to track only w1 and w2 (assuming w0 is fixed or learned separately)
     */
    public void train(double[][] X, double[] y) {
        int numSamples = X.length;

        // Initialize weights randomly
        w0 = (Math.random() - 0.5) * 0.1;
        w1 = (Math.random() - 0.5) * 0.1;
        w2 = (Math.random() - 0.5) * 0.1;

        System.out.println("Starting gradient descent from: w0=" + w0 + ", w1=" + w1 + ", w2=" + w2);

        // Store initial position
        optimizationPath.add(new double[]{w1, w2});
        errorHistory.add(computeMSE(X, y));

        // Gradient descent
        for (int iter = 0; iter < maxIterations; iter++) {
            double grad0 = 0.0;
            double grad1 = 0.0;
            double grad2 = 0.0;
            double totalError = 0.0;

            // Compute gradients
            for (int i = 0; i < numSamples; i++) {
                double prediction = w0 + w1 * X[i][0] + w2 * X[i][1];
                double error = prediction - y[i];

                grad0 += error;
                grad1 += error * X[i][0];
                grad2 += error * X[i][1];

                totalError += error * error;
            }

            // Average gradients
            grad0 /= numSamples;
            grad1 /= numSamples;
            grad2 /= numSamples;

            // Update weights
            w0 -= learningRate * grad0;
            w1 -= learningRate * grad1;
            w2 -= learningRate * grad2;

            // Store path (every 5 iterations to avoid too many points)
            if (iter % 5 == 0 || iter == maxIterations - 1) {
                optimizationPath.add(new double[]{w1, w2});
                errorHistory.add(totalError / numSamples);
            }

            if (iter % 100 == 0 || iter == maxIterations - 1) {
                double mse = totalError / numSamples;
                System.out.printf("Iteration %d: MSE = %.6f, w1=%.3f, w2=%.3f%n", iter, mse, w1, w2);
            }
        }

        trained = true;
        System.out.println("Final weights: w0=" + w0 + ", w1=" + w1 + ", w2=" + w2);
    }

    public double predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained first");
        }
        return w0 + w1 * x[0] + w2 * x[1];
    }

    /**
     * Compute MSE for given weights (for visualization)
     */
    public double computeMSE(double[][] X, double[] y) {
        double totalError = 0.0;
        for (int i = 0; i < X.length; i++) {
            double prediction = w0 + w1 * X[i][0] + w2 * X[i][1];
            double error = prediction - y[i];
            totalError += error * error;
        }
        return totalError / X.length;
    }

    /**
     * Compute MSE for arbitrary weights (for surface visualization)
     */
    public static double computeMSE(double[][] X, double[] y, double w0, double w1, double w2) {
        double totalError = 0.0;
        for (int i = 0; i < X.length; i++) {
            double prediction = w0 + w1 * X[i][0] + w2 * X[i][1];
            double error = prediction - y[i];
            totalError += error * error;
        }
        return totalError / X.length;
    }

    public List<double[]> getOptimizationPath() {
        return optimizationPath;
    }

    public List<Double> getErrorHistory() {
        return errorHistory;
    }

    public double getW0() { return w0; }
    public double getW1() { return w1; }
    public double getW2() { return w2; }
}