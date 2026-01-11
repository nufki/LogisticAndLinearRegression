package ch.innuvation.experiments;

import ch.innuvation.linearregression.SimpleLinearRegression;

/**
 * Wrapper to make SimpleLinearRegression compatible with BinaryBoundaryPanel
 */
public class SimpleLinearRegressionWrapper {
    private final SimpleLinearRegression model;

    public SimpleLinearRegressionWrapper(SimpleLinearRegression model) {
        this.model = model;
    }

    public double[] predict(double[] x) {
        double prediction = model.predict(x);
        // Clamp prediction to [0, 1] range
        prediction = Math.max(0.0, Math.min(1.0, prediction));
        // Convert single output to binary probabilities
        // Class 0 probability = 1 - prediction, Class 1 probability = prediction
        return new double[]{1.0 - prediction, prediction};
    }

    public double[][] getWeights() {
        // BinaryBoundaryPanel draws boundary where: (w0[0]-w0[1]) + (w1[0]-w1[1])*x1 + (w2[0]-w2[1])*x2 = 0
        // For linear regression: prediction = w0 + w1*x1 + w2*x2
        // Decision boundary is where prediction = 0.5
        // So: w0 + w1*x1 + w2*x2 = 0.5
        // Rearranged: (w0 - 0.5) + w1*x1 + w2*x2 = 0

        // To match BinaryBoundaryPanel's format:
        // (w0[0]-w0[1]) = w0 - 0.5  =>  w0[0] = w0 - 0.5, w0[1] = 0
        // (w1[0]-w1[1]) = w1        =>  w1[0] = w1, w1[1] = 0
        // (w2[0]-w2[1]) = w2        =>  w2[0] = w2, w2[1] = 0

        double threshold = 0.5;
        return new double[][]{
                {model.getW0() - threshold, 0},  // bias
                {model.getW1(), 0},              // w1
                {model.getW2(), 0}               // w2
        };
    }
}