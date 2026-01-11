package ch.innuvation.logisticregression;

/**
 * Logistic Regression for binary and multi-class classification
 * Uses softmax activation for multi-class and sigmoid for binary classification
 * Trained with gradient descent
 */
public class LogisticRegression {
    private double[][] weights;  // [numFeatures + 1][numClasses] (includes bias)
    private double learningRate;
    private int maxIterations;
    private boolean trained = false;
    private int numClasses;

    /**
     * Create a Logistic Regression model
     * @param learningRate Learning rate for gradient descent
     * @param maxIterations Maximum number of training iterations
     */
    public LogisticRegression(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
    }

    /**
     * Train the model using gradient descent with cross-entropy loss
     * @param X Training features [numSamples][numFeatures]
     * @param Y Training labels (one-hot encoded) [numSamples][numClasses]
     */
    public void train(double[][] X, double[][] Y) {
        if (X.length == 0 || Y.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        if (X.length != Y.length) {
            throw new IllegalArgumentException("X and Y must have same number of samples");
        }

        int numSamples = X.length;
        int numFeatures = X[0].length;
        numClasses = Y[0].length;

        // Initialize weights randomly (small values)
        weights = new double[numFeatures + 1][numClasses];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (rand.nextDouble() - 0.5) * 0.01;
            }
        }

        // Gradient descent
        for (int iter = 0; iter < maxIterations; iter++) {
            // Compute gradients
            double[][] gradients = new double[numFeatures + 1][numClasses];

            double totalLoss = 0.0;
            for (int i = 0; i < numSamples; i++) {
                // Forward pass (get probabilities)
                double[] probabilities = predictProbabilities(X[i]);

                // Compute cross-entropy loss
                for (int j = 0; j < numClasses; j++) {
                    if (Y[i][j] == 1.0) {
                        totalLoss -= Math.log(probabilities[j] + 1e-15); // add small epsilon to avoid log(0)
                    }
                }

                // Compute gradients (derivative of cross-entropy with softmax)
                // Gradient is simply (predicted - actual)
                double[] error = new double[numClasses];
                for (int j = 0; j < numClasses; j++) {
                    error[j] = probabilities[j] - Y[i][j];
                }

                // Accumulate gradients
                // Bias gradient
                for (int j = 0; j < numClasses; j++) {
                    gradients[0][j] += error[j];
                }

                // Feature gradients
                for (int f = 0; f < numFeatures; f++) {
                    for (int j = 0; j < numClasses; j++) {
                        gradients[f + 1][j] += error[j] * X[i][f];
                    }
                }
            }

            // Update weights
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] -= learningRate * gradients[i][j] / numSamples;
                }
            }

            // Print progress every 200 iterations
            if (iter % 200 == 0 || iter == maxIterations - 1) {
                double avgLoss = totalLoss / numSamples;
                System.out.printf("Iteration %d: Cross-Entropy Loss = %.6f%n", iter, avgLoss);
            }
        }

        trained = true;
    }

    /**
     * Predict class probabilities for a single input
     * @param x Input features [numFeatures]
     * @return Class probabilities [numClasses] (sum to 1.0)
     */
    public double[] predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        return predictProbabilities(x);
    }

    /**
     * Predict the most likely class
     * @param x Input features [numFeatures]
     * @return Predicted class index
     */
    public int predictClass(double[] x) {
        double[] probs = predict(x);
        int bestClass = 0;
        for (int i = 1; i < probs.length; i++) {
            if (probs[i] > probs[bestClass]) {
                bestClass = i;
            }
        }
        return bestClass;
    }

    /**
     * Internal method to compute probabilities (works before training)
     */
    private double[] predictProbabilities(double[] x) {
        // Compute logits (linear combination)
        double[] logits = new double[numClasses];

        // Initialize with bias
        for (int j = 0; j < numClasses; j++) {
            logits[j] = weights[0][j];
        }

        // Add weighted features
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < numClasses; j++) {
                logits[j] += weights[i + 1][j] * x[i];
            }
        }

        // Apply softmax (or sigmoid for binary case)
        return softmax(logits);
    }

    /**
     * Softmax activation function
     * Converts logits to probabilities that sum to 1.0
     */
    private double[] softmax(double[] logits) {
        // Find max for numerical stability
        double max = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }

        // Compute exp(logit - max) and sum
        double[] probs = new double[logits.length];
        double sum = 0.0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = Math.exp(logits[i] - max);
            sum += probs[i];
        }

        // Normalize
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }

        return probs;
    }

    /**
     * Get the learned weights (for inspection/debugging)
     * @return weights[numFeatures + 1][numClasses]
     */
    public double[][] getWeights() {
        return weights;
    }

    /**
     * Check if model has been trained
     */
    public boolean isTrained() {
        return trained;
    }
}