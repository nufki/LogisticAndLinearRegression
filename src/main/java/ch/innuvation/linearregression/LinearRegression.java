package ch.innuvation.linearregression;

/**
 * Linear Regression with two training methods:
 * 1. Gradient Descent (iterative optimization)
 * 2. Closed-form solution using Normal Equation: β = (X^T X)^-1 X^T y
 */
public class LinearRegression {
    private double[][] weights;  // [numFeatures + 1][numOutputs] (includes bias)
    private double learningRate;
    private int maxIterations;
    private boolean trained = false;
    private boolean useClosedForm;

    /**
     * Create a Linear Regression model with gradient descent
     * @param learningRate Learning rate for gradient descent
     * @param maxIterations Maximum number of training iterations
     */
    public LinearRegression(double learningRate, int maxIterations) {
        this.learningRate = learningRate;
        this.maxIterations = maxIterations;
        this.useClosedForm = false;
    }

    /**
     * Create a Linear Regression model with closed-form solution
     * @param useClosedForm Set to true to use Normal Equation: β = (X^T X)^-1 X^T y
     */
    public LinearRegression(boolean useClosedForm) {
        this.useClosedForm = useClosedForm;
        this.learningRate = 0.0;  // Not used for closed-form
        this.maxIterations = 0;   // Not used for closed-form
    }

    /**
     * Train the model using either gradient descent or closed-form solution
     * @param X Training features [numSamples][numFeatures]
     * @param Y Training targets [numSamples][numOutputs]
     */
    public void train(double[][] X, double[][] Y) {
        if (X.length == 0 || Y.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        if (X.length != Y.length) {
            throw new IllegalArgumentException("X and Y must have same number of samples");
        }

        if (useClosedForm) {
            trainClosedForm(X, Y);
        } else {
            trainGradientDescent(X, Y);
        }

        trained = true;
    }

    /**
     * Train using closed-form solution: β = (X^T X)^-1 X^T y
     * This computes the optimal weights directly without iteration
     */
    private void trainClosedForm(double[][] X, double[][] Y) {
        System.out.println("Training using closed-form solution (Normal Equation)...");
        long startTime = System.currentTimeMillis();

        int numSamples = X.length;
        int numFeatures = X[0].length;
        int numOutputs = Y[0].length;

        // Add bias column to X: X_augmented = [1, x1, x2, ...]
        double[][] X_augmented = new double[numSamples][numFeatures + 1];
        for (int i = 0; i < numSamples; i++) {
            X_augmented[i][0] = 1.0; // bias term
            System.arraycopy(X[i], 0, X_augmented[i], 1, numFeatures);
        }

        // Compute X^T
        double[][] X_T = transpose(X_augmented);

        // Compute X^T X
        double[][] XTX = matrixMultiply(X_T, X_augmented);

        // Compute (X^T X)^-1
        double[][] XTX_inv = invert(XTX);

        // Compute X^T y
        double[][] XTY = matrixMultiply(X_T, Y);

        // Compute β = (X^T X)^-1 X^T y
        weights = matrixMultiply(XTX_inv, XTY);

        long endTime = System.currentTimeMillis();
        System.out.println("Closed-form solution computed in " + (endTime - startTime) + " ms");

        // Compute final MSE for reporting
        double totalLoss = 0.0;
        for (int i = 0; i < numSamples; i++) {
            double[] prediction = predictSingle(X[i]);
            for (int j = 0; j < numOutputs; j++) {
                double error = prediction[j] - Y[i][j];
                totalLoss += error * error;
            }
        }
        double mse = totalLoss / (numSamples * numOutputs);
        System.out.printf("Final MSE = %.6f%n", mse);
    }

    /**
     * Train using gradient descent (original implementation)
     */
    private void trainGradientDescent(double[][] X, double[][] Y) {
        System.out.println("Training using gradient descent...");

        int numSamples = X.length;
        int numFeatures = X[0].length;
        int numOutputs = Y[0].length;

        // Initialize weights randomly (small values)
        weights = new double[numFeatures + 1][numOutputs];
        java.util.Random rand = new java.util.Random(42);
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (rand.nextDouble() - 0.5) * 0.01;
            }
        }

        // Gradient descent
        for (int iter = 0; iter < maxIterations; iter++) {
            // Compute gradients
            double[][] gradients = new double[numFeatures + 1][numOutputs];

            double totalLoss = 0.0;
            for (int i = 0; i < numSamples; i++) {
                // Forward pass
                double[] prediction = predictSingle(X[i]);

                // Compute error
                double[] error = new double[numOutputs];
                for (int j = 0; j < numOutputs; j++) {
                    error[j] = prediction[j] - Y[i][j];
                    totalLoss += error[j] * error[j];
                }

                // Accumulate gradients
                // Bias gradient
                for (int j = 0; j < numOutputs; j++) {
                    gradients[0][j] += error[j];
                }

                // Feature gradients
                for (int f = 0; f < numFeatures; f++) {
                    for (int j = 0; j < numOutputs; j++) {
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
                double mse = totalLoss / (numSamples * numOutputs);
                System.out.printf("Iteration %d: MSE = %.6f%n", iter, mse);
            }
        }
    }

    /**
     * Predict output for a single input
     * @param x Input features [numFeatures]
     * @return Predicted output [numOutputs]
     */
    public double[] predict(double[] x) {
        if (!trained) {
            throw new IllegalStateException("Model must be trained before prediction");
        }
        return predictSingle(x);
    }

    /**
     * Internal prediction (works before training for gradient computation)
     *
     * Single output: ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
     * Multiple outputs: ŷⱼ = w₀ⱼ + w₁ⱼx₁ + w₂ⱼx₂ + ... + wₙⱼxₙ
     * or in vector form: ŷ = w₀ + W^T x
     *
     * weight matrix structure:
     * ```
     * weights[0]   = [w₀₀, w₀₁, w₀₂, ...]  // biases
     * weights[1]   = [w₁₀, w₁₁, w₁₂, ...]  // weights for x₁
     * weights[2]   = [w₂₀, w₂₁, w₂₂, ...]  // weights for x₂
     *
     * output[j] = weights[0][j] + Σᵢ (weights[i+1][j] * x[i])
     */
    private double[] predictSingle(double[] x) {
        int numOutputs = weights[0].length;
        double[] output = new double[numOutputs];

        // Initialize with bias
        for (int j = 0; j < numOutputs; j++) {
            output[j] = weights[0][j];
        }

        // Add weighted features
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < numOutputs; j++) {
                output[j] += weights[i + 1][j] * x[i];
            }
        }

        return output;
    }

    // ==================== Matrix Operations ====================

    /**
     * Transpose a matrix
     */
    private double[][] transpose(double[][] A) {
        int rows = A.length;
        int cols = A[0].length;
        double[][] result = new double[cols][rows];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j][i] = A[i][j];
            }
        }
        return result;
    }

    /**
     * Multiply two matrices: C = A * B
     */
    private double[][] matrixMultiply(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;

        double[][] result = new double[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return result;
    }

    /**
     * Invert a matrix using Gauss-Jordan elimination
     * Note: This is numerically stable for small matrices but may fail for ill-conditioned matrices
     */
    private double[][] invert(double[][] A) {
        int n = A.length;

        // Create augmented matrix [A | I]
        double[][] augmented = new double[n][2 * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented[i][j] = A[i][j];
            }
            augmented[i][i + n] = 1.0;
        }

        // Forward elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int maxRow = i;
            for (int k = i + 1; k < n; k++) {
                if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                    maxRow = k;
                }
            }

            // Swap rows
            double[] temp = augmented[i];
            augmented[i] = augmented[maxRow];
            augmented[maxRow] = temp;

            // Check for singular matrix
            if (Math.abs(augmented[i][i]) < 1e-10) {
                throw new RuntimeException("Matrix is singular or nearly singular");
            }

            // Make diagonal 1
            double pivot = augmented[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[i][j] /= pivot;
            }

            // Eliminate column
            for (int k = 0; k < n; k++) {
                if (k != i) {
                    double factor = augmented[k][i];
                    for (int j = 0; j < 2 * n; j++) {
                        augmented[k][j] -= factor * augmented[i][j];
                    }
                }
            }
        }

        // Extract inverse from augmented matrix
        double[][] inverse = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse[i][j] = augmented[i][j + n];
            }
        }

        return inverse;
    }

    /**
     * Get the learned weights (for inspection/debugging)
     * @return weights[numFeatures + 1][numOutputs]
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