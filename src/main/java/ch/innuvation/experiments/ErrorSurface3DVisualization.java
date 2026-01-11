package ch.innuvation.experiments;

import ch.innuvation.linearregression.SimpleLinearRegression;
import ch.innuvation.ui.ErrorSurface3DPanel;

import javax.swing.*;

public class ErrorSurface3DVisualization {

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("3D ERROR SURFACE VISUALIZATION");
        System.out.println("=".repeat(70));
        System.out.println();

        // Create dataset with ORTHOGONAL separation
        // Class 0: small x1, any x2
        // Class 1: large x1, any x2
        // This makes w1 >> w2, giving VISIBLE asymmetry
        double[][] X = {
                // Class 0: small x1 (x2 varies randomly)
                {0.1, 0.3}, {0.15, 0.7}, {0.2, 0.2}, {0.12, 0.8},
                {0.18, 0.4}, {0.14, 0.6}, {0.16, 0.5}, {0.11, 0.9},
                {0.13, 0.1}, {0.19, 0.7}, {0.17, 0.3}, {0.15, 0.6},

                // Class 1: large x1 (x2 varies randomly)
                {0.8, 0.3}, {0.85, 0.7}, {0.9, 0.2}, {0.82, 0.8},
                {0.88, 0.4}, {0.84, 0.6}, {0.86, 0.5}, {0.81, 0.9},
                {0.83, 0.1}, {0.89, 0.7}, {0.87, 0.3}, {0.85, 0.6},
        };

        double[] y = {
                // Class 0 -> 0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // Class 1 -> 1
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        };

        System.out.println("Dataset characteristics:");
        System.out.println("  Class 0: x1 ≈ 0.15 (small), x2 = random [0.1-0.9]");
        System.out.println("  Class 1: x1 ≈ 0.85 (large), x2 = random [0.1-0.9]");
        System.out.println("  Separation depends MAINLY on x1, less on x2");
        System.out.println("  This creates anisotropic parabola (stretched in one direction)");
        System.out.println();

        // Train with gradient descent
        SimpleLinearRegression model = new SimpleLinearRegression(0.5, 500);
        model.train(X, y);

        System.out.println();
        System.out.println("=".repeat(70));
        System.out.println("The 3D surface shows:");
        System.out.println("  - X-axis: w₁ (weight for feature x₁)");
        System.out.println("  - Y-axis: w₂ (weight for feature x₂)");
        System.out.println("  - Z-axis: MSE (Mean Squared Error)");
        System.out.println();
        System.out.println("The BLACK LINE shows the path gradient descent took");
        System.out.println("from the random starting point (BLACK DOT) to the");
        System.out.println("optimal solution (RED DOT) at the bottom of the bowl.");
        System.out.println();
        System.out.println("You can DRAG the visualization to rotate it!");
        System.out.println("=".repeat(70));

        // Visualize the error surface
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("3D Error Surface: Gradient Descent Path");
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

            ErrorSurface3DPanel panel = new ErrorSurface3DPanel(
                    X, y,
                    model.getW0(),
                    model.getOptimizationPath(),
                    model.getErrorHistory()
            );

            frame.setContentPane(panel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}