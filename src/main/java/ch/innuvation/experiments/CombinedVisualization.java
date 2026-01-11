package ch.innuvation.experiments;

import ch.innuvation.linearregression.SimpleLinearRegression;
import ch.innuvation.ui.BinaryBoundaryPanel;
import ch.innuvation.ui.ErrorSurface3DPanel;

import javax.swing.*;
import java.awt.*;

public class CombinedVisualization {

    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("COMBINED VISUALIZATION: 3D ERROR SURFACE + 2D DECISION BOUNDARY");
        System.out.println("=".repeat(70));
        System.out.println();

        // Create dataset with ORTHOGONAL separation
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

        double[] yRegression = {
                // Class 0 -> 0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                // Class 1 -> 1
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        };

        // Convert to one-hot for BinaryBoundaryPanel
        double[][] yOneHot = new double[yRegression.length][2];
        for (int i = 0; i < yRegression.length; i++) {
            if (yRegression[i] == 0) {
                yOneHot[i][0] = 1;
                yOneHot[i][1] = 0;
            } else {
                yOneHot[i][0] = 0;
                yOneHot[i][1] = 1;
            }
        }

        System.out.println("Dataset characteristics:");
        System.out.println("  Class 0: x1 ≈ 0.15 (small), x2 = random [0.1-0.9]");
        System.out.println("  Class 1: x1 ≈ 0.85 (large), x2 = random [0.1-0.9]");
        System.out.println("  Separation depends MAINLY on x1, less on x2");
        System.out.println();

        // Train with gradient descent
        SimpleLinearRegression model = new SimpleLinearRegression(0.5, 500);
        model.train(X, yRegression);

        System.out.println();
        System.out.println("=".repeat(70));
        System.out.println("LEFT: 3D Error Surface - shows how MSE changes with weights");
        System.out.println("RIGHT: 2D Decision Boundary - shows the classification result");
        System.out.println("=".repeat(70));

        // Create wrapper for BinaryBoundaryPanel compatibility
        SimpleLinearRegressionWrapper wrapper = new SimpleLinearRegressionWrapper(model);

        // Visualize both side by side
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Combined Visualization: Error Surface + Decision Boundary");
            frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);

            // Create split pane with both visualizations
            JSplitPane splitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);

            // Left: 3D Error Surface
            ErrorSurface3DPanel errorPanel = new ErrorSurface3DPanel(
                    X, yRegression,
                    model.getW0(),
                    model.getOptimizationPath(),
                    model.getErrorHistory()
            );

            // Right: 2D Decision Boundary
            BinaryBoundaryPanel boundaryPanel = new BinaryBoundaryPanel(wrapper, X, yOneHot);

            splitPane.setLeftComponent(errorPanel);
            splitPane.setRightComponent(boundaryPanel);
            splitPane.setDividerLocation(700);  // Match panel width

            frame.setContentPane(splitPane);
            frame.setSize(1450, 750);  // Adjusted for smaller panels (700x2 + margins)
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }
}