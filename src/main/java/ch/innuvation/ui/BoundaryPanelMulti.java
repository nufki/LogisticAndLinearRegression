package ch.innuvation.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;

/**
 * Panel to visualize multi-class decision boundaries for any classifier
 * Works with RandomForest, LogisticRegression, or any model with a predict(double[]) method
 */
public class BoundaryPanelMulti extends JPanel {
    private static final int WIDTH            = 800;
    private static final int HEIGHT           = 800;
    private static final int MARGIN           = 60;  // margin for axes
    private static final int PLOT_WIDTH       = WIDTH - 2 * MARGIN;
    private static final int PLOT_HEIGHT      = HEIGHT - 2 * MARGIN;
    private static final int GRID_RESOLUTION  = 200; // points per dimension

    private final Object     model;      // Can be RandomForest or LogisticRegression
    private final double[][] trainX;
    private final double[][] trainY;
    private final int        numClasses;

    public BoundaryPanelMulti(Object model, double[][] trainX, double[][] trainY, int numClasses) {
        this.model = model;
        this.trainX = trainX;
        this.trainY = trainY;
        this.numClasses = numClasses;
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(Color.WHITE);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw decision boundary regions
        drawDecisionRegions(g2);

        // Draw training points on top
        drawTrainingPoints(g2);

        // Draw axes
        drawAxes(g2);

        // Draw legend
        drawLegend(g2);
    }

    private void drawDecisionRegions(Graphics2D g2) {
        double step = 1.0 / GRID_RESOLUTION;
        double minCoord = -0.30;
        double maxCoord = 1.30;
        double range = maxCoord - minCoord;

        for (int i = 0; i < GRID_RESOLUTION; i++) {
            for (int j = 0; j < GRID_RESOLUTION; j++) {
                // Map from pixel coordinates to data coordinates
                double x = minCoord + (i / (double) GRID_RESOLUTION) * range;
                double y = minCoord + (j / (double) GRID_RESOLUTION) * range;

                double[] input = {x, y};
                double[] prediction = predict(input);
                int predictedClass = argMax(prediction);

                // Map class to color with transparency
                Color c = classToColor(predictedClass, numClasses);
                g2.setColor(c);

                int px = MARGIN + i * PLOT_WIDTH / GRID_RESOLUTION;
                int py = MARGIN + (GRID_RESOLUTION - 1 - j) * PLOT_HEIGHT / GRID_RESOLUTION; // flip Y
                int size = (int) Math.ceil(PLOT_WIDTH / (double) GRID_RESOLUTION) + 1;

                g2.fillRect(px, py, size, size);
            }
        }
    }

    private void drawTrainingPoints(Graphics2D g2) {
        double minCoord = -0.30;
        double maxCoord = 1.30;
        double range = maxCoord - minCoord;

        for (int i = 0; i < trainX.length; i++) {
            double x = trainX[i][0];
            double y = trainX[i][1];
            int trueClass = argMax(trainY[i]);

            // Map data coordinates to pixel coordinates
            int px = MARGIN + (int) (((x - minCoord) / range) * PLOT_WIDTH);
            int py = MARGIN + (int) (((maxCoord - y) / range) * PLOT_HEIGHT); // flip Y

            // Draw outer circle (black border for visibility)
            g2.setColor(Color.BLACK);
            g2.fill(new Ellipse2D.Double(px - 8, py - 8, 16, 16));

            // Draw inner circle (colored by true class - fully saturated)
            Color classColor = getClassColorSolid(trueClass, numClasses);
            g2.setColor(classColor);
            g2.fill(new Ellipse2D.Double(px - 6, py - 6, 12, 12));
        }
    }

    private void drawAxes(Graphics2D g2) {
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(2));

        // Draw axes box
        g2.drawRect(MARGIN, MARGIN, PLOT_WIDTH, PLOT_HEIGHT);

        // Draw tick marks and labels
        g2.setFont(new Font("Arial", Font.PLAIN, 12));
        FontMetrics fm = g2.getFontMetrics();

        double minCoord = -0.30;
        double maxCoord = 1.30;
        double range = maxCoord - minCoord;

        // X-axis ticks and labels
        double[] xTicks = {-0.30, 0.02, 0.34, 0.66, 0.98, 1.30};
        for (double tick : xTicks) {
            int px = MARGIN + (int) (((tick - minCoord) / range) * PLOT_WIDTH);
            g2.drawLine(px, MARGIN + PLOT_HEIGHT, px, MARGIN + PLOT_HEIGHT + 5);

            String label = String.format("%.2f", tick);
            int labelWidth = fm.stringWidth(label);
            g2.drawString(label, px - labelWidth / 2, MARGIN + PLOT_HEIGHT + 20);
        }

        // X-axis label
        String xLabel = "x[0]";
        int xLabelWidth = fm.stringWidth(xLabel);
        g2.drawString(xLabel, MARGIN + PLOT_WIDTH / 2 - xLabelWidth / 2, HEIGHT - 10);

        // Y-axis ticks and labels
        double[] yTicks = {-0.30, 0.02, 0.34, 0.66, 0.98, 1.30};
        for (double tick : yTicks) {
            int py = MARGIN + (int) (((maxCoord - tick) / range) * PLOT_HEIGHT);
            g2.drawLine(MARGIN - 5, py, MARGIN, py);

            String label = String.format("%.2f", tick);
            int labelWidth = fm.stringWidth(label);
            g2.drawString(label, MARGIN - labelWidth - 10, py + fm.getAscent() / 2);
        }

        // Y-axis label (rotated)
        g2.rotate(-Math.PI / 2);
        String yLabel = "x[1]";
        int yLabelWidth = fm.stringWidth(yLabel);
        g2.drawString(yLabel, -(MARGIN + PLOT_HEIGHT / 2 + yLabelWidth / 2), 20);
        g2.rotate(Math.PI / 2);
    }

    private void drawLegend(Graphics2D g2) {
        int legendX = MARGIN + 20;
        int legendY = MARGIN + 20;
        int legendWidth = 120;
        int legendHeight = 30 + numClasses * 25;

        // Draw legend box
        g2.setColor(Color.WHITE);
        g2.fillRect(legendX, legendY, legendWidth, legendHeight);
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(2));
        g2.drawRect(legendX, legendY, legendWidth, legendHeight);

        // Draw legend title
        g2.setFont(new Font("Arial", Font.BOLD, 14));
        g2.drawString("Classes", legendX + 10, legendY + 20);

        // Draw class entries
        g2.setFont(new Font("Arial", Font.PLAIN, 12));
        for (int i = 0; i < numClasses; i++) {
            int entryY = legendY + 35 + i * 25;

            // Draw colored circle
            Color classColor = getClassColorSolid(i, numClasses);
            g2.setColor(classColor);
            g2.fill(new Ellipse2D.Double(legendX + 10, entryY - 6, 12, 12));

            g2.setColor(Color.BLACK);
            g2.draw(new Ellipse2D.Double(legendX + 10, entryY - 6, 12, 12));

            // Draw class label
            g2.drawString("Class " + i, legendX + 30, entryY + 4);
        }
    }

    /**
     * Call predict on the model using reflection to support different types
     */
    private double[] predict(double[] input) {
        try {
            java.lang.reflect.Method predictMethod = model.getClass().getMethod("predict", double[].class);
            return (double[]) predictMethod.invoke(model, input);
        } catch (Exception e) {
            throw new RuntimeException("Model must have a predict(double[]) method", e);
        }
    }

    /**
     * Map class index to a distinct color (transparent for decision regions)
     */
    private Color classToColor(int classIdx, int totalClasses) {
        // Use transparent colors for decision regions
        int alpha = 100; // transparency for background regions

        if (totalClasses == 2) {
            // Binary classification: blue vs red
            return classIdx == 0
                    ? new Color(100, 150, 255, alpha)  // light blue
                    : new Color(255, 100, 100, alpha); // light red
        } else if (totalClasses == 3) {
            // 3-class: red, blue, green (matching the image)
            switch (classIdx) {
                case 0:  return new Color(255, 120, 120, alpha);  // red
                case 1:  return new Color(120, 150, 255, alpha);  // blue
                case 2:  return new Color(120, 255, 120, alpha);  // green
                default: return new Color(128, 128, 128, alpha);  // gray fallback
            }
        } else {
            // Multi-class: use HSB color wheel
            float hue = classIdx / (float) totalClasses;
            Color base = Color.getHSBColor(hue, 0.7f, 0.9f);
            return new Color(base.getRed(), base.getGreen(), base.getBlue(), alpha);
        }
    }

    /**
     * Get solid (opaque) color for training points and legend
     */
    private Color getClassColorSolid(int classIdx, int totalClasses) {
        if (totalClasses == 2) {
            return classIdx == 0
                    ? new Color(50, 100, 200)   // darker blue
                    : new Color(200, 50, 50);   // darker red
        } else if (totalClasses == 3) {
            // 3-class: red, blue, green (solid versions)
            switch (classIdx) {
                case 0:  return new Color(200, 50, 50);    // red
                case 1:  return new Color(50, 100, 200);   // blue
                case 2:  return new Color(50, 150, 50);    // green
                default: return new Color(100, 100, 100);  // gray fallback
            }
        } else {
            float hue = classIdx / (float) totalClasses;
            return Color.getHSBColor(hue, 0.8f, 0.8f);
        }
    }

    private int argMax(double[] v) {
        int best = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[best]) best = i;
        }
        return best;
    }
}