package ch.innuvation.ui;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Line2D;

/**
 * Panel to visualize binary classification with explicit decision boundary line
 */
public class BinaryBoundaryPanel extends JPanel {
    private static final int WIDTH            = 700;  // Reduced from 800
    private static final int HEIGHT           = 700;  // Reduced from 800
    private static final int MARGIN           = 60;
    private static final int PLOT_WIDTH       = WIDTH - 2 * MARGIN;
    private static final int PLOT_HEIGHT      = HEIGHT - 2 * MARGIN;
    private static final int GRID_RESOLUTION  = 200;

    private final Object     model;
    private final double[][] trainX;
    private final double[][] trainY;

    public BinaryBoundaryPanel(Object model, double[][] trainX, double[][] trainY) {
        this.model = model;
        this.trainX = trainX;
        this.trainY = trainY;
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(Color.WHITE);
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw decision regions
        drawDecisionRegions(g2);

        // Draw decision boundary line
        drawDecisionBoundaryLine(g2);

        // Draw training points
        drawTrainingPoints(g2);

        // Draw axes
        drawAxes(g2);

        // Draw legend
        drawLegend(g2);
    }

    private void drawDecisionRegions(Graphics2D g2) {
        double minCoord = -0.30;
        double maxCoord = 1.30;
        double range = maxCoord - minCoord;

        for (int i = 0; i < GRID_RESOLUTION; i++) {
            for (int j = 0; j < GRID_RESOLUTION; j++) {
                double x = minCoord + (i / (double) GRID_RESOLUTION) * range;
                double y = minCoord + (j / (double) GRID_RESOLUTION) * range;

                double[] input = {x, y};
                double[] prediction = predict(input);
                int predictedClass = argMax(prediction);

                Color c = predictedClass == 0
                        ? new Color(200, 100, 100, 80)  // Red for class 0
                        : new Color(100, 150, 255, 80); // Blue for class 1
                g2.setColor(c);

                int px = MARGIN + i * PLOT_WIDTH / GRID_RESOLUTION;
                int py = MARGIN + (GRID_RESOLUTION - 1 - j) * PLOT_HEIGHT / GRID_RESOLUTION;
                int size = (int) Math.ceil(PLOT_WIDTH / (double) GRID_RESOLUTION) + 1;

                g2.fillRect(px, py, size, size);
            }
        }
    }

    private void drawDecisionBoundaryLine(Graphics2D g2) {
        // Extract weights from model
        double[][] weights = getWeights();
        if (weights == null) return;

        // For binary classification: w0 + w1*x1 + w2*x2 = 0
        // where w = weights_class0 - weights_class1
        double w0 = weights[0][0] - weights[0][1];  // bias
        double w1 = weights[1][0] - weights[1][1];  // x1 coefficient
        double w2 = weights[2][0] - weights[2][1];  // x2 coefficient

        // Draw the line w0 + w1*x1 + w2*x2 = 0
        // Rearranged: x2 = -(w0 + w1*x1) / w2

        if (Math.abs(w2) > 1e-6) {
            // Calculate line endpoints in data coordinates
            double minCoord = -0.30;
            double maxCoord = 1.30;

            double x1_start = minCoord;
            double x2_start = -(w0 + w1 * x1_start) / w2;

            double x1_end = maxCoord;
            double x2_end = -(w0 + w1 * x1_end) / w2;

            // Convert to pixel coordinates
            double range = maxCoord - minCoord;
            int px1 = MARGIN + (int) (((x1_start - minCoord) / range) * PLOT_WIDTH);
            int py1 = MARGIN + (int) (((maxCoord - x2_start) / range) * PLOT_HEIGHT);
            int px2 = MARGIN + (int) (((x1_end - minCoord) / range) * PLOT_WIDTH);
            int py2 = MARGIN + (int) (((maxCoord - x2_end) / range) * PLOT_HEIGHT);

            // Draw the decision boundary as a thick black line
            g2.setColor(Color.BLACK);
            g2.setStroke(new BasicStroke(3));
            g2.draw(new Line2D.Double(px1, py1, px2, py2));

            // Add label
            int labelX = (px1 + px2) / 2;
            int labelY = (py1 + py2) / 2 - 10;
            g2.setFont(new Font("Arial", Font.BOLD, 14));
            g2.setColor(Color.BLACK);

            // Draw label background
            String label = "Decision Boundary";
            FontMetrics fm = g2.getFontMetrics();
            int labelWidth = fm.stringWidth(label);
            g2.setColor(new Color(255, 255, 255, 200));
            g2.fillRect(labelX - labelWidth/2 - 5, labelY - fm.getAscent(), labelWidth + 10, fm.getHeight());

            g2.setColor(Color.BLACK);
            g2.drawString(label, labelX - labelWidth/2, labelY);
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

            int px = MARGIN + (int) (((x - minCoord) / range) * PLOT_WIDTH);
            int py = MARGIN + (int) (((maxCoord - y) / range) * PLOT_HEIGHT);

            // Draw outer circle (black border)
            g2.setColor(Color.BLACK);
            g2.fill(new Ellipse2D.Double(px - 8, py - 8, 16, 16));

            // Draw inner circle
            Color classColor = trueClass == 0
                    ? new Color(200, 50, 50)   // Red for class 0
                    : new Color(50, 100, 200); // Blue for class 1
            g2.setColor(classColor);
            g2.fill(new Ellipse2D.Double(px - 6, py - 6, 12, 12));
        }
    }

    private void drawAxes(Graphics2D g2) {
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(2));

        // Draw axes box
        g2.drawRect(MARGIN, MARGIN, PLOT_WIDTH, PLOT_HEIGHT);

        g2.setFont(new Font("Arial", Font.PLAIN, 12));
        FontMetrics fm = g2.getFontMetrics();

        double minCoord = -0.30;
        double maxCoord = 1.30;
        double range = maxCoord - minCoord;

        // X-axis ticks
        double[] xTicks = {-0.30, 0.02, 0.34, 0.66, 0.98, 1.30};
        for (double tick : xTicks) {
            int px = MARGIN + (int) (((tick - minCoord) / range) * PLOT_WIDTH);
            g2.drawLine(px, MARGIN + PLOT_HEIGHT, px, MARGIN + PLOT_HEIGHT + 5);

            String label = String.format("%.2f", tick);
            int labelWidth = fm.stringWidth(label);
            g2.drawString(label, px - labelWidth / 2, MARGIN + PLOT_HEIGHT + 20);
        }

        // X-axis label
        String xLabel = "x₁";
        int xLabelWidth = fm.stringWidth(xLabel);
        g2.drawString(xLabel, MARGIN + PLOT_WIDTH / 2 - xLabelWidth / 2, HEIGHT - 10);

        // Y-axis ticks
        double[] yTicks = {-0.30, 0.02, 0.34, 0.66, 0.98, 1.30};
        for (double tick : yTicks) {
            int py = MARGIN + (int) (((maxCoord - tick) / range) * PLOT_HEIGHT);
            g2.drawLine(MARGIN - 5, py, MARGIN, py);

            String label = String.format("%.2f", tick);
            int labelWidth = fm.stringWidth(label);
            g2.drawString(label, MARGIN - labelWidth - 10, py + fm.getAscent() / 2);
        }

        // Y-axis label
        g2.rotate(-Math.PI / 2);
        String yLabel = "x₂";
        int yLabelWidth = fm.stringWidth(yLabel);
        g2.drawString(yLabel, -(MARGIN + PLOT_HEIGHT / 2 + yLabelWidth / 2), 20);
        g2.rotate(Math.PI / 2);
    }

    private void drawLegend(Graphics2D g2) {
        int legendX = MARGIN + 20;
        int legendY = MARGIN + 20;
        int legendWidth = 120;
        int legendHeight = 85;

        // Draw legend box
        g2.setColor(Color.WHITE);
        g2.fillRect(legendX, legendY, legendWidth, legendHeight);
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(2));
        g2.drawRect(legendX, legendY, legendWidth, legendHeight);

        // Title
        g2.setFont(new Font("Arial", Font.BOLD, 14));
        g2.drawString("Classes", legendX + 10, legendY + 20);

        // Class 0
        g2.setFont(new Font("Arial", Font.PLAIN, 12));
        g2.setColor(new Color(200, 50, 50));
        g2.fill(new Ellipse2D.Double(legendX + 10, legendY + 30, 12, 12));
        g2.setColor(Color.BLACK);
        g2.draw(new Ellipse2D.Double(legendX + 10, legendY + 30, 12, 12));
        g2.drawString("Class 0", legendX + 30, legendY + 40);

        // Class 1
        g2.setColor(new Color(50, 100, 200));
        g2.fill(new Ellipse2D.Double(legendX + 10, legendY + 50, 12, 12));
        g2.setColor(Color.BLACK);
        g2.draw(new Ellipse2D.Double(legendX + 10, legendY + 50, 12, 12));
        g2.drawString("Class 1", legendX + 30, legendY + 60);
    }

    private double[] predict(double[] input) {
        try {
            java.lang.reflect.Method predictMethod = model.getClass().getMethod("predict", double[].class);
            return (double[]) predictMethod.invoke(model, input);
        } catch (Exception e) {
            throw new RuntimeException("Model must have a predict(double[]) method", e);
        }
    }

    private double[][] getWeights() {
        try {
            java.lang.reflect.Method getWeightsMethod = model.getClass().getMethod("getWeights");
            return (double[][]) getWeightsMethod.invoke(model);
        } catch (Exception e) {
            return null;
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