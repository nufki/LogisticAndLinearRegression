package ch.innuvation.ui;

import ch.innuvation.linearregression.LinearRegression;

import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;

/**
 * Panel to visualize linear regression predictions as a continuous surface
 */
public class RegressionSurfacePanel extends JPanel {
    private static final int WIDTH            = 800;
    private static final int HEIGHT           = 800;
    private static final int MARGIN           = 60;  // margin for axes
    private static final int PLOT_WIDTH       = WIDTH - 2 * MARGIN;
    private static final int PLOT_HEIGHT      = HEIGHT - 2 * MARGIN;
    private static final int GRID_RESOLUTION  = 200; // points per dimension

    private final LinearRegression model;
    private final double[][]       trainX;
    private final double[][]       trainY;

    public RegressionSurfacePanel(LinearRegression model, double[][] trainX, double[][] trainY) {
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

        // Draw the prediction surface as a heatmap
        drawPredictionSurface(g2);

        // Draw training points on top
        drawTrainingPoints(g2);

        // Draw axes
        drawAxes(g2);

        // Draw color scale legend
        drawColorScale(g2);
    }

    private void drawPredictionSurface(Graphics2D g2) {
        double minCoord = -0.30;
        double maxCoord = 1.30;
        double range = maxCoord - minCoord;

        for (int i = 0; i < GRID_RESOLUTION; i++) {
            for (int j = 0; j < GRID_RESOLUTION; j++) {
                // Map from pixel coordinates to data coordinates
                double x = minCoord + (i / (double) GRID_RESOLUTION) * range;
                double y = minCoord + (j / (double) GRID_RESOLUTION) * range;

                double[] input = {x, y};
                double[] prediction = model.predict(input);
                double value = prediction[0];

                // Clamp value to [0, 1] for color mapping
                value = Math.max(0.0, Math.min(1.0, value));

                // Map to color (blue=low, green=mid, red=high)
                Color c = valueToColor(value);
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
            double value = trainY[i][0];

            // Map data coordinates to pixel coordinates
            int px = MARGIN + (int) (((x - minCoord) / range) * PLOT_WIDTH);
            int py = MARGIN + (int) (((maxCoord - y) / range) * PLOT_HEIGHT); // flip Y

            // Draw outer circle (black border)
            g2.setColor(Color.BLACK);
            g2.fill(new Ellipse2D.Double(px - 8, py - 8, 16, 16));

            // Draw inner circle (colored by actual value)
            g2.setColor(valueToColor(value));
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

    private void drawColorScale(Graphics2D g2) {
        int scaleX = WIDTH - 80;
        int scaleY = MARGIN + 20;
        int scaleWidth = 40;
        int scaleHeight = 200;

        // Draw gradient bar
        for (int i = 0; i < scaleHeight; i++) {
            double value = 1.0 - (i / (double) scaleHeight); // top=1.0, bottom=0.0
            g2.setColor(valueToColor(value));
            g2.fillRect(scaleX, scaleY + i, scaleWidth, 1);
        }

        // Draw border
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(2));
        g2.drawRect(scaleX, scaleY, scaleWidth, scaleHeight);

        // Draw labels
        g2.setFont(new Font("Arial", Font.BOLD, 12));
        g2.drawString("1.0", scaleX + scaleWidth + 5, scaleY + 5);
        g2.drawString("0.5", scaleX + scaleWidth + 5, scaleY + scaleHeight / 2 + 5);
        g2.drawString("0.0", scaleX + scaleWidth + 5, scaleY + scaleHeight + 5);
    }

    /**
     * Map a value in [0,1] to a color gradient
     * Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
     */
    private Color valueToColor(double value) {
        // Clamp to [0,1]
        value = Math.max(0.0, Math.min(1.0, value));

        if (value < 0.25) {
            // Blue to Cyan
            double t = value / 0.25;
            return new Color(0, (int) (255 * t), 255);
        } else if (value < 0.5) {
            // Cyan to Green
            double t = (value - 0.25) / 0.25;
            return new Color(0, 255, (int) (255 * (1 - t)));
        } else if (value < 0.75) {
            // Green to Yellow
            double t = (value - 0.5) / 0.25;
            return new Color((int) (255 * t), 255, 0);
        } else {
            // Yellow to Red
            double t = (value - 0.75) / 0.25;
            return new Color(255, (int) (255 * (1 - t)), 0);
        }
    }
}