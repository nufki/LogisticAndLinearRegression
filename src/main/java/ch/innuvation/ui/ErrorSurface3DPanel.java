package ch.innuvation.ui;

import ch.innuvation.linearregression.SimpleLinearRegression;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.util.List;

/**
 * 3D visualization of error surface for linear regression
 * Shows MSE as a function of w1 and w2
 */
public class ErrorSurface3DPanel extends JPanel {
    private static final int WIDTH  = 700;  // Reduced from 900
    private static final int HEIGHT = 700;  // Reduced from 900

    private final double[][] X;
    private final double[]   y;
    private final double     w0;  // Fixed bias
    private final List<double[]> path;
    private final List<Double>   errorHistory;

    // Rotation angles for 3D view - optimized to see bowl shape
    private double rotationX = 30;  // degrees - looking down from above
    private double rotationZ = -45; // degrees - diagonal view to see both axes

    // Weight ranges for visualization
    private double w1Min = -2.0;
    private double w1Max = 2.0;
    private double w2Min = -2.0;
    private double w2Max = 2.0;

    // Mouse drag support
    private int lastMouseX;
    private int lastMouseY;

    public ErrorSurface3DPanel(double[][] X, double[] y, double w0,
                               List<double[]> path, List<Double> errorHistory) {
        this.X = X;
        this.y = y;
        this.w0 = w0;
        this.path = path;
        this.errorHistory = errorHistory;

        // Center the weight range around the optimal solution
        if (path != null && !path.isEmpty()) {
            double[] optimalWeights = path.get(path.size() - 1);
            double w1Opt = optimalWeights[0];
            double w2Opt = optimalWeights[1];

            // Use SMALLER range for better curvature visibility
            double range = 0.5;  // Smaller range = more visible curvature
            w1Min = w1Opt - range;
            w1Max = w1Opt + range;
            w2Min = w2Opt - range;
            w2Max = w2Opt + range;

            System.out.println("Optimal weights: w1=" + w1Opt + ", w2=" + w2Opt);
            System.out.println("Visualization range: w1=[" + w1Min + ", " + w1Max + "], w2=[" + w2Min + ", " + w2Max + "]");
        }

        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        setBackground(Color.WHITE);

        // Add mouse listeners for rotation
        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                lastMouseX = e.getX();
                lastMouseY = e.getY();
            }
        });

        addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                int dx = e.getX() - lastMouseX;
                int dy = e.getY() - lastMouseY;

                rotationZ += dx * 0.5;
                rotationX -= dy * 0.5;

                // Clamp rotationX
                rotationX = Math.max(-89, Math.min(89, rotationX));

                lastMouseX = e.getX();
                lastMouseY = e.getY();
                repaint();
            }
        });
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        // Draw title
        g2.setFont(new Font("Arial", Font.BOLD, 18));
        g2.setColor(Color.BLACK);
        g2.drawString("3D Error Surface: log(MSE) vs (w₁, w₂)", 20, 30);

        g2.setFont(new Font("Arial", Font.PLAIN, 12));
        g2.drawString("Drag to rotate | Black line: Gradient Descent Path | Log scale reveals parabola shape", 20, 50);

        // Draw 3D surface
        draw3DSurface(g2);

        // Draw legend
        drawLegend(g2);
    }

    private void draw3DSurface(Graphics2D g2) {
        int centerX = WIDTH / 2;
        int centerY = HEIGHT / 2 + 300;
        int scale = 100;

        // Create grid of error values with HIGHER resolution
        int gridSize = 60;  // Increased from 40 for smoother surface
        double[][] errorGrid = new double[gridSize][gridSize];
        double[][] logErrorGrid = new double[gridSize][gridSize];
        double maxError = 0.0;
        double maxLogError = 0.0;

        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                double w1 = w1Min + (w1Max - w1Min) * i / (gridSize - 1);
                double w2 = w2Min + (w2Max - w2Min) * j / (gridSize - 1);
                errorGrid[i][j] = SimpleLinearRegression.computeMSE(X, y, w0, w1, w2);

                // Apply logarithmic scaling to compress large errors
                logErrorGrid[i][j] = Math.log(1 + errorGrid[i][j]);

                maxError = Math.max(maxError, errorGrid[i][j]);
                maxLogError = Math.max(maxLogError, logErrorGrid[i][j]);
            }
        }

        // Draw surface as wireframe with color
        for (int i = 0; i < gridSize - 1; i++) {
            for (int j = 0; j < gridSize - 1; j++) {
                double w1_1 = w1Min + (w1Max - w1Min) * i / (gridSize - 1);
                double w2_1 = w2Min + (w2Max - w2Min) * j / (gridSize - 1);
                double logErr_1 = logErrorGrid[i][j];

                double w1_2 = w1Min + (w1Max - w1Min) * (i + 1) / (gridSize - 1);
                double w2_2 = w2Min + (w2Max - w2Min) * j / (gridSize - 1);
                double logErr_2 = logErrorGrid[i + 1][j];

                double w1_3 = w1Min + (w1Max - w1Min) * i / (gridSize - 1);
                double w2_3 = w2Min + (w2Max - w2Min) * (j + 1) / (gridSize - 1);
                double logErr_3 = logErrorGrid[i][j + 1];

                // Project to 2D using log-scaled error for Z coordinate
                Point p1 = project3D(w1_1, w2_1, logErr_1 / maxLogError, centerX, centerY, scale);
                Point p2 = project3D(w1_2, w2_2, logErr_2 / maxLogError, centerX, centerY, scale);
                Point p3 = project3D(w1_3, w2_3, logErr_3 / maxLogError, centerX, centerY, scale);

                // Color based on original error (not log) for intuitive coloring
                Color color = getErrorColor(errorGrid[i][j] / maxError);
                g2.setColor(color);
                g2.setStroke(new BasicStroke(1));
                g2.drawLine(p1.x, p1.y, p2.x, p2.y);
                g2.drawLine(p1.x, p1.y, p3.x, p3.y);
            }
        }

        // Draw gradient descent path
        if (path != null && path.size() > 1) {
            g2.setStroke(new BasicStroke(3));

            for (int i = 0; i < path.size() - 1; i++) {
                double[] point1 = path.get(i);
                double[] point2 = path.get(i + 1);

                // Use log-scaled error for Z coordinate
                double logErr1 = Math.log(1 + errorHistory.get(i)) / maxLogError;
                double logErr2 = Math.log(1 + errorHistory.get(i + 1)) / maxLogError;

                Point p1 = project3D(point1[0], point1[1], logErr1, centerX, centerY, scale);
                Point p2 = project3D(point2[0], point2[1], logErr2, centerX, centerY, scale);

                g2.setColor(Color.BLACK);
                g2.drawLine(p1.x, p1.y, p2.x, p2.y);
            }

            // Draw start point (black dot)
            double[] startPoint = path.get(0);
            double logStartErr = Math.log(1 + errorHistory.get(0)) / maxLogError;
            Point pStart = project3D(startPoint[0], startPoint[1], logStartErr, centerX, centerY, scale);
            g2.setColor(Color.BLACK);
            g2.fillOval(pStart.x - 6, pStart.y - 6, 12, 12);

            // Draw end point (red dot)
            double[] endPoint = path.get(path.size() - 1);
            double logEndErr = Math.log(1 + errorHistory.get(errorHistory.size() - 1)) / maxLogError;
            Point pEnd = project3D(endPoint[0], endPoint[1], logEndErr, centerX, centerY, scale);
            g2.setColor(Color.RED);
            g2.fillOval(pEnd.x - 6, pEnd.y - 6, 12, 12);
            g2.setColor(Color.BLACK);
            g2.drawOval(pEnd.x - 6, pEnd.y - 6, 12, 12);
        }

        // Draw axes
        drawAxes(g2, centerX, centerY, scale);
    }

    private Point project3D(double x, double y, double z, int centerX, int centerY, int scale) {
        // Normalize coordinates to [-1, 1]
        double xNorm = 2 * (x - w1Min) / (w1Max - w1Min) - 1;
        double yNorm = 2 * (y - w2Min) / (w2Max - w2Min) - 1;
        double zNorm = z * 4.0;  // Increased from 3.0 to make edges steeper

        // Apply rotations
        double radX = Math.toRadians(rotationX);
        double radZ = Math.toRadians(rotationZ);

        // Rotate around Z axis
        double x1 = xNorm * Math.cos(radZ) - yNorm * Math.sin(radZ);
        double y1 = xNorm * Math.sin(radZ) + yNorm * Math.cos(radZ);
        double z1 = zNorm;

        // Rotate around X axis
        double y2 = y1 * Math.cos(radX) - z1 * Math.sin(radX);
        double z2 = y1 * Math.sin(radX) + z1 * Math.cos(radX);

        // Project to 2D (simple orthographic projection)
        int screenX = centerX + (int) (x1 * scale);
        int screenY = centerY - (int) (y2 * scale);

        return new Point(screenX, screenY);
    }

    private Color getErrorColor(double normalizedError) {
        // Green (low error) -> Yellow -> Red (high error)
        if (normalizedError < 0.5) {
            // Green to Yellow
            float ratio = (float) (normalizedError * 2);
            return new Color(
                    (int) (100 + 155 * ratio),
                    (int) (200 + 55 * ratio),
                    100,
                    180
            );
        } else {
            // Yellow to Red
            float ratio = (float) ((normalizedError - 0.5) * 2);
            return new Color(
                    255,
                    (int) (255 - 155 * ratio),
                    (int) (100 - 100 * ratio),
                    180
            );
        }
    }

    private void drawAxes(Graphics2D g2, int centerX, int centerY, int scale) {
        g2.setColor(Color.BLUE);
        g2.setStroke(new BasicStroke(2));

        // Use center of weight range as origin, not (0,0)
        double w1Center = (w1Min + w1Max) / 2;
        double w2Center = (w2Min + w2Max) / 2;

        // w1 axis
        Point origin = project3D(w1Center, w2Center, 0, centerX, centerY, scale);
        Point w1Axis = project3D(w1Max, w2Center, 0, centerX, centerY, scale);
        g2.drawLine(origin.x, origin.y, w1Axis.x, w1Axis.y);
        g2.drawString("w₁", w1Axis.x + 10, w1Axis.y);

        // w2 axis
        Point w2Axis = project3D(w1Center, w2Max, 0, centerX, centerY, scale);
        g2.drawLine(origin.x, origin.y, w2Axis.x, w2Axis.y);
        g2.drawString("w₂", w2Axis.x + 10, w2Axis.y);

        // Error axis (log scale)
        g2.setColor(Color.RED);
        Point errAxis = project3D(w1Center, w2Center, 1.0, centerX, centerY, scale);
        g2.drawLine(origin.x, origin.y, errAxis.x, errAxis.y);
        g2.drawString("log(MSE)", errAxis.x + 10, errAxis.y);
    }

    private void drawLegend(Graphics2D g2) {
        int legendX = WIDTH - 200;
        int legendY = HEIGHT - 150;

        g2.setColor(Color.WHITE);
        g2.fillRect(legendX, legendY, 180, 130);
        g2.setColor(Color.BLACK);
        g2.setStroke(new BasicStroke(2));
        g2.drawRect(legendX, legendY, 180, 130);

        g2.setFont(new Font("Arial", Font.BOLD, 12));
        g2.drawString("Legend", legendX + 10, legendY + 20);

        g2.setFont(new Font("Arial", Font.PLAIN, 11));

        // Start point
        g2.setColor(Color.BLACK);
        g2.fillOval(legendX + 10, legendY + 30, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawString("Start (random)", legendX + 30, legendY + 39);

        // End point
        g2.setColor(Color.RED);
        g2.fillOval(legendX + 10, legendY + 50, 10, 10);
        g2.setColor(Color.BLACK);
        g2.drawOval(legendX + 10, legendY + 50, 10, 10);
        g2.drawString("End (optimum)", legendX + 30, legendY + 59);

        // Path
        g2.setStroke(new BasicStroke(3));
        g2.drawLine(legendX + 10, legendY + 70, legendX + 20, legendY + 70);
        g2.setFont(new Font("Arial", Font.PLAIN, 11));
        g2.drawString("GD Path", legendX + 30, legendY + 74);

        // Color scale
        g2.setFont(new Font("Arial", Font.PLAIN, 10));
        g2.drawString("Error:", legendX + 10, legendY + 95);

        for (int i = 0; i < 100; i++) {
            g2.setColor(getErrorColor(i / 100.0));
            g2.fillRect(legendX + 10 + i, legendY + 100, 1, 10);
        }
        g2.setColor(Color.BLACK);
        g2.drawString("Low", legendX + 10, legendY + 125);
        g2.drawString("High", legendX + 140, legendY + 125);
    }
}