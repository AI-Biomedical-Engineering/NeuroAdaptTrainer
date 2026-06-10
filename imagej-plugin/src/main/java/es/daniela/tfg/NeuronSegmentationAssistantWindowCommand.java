package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.ImagePanel;
import ij.gui.Overlay;
import ij.gui.OvalRoi;
import ij.io.FileSaver;
import ij.process.ImageProcessor;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Neuron Segmentation Assistant Window")
public class NeuronSegmentationAssistantWindowCommand implements Command {

    private static final String PYTHON_EXE =
            "/Users/danielaerasocasas/tfg/venv/bin/python3";

    private static final String SCRIPT_PATH =
            "/Users/danielaerasocasas/Documents/gitHub/fiji-yolo-neuron-segmentation/yolo-inference/infer_one.py";

    private JFrame frame;
    private DefaultListModel<String> roiListModel;
    private JList<String> roiList;
    private JLabel imageLabel;
    private JLabel statusLabel;

    private JScrollPane imageScrollPane;
    private JLabel zoomLabel;

    private JButton zoomInButton;
    private JButton zoomOutButton;
    private JButton fitButton;
    private JButton actualSizeButton;

    private ImagePlus imageToDisplay;
    private double zoomFactor = 1.0;
    private boolean fitToPanel = true;

    private JButton deleteSelectedButton;
    private JButton addNeuronButton;
    private boolean addNeuronMode = false;

    private ImagePanel imagePanel;

    private JButton detectButton;
    private JButton correctButton;
    private JButton saveButton;
    private JButton retrainButton;

    private ImagePlus sourceImage;
    private ImagePlus detectedImage;
    private final List<NeuronDetection> detections = new ArrayList<>();

    @Override
    public void run() {
        SwingUtilities.invokeLater(this::createWindow);
    }

    private void createWindow() {
        sourceImage = getCurrentOrFirstImage();

        frame = new JFrame("Neuron Segmentation Assistant");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(1050, 720);
        frame.setLayout(new BorderLayout(8, 8));

        frame.add(createLeftPanel(), BorderLayout.WEST);
        frame.add(createCenterPanel(), BorderLayout.CENTER);
        frame.add(createRightPanel(), BorderLayout.EAST);
        frame.add(createBottomPanel(), BorderLayout.SOUTH);

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);

        if (sourceImage != null) {
            imageToDisplay = sourceImage;
            statusLabel.setText("Selected image: " + sourceImage.getTitle());
            updateDisplayedImage(false);
        } else {
            statusLabel.setText("No Fiji image selected.");
        }
    }

    private ImagePlus getCurrentOrFirstImage() {
        ImagePlus current = WindowManager.getCurrentImage();

        if (current != null) {
            return current;
        }

        int[] imageIds = WindowManager.getIDList();

        if (imageIds != null && imageIds.length > 0) {
            return WindowManager.getImage(imageIds[0]);
        }

        return null;
    }

    private JPanel createLeftPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(180, 0));
        panel.setBorder(BorderFactory.createTitledBorder("ROI list"));

        roiListModel = new DefaultListModel<>();
        roiList = new JList<>(roiListModel);
        roiList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        panel.add(new JScrollPane(roiList), BorderLayout.CENTER);

        return panel;
    }

    private JPanel createCenterPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Image"));

        imagePanel = new ImagePanel();
        imagePanel.setBackground(Color.DARK_GRAY);

        imageScrollPane = new JScrollPane(imagePanel);
        panel.add(imageScrollPane, BorderLayout.CENTER);

        JPanel zoomPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));

        zoomOutButton = new JButton("-");
        zoomInButton = new JButton("+");
        fitButton = new JButton("Fit");
        actualSizeButton = new JButton("100%");
        zoomLabel = new JLabel("Fit");

        zoomOutButton.addActionListener(e -> {
            fitToPanel = false;
            zoomFactor = Math.max(0.25, zoomFactor - 0.25);
            updateDisplayedImage(true);
        });

        zoomInButton.addActionListener(e -> {
            fitToPanel = false;
            zoomFactor = Math.min(5.0, zoomFactor + 0.25);
            updateDisplayedImage(true);
        });

        fitButton.addActionListener(e -> {
            fitToPanel = true;
            updateDisplayedImage(true);
        });

        actualSizeButton.addActionListener(e -> {
            fitToPanel = false;
            zoomFactor = 1.0;
            updateDisplayedImage(true);
        });

        zoomPanel.add(new JLabel("Zoom:"));
        zoomPanel.add(zoomOutButton);
        zoomPanel.add(zoomInButton);
        zoomPanel.add(fitButton);
        zoomPanel.add(actualSizeButton);
        zoomPanel.add(zoomLabel);

        panel.add(zoomPanel, BorderLayout.SOUTH);

        return panel;
    }

    private JPanel createRightPanel() {
        JPanel panel = new JPanel();
        panel.setPreferredSize(new Dimension(190, 0));
        panel.setLayout(new GridLayout(8, 1, 8, 8));
        panel.setBorder(BorderFactory.createTitledBorder("Actions"));

        detectButton = new JButton("1. Detect neurons");
        correctButton = new JButton("2. Correct detections");
        saveButton = new JButton("3. Save corrections");
        retrainButton = new JButton("4. Retrain model");
        deleteSelectedButton = new JButton("Delete selected ROI");
        addNeuronButton = new JButton("Add neuron mode");

        correctButton.setEnabled(false);
        saveButton.setEnabled(false);
        retrainButton.setEnabled(false);
        deleteSelectedButton.setEnabled(false);
        addNeuronButton.setEnabled(false);

        detectButton.addActionListener(e -> detectNeurons());
        correctButton.addActionListener(e -> enableCorrectionMode());
        saveButton.addActionListener(e -> saveCorrections());
        retrainButton.addActionListener(e -> retrainModel());
        deleteSelectedButton.addActionListener(e -> deleteSelectedDetection());

        addNeuronButton.addActionListener(e -> {
            addNeuronMode = !addNeuronMode;

            if (addNeuronMode) {
                addNeuronButton.setText("Adding neuron... click image");
                statusLabel.setText("Click on the image to add a missing neuron.");
            } else {
                addNeuronButton.setText("Add neuron mode");
                statusLabel.setText("Add neuron mode disabled.");
            }
        });

        panel.add(detectButton);
        panel.add(correctButton);
        panel.add(deleteSelectedButton);
        panel.add(addNeuronButton);
        panel.add(saveButton);
        panel.add(retrainButton);

        return panel;
    }

    private JPanel createBottomPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        statusLabel = new JLabel("Detected neurons: 0");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
        panel.add(statusLabel, BorderLayout.WEST);
        return panel;
    }

    private void detectNeurons() {
        ImagePlus currentImage = sourceImage != null ? sourceImage : getCurrentOrFirstImage();

        if (currentImage == null) {
            IJ.error("No image is open in Fiji.");
            return;
        }

        sourceImage = currentImage;
        IJ.log("Assistant selected source image: " + sourceImage.getTitle());

        detectButton.setEnabled(false);
        correctButton.setEnabled(false);
        saveButton.setEnabled(false);
        retrainButton.setEnabled(false);

        statusLabel.setText("Detecting neurons...");
        roiListModel.clear();
        detections.clear();

        new Thread(() -> {
            try {
                DetectionResult result = runPythonDetection(currentImage);
                detectedImage = result.image;

                loadDetectionsFromCsv();
                applyOverlayToDetectedImage();

                imageToDisplay = detectedImage;
                fitToPanel = true;
                updateDisplayedImage(true);

                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Detected neurons: " + detections.size());

                    correctButton.setEnabled(true);
                    saveButton.setEnabled(true);

                    IJ.showStatus("Detected neurons: " + detections.size());
                    IJ.log("Assistant detection completed. Count: " + detections.size());
                });

            } catch (Exception e) {
                IJ.handleException(e);

                SwingUtilities.invokeLater(() -> {
                    statusLabel.setText("Detection failed.");
                    detectButton.setEnabled(true);
                });
            }
        }).start();
    }

    private DetectionResult runPythonDetection(ImagePlus imp) throws Exception {
        String tmpDir = System.getProperty("java.io.tmpdir");

        File inputFile = new File(tmpDir, "imagej_input.png");
        File outputFile = new File(tmpDir, "imagej_output.png");
        File csvFile = new File(tmpDir, "imagej_output.csv");

        deleteIfExists(outputFile);
        deleteIfExists(csvFile);

        boolean saved = new FileSaver(imp).saveAsPng(inputFile.getAbsolutePath());

        if (!saved) {
            throw new RuntimeException("Could not save the input image.");
        }

        ProcessBuilder pb = new ProcessBuilder(
                PYTHON_EXE,
                "-u",
                SCRIPT_PATH,
                inputFile.getAbsolutePath(),
                outputFile.getAbsolutePath()
        );

        pb.redirectErrorStream(true);

        IJ.log("Running Python inference from assistant window...");
        IJ.log(String.join(" ", pb.command()));

        Process process = pb.start();

        int neuronCount = -1;

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {

            String line;

            while ((line = reader.readLine()) != null) {
                IJ.log(line);

                if (line.startsWith("Detected neurons")) {
                    neuronCount = parseNeuronCount(line);
                }
            }
        }

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Python process failed with exit code " + exitCode);
        }

        if (!outputFile.exists()) {
            throw new RuntimeException("The output image was not created.");
        }

        ImagePlus resultImage = IJ.openImage(outputFile.getAbsolutePath());

        if (resultImage == null) {
            throw new RuntimeException("The output image could not be opened.");
        }

        return new DetectionResult(resultImage, neuronCount);
    }

    private void loadDetectionsFromCsv() throws Exception {
        String tmpDir = System.getProperty("java.io.tmpdir");
        File csvFile = new File(tmpDir, "imagej_output.csv");

        if (!csvFile.exists()) {
            throw new RuntimeException("CSV not found: " + csvFile.getAbsolutePath());
        }

        detections.clear();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line = br.readLine(); // header
            int index = 1;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");

                if (parts.length < 3) {
                    IJ.log("Skipping malformed CSV line: " + line);
                    continue;
                }

                double cx = Double.parseDouble(parts[0]);
                double cy = Double.parseDouble(parts[1]);
                double radius = Double.parseDouble(parts[2]);

                detections.add(new NeuronDetection("AUTO_" + index, cx, cy, radius));
                index++;
            }
        }

        SwingUtilities.invokeLater(() -> {
            roiListModel.clear();

            for (NeuronDetection detection : detections) {
                roiListModel.addElement(detection.name);
            }
        });
    }

    private void applyOverlayToDetectedImage() {
        if (detectedImage == null) {
            return;
        }

        Overlay overlay = new Overlay();

        for (NeuronDetection detection : detections) {
            Color roiColor = detection.name.startsWith("MANUAL_") ? Color.GREEN : Color.BLUE;

            double x = detection.cx - detection.radius;
            double y = detection.cy - detection.radius;
            double diameter = detection.radius * 2;

            OvalRoi circle = new OvalRoi(x, y, diameter, diameter);
            circle.setStrokeColor(roiColor);
            circle.setStrokeWidth(1.5);
            circle.setName(detection.name);
            overlay.add(circle);

            double dotRadius = 1.8;
            OvalRoi dot = new OvalRoi(
                    detection.cx - dotRadius,
                    detection.cy - dotRadius,
                    dotRadius * 2,
                    dotRadius * 2
            );
            dot.setStrokeColor(roiColor);
            dot.setFillColor(roiColor);
            dot.setName("CENTER_" + detection.name);
            overlay.add(dot);
        }

        detectedImage.setOverlay(overlay);
        detectedImage.updateAndDraw();
    }

    private void updateDisplayedImage(boolean flattenOverlay) {
        if (imageToDisplay == null) {
            return;
        }

        ImagePlus displayImage = imageToDisplay.flatten();
        ImageProcessor processor = displayImage.getProcessor();
        BufferedImage bufferedImage = processor.getBufferedImage();

        int originalWidth = bufferedImage.getWidth();
        int originalHeight = bufferedImage.getHeight();

        int targetWidth;
        int targetHeight;

        if (fitToPanel && imageScrollPane != null) {
            int availableWidth = Math.max(100, imageScrollPane.getViewport().getWidth() - 20);
            int availableHeight = Math.max(100, imageScrollPane.getViewport().getHeight() - 20);

            double scaleX = availableWidth / (double) originalWidth;
            double scaleY = availableHeight / (double) originalHeight;

            double scale = Math.min(scaleX, scaleY);
            scale = Math.max(0.1, scale);

            targetWidth = (int) Math.round(originalWidth * scale);
            targetHeight = (int) Math.round(originalHeight * scale);

            zoomFactor = scale;

        } else {
            targetWidth = (int) Math.round(originalWidth * zoomFactor);
            targetHeight = (int) Math.round(originalHeight * zoomFactor);
        }

        Image scaledImage = bufferedImage.getScaledInstance(
                targetWidth,
                targetHeight,
                Image.SCALE_SMOOTH
        );

        SwingUtilities.invokeLater(() -> {
            imagePanel.setImage(scaledImage, targetWidth, targetHeight);

            if (fitToPanel) {
                zoomLabel.setText("Fit");
            } else {
                zoomLabel.setText((int) Math.round(zoomFactor * 100) + "%");
            }

            frame.revalidate();
            frame.repaint();
        });
    }

    private void enableCorrectionMode() {
        addNeuronMode = false;

        deleteSelectedButton.setEnabled(true);
        addNeuronButton.setEnabled(true);
        saveButton.setEnabled(true);

        statusLabel.setText("Correction mode enabled. Select a ROI to delete it or click Add neuron mode.");

        IJ.showMessage(
                "Correction mode",
                "Correction mode enabled.\n\n" +
                        "- Select a detection from the ROI list and click 'Delete selected ROI'.\n" +
                        "- Click 'Add neuron mode' and then click on the image to add a missing neuron.\n" +
                        "- When finished, click 'Save corrections'."
        );
    }

    private void saveCorrections() {
        IJ.showMessage(
                "Save corrections",
                "This step will save the corrected detections for transfer learning.\n\n" +
                        "Current status: pending implementation."
        );

        statusLabel.setText("Save corrections pending implementation.");
        retrainButton.setEnabled(true);
    }

    private void retrainModel() {
        IJ.showMessage(
                "Transfer Learning",
                "This step will retrain the model using corrected detections.\n\n" +
                        "Current status: pending implementation."
        );

        statusLabel.setText("Transfer learning pending implementation.");
    }

    private void deleteSelectedDetection() {
        int selectedIndex = roiList.getSelectedIndex();

        if (selectedIndex < 0) {
            IJ.showMessage("Delete ROI", "Please select a ROI from the list first.");
            return;
        }

        detections.remove(selectedIndex);
        refreshAfterEditing();

        statusLabel.setText("Deleted selected ROI. Current neurons: " + detections.size());
    }

    private void addManualDetectionAt(double imageX, double imageY) {
        int manualIndex = 1;

        for (NeuronDetection detection : detections) {
            if (detection.name.startsWith("MANUAL_")) {
                manualIndex++;
            }
        }

        double defaultRadius = 8.0;

        detections.add(new NeuronDetection(
                "MANUAL_" + manualIndex,
                imageX,
                imageY,
                defaultRadius
        ));

        addNeuronMode = false;
        addNeuronButton.setText("Add neuron mode");

        refreshAfterEditing();

        statusLabel.setText("Manual neuron added. Current neurons: " + detections.size());
    }

    private void refreshAfterEditing() {
        roiListModel.clear();

        for (NeuronDetection detection : detections) {
            roiListModel.addElement(detection.name);
        }

        applyOverlayToDetectedImage();

        imageToDisplay = detectedImage != null ? detectedImage : sourceImage;
        updateDisplayedImage(true);

        statusLabel.setText("Corrected neurons: " + detections.size());
    }

    private int parseNeuronCount(String line) {
        try {
            String[] parts = line.split(":");

            if (parts.length > 1) {
                return Integer.parseInt(parts[1].trim());
            }

        } catch (Exception e) {
            IJ.log("Could not parse neuron count from line: " + line);
        }

        return -1;
    }

    private void deleteIfExists(File file) {
        if (file.exists() && !file.delete()) {
            IJ.log("Warning: previous file could not be deleted: " + file.getAbsolutePath());
        }
    }

    private static class DetectionResult {
        private final ImagePlus image;
        private final int count;

        private DetectionResult(ImagePlus image, int count) {
            this.image = image;
            this.count = count;
        }
    }

    private static class NeuronDetection {
        private final String name;
        private final double cx;
        private final double cy;
        private final double radius;

        private NeuronDetection(String name, double cx, double cy, double radius) {
            this.name = name;
            this.cx = cx;
            this.cy = cy;
            this.radius = radius;
        }
    }

    private class ImagePanel extends JPanel {

        private Image image;
        private int imageWidth;
        private int imageHeight;

        private int offsetX;
        private int offsetY;

        private ImagePanel() {
            setPreferredSize(new Dimension(640, 480));

            addMouseListener(new java.awt.event.MouseAdapter() {
                @Override
                public void mouseClicked(java.awt.event.MouseEvent e) {
                    if (!addNeuronMode || imageToDisplay == null) {
                        return;
                    }

                    double imageX = (e.getX() - offsetX) / zoomFactor;
                    double imageY = (e.getY() - offsetY) / zoomFactor;

                    if (imageX < 0 || imageY < 0 ||
                            imageX >= imageToDisplay.getWidth() ||
                            imageY >= imageToDisplay.getHeight()) {
                        return;
                    }

                    addManualDetectionAt(imageX, imageY);
                }
            });
        }

        private void setImage(Image image, int width, int height) {
            this.image = image;
            this.imageWidth = width;
            this.imageHeight = height;
            setPreferredSize(new Dimension(width, height));
            revalidate();
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);

            if (image == null) {
                g.setColor(Color.WHITE);
                g.drawString("Open an image in Fiji and click Detect neurons.", 30, 40);
                return;
            }

            offsetX = Math.max(0, (getWidth() - imageWidth) / 2);
            offsetY = Math.max(0, (getHeight() - imageHeight) / 2);

            g.drawImage(image, offsetX, offsetY, imageWidth, imageHeight, this);
        }
    }
}