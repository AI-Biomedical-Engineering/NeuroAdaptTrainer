package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.OvalRoi;
import ij.io.FileSaver;
import ij.plugin.frame.RoiManager;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;
import javax.swing.BorderFactory;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.GridLayout;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.InputStreamReader;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Neuron Segmentation Assistant")
public class NeuronSegmentationAssistantCommand implements Command {

    private static final String PYTHON_EXE =
            "/Users/danielaerasocasas/tfg/venv/bin/python3";

    private static final String SCRIPT_PATH =
            "/Users/danielaerasocasas/Documents/gitHub/fiji-yolo-neuron-segmentation/yolo-inference/infer_one.py";

    private JFrame frame;
    private JLabel statusLabel;
    private JButton detectButton;
    private JButton correctButton;
    private JButton saveCorrectionsButton;
    private JButton retrainButton;

    private ImagePlus detectedImage;
    private int lastDetectedCount = -1;

    @Override
    public void run() {
        SwingUtilities.invokeLater(this::createAndShowGui);
    }

    private void createAndShowGui() {
        frame = new JFrame("Neuron Segmentation Assistant");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(460, 280);
        frame.setLayout(new BorderLayout());

        JPanel mainPanel = new JPanel();
        mainPanel.setLayout(new GridLayout(5, 1, 8, 8));
        mainPanel.setBorder(BorderFactory.createEmptyBorder(15, 15, 15, 15));

        statusLabel = new JLabel("Open an image in Fiji and start detection.");
        statusLabel.setHorizontalAlignment(SwingConstants.CENTER);

        detectButton = new JButton("1. Detect neurons");
        correctButton = new JButton("2. Correct detections");
        saveCorrectionsButton = new JButton("3. Save corrections for transfer learning");
        retrainButton = new JButton("4. Retrain model");

        correctButton.setEnabled(false);
        saveCorrectionsButton.setEnabled(false);
        retrainButton.setEnabled(false);

        detectButton.addActionListener(e -> detectNeurons());
        correctButton.addActionListener(e -> enableCorrectionMode());
        saveCorrectionsButton.addActionListener(e -> saveCorrections());
        retrainButton.addActionListener(e -> retrainModel());

        mainPanel.add(statusLabel);
        mainPanel.add(detectButton);
        mainPanel.add(correctButton);
        mainPanel.add(saveCorrectionsButton);
        mainPanel.add(retrainButton);

        frame.add(mainPanel, BorderLayout.CENTER);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private void detectNeurons() {
        ImagePlus imp = WindowManager.getCurrentImage();

        if (imp == null) {
            IJ.error("No image is open.");
            return;
        }

        statusLabel.setText("Detecting neurons...");
        detectButton.setEnabled(false);
        correctButton.setEnabled(false);
        saveCorrectionsButton.setEnabled(false);
        retrainButton.setEnabled(false);

        new Thread(() -> {
            try {
                DetectionResult detectionResult = runPythonDetection(imp);

                detectedImage = detectionResult.image;
                lastDetectedCount = detectionResult.count;

                SwingUtilities.invokeLater(() -> {
                    if (lastDetectedCount >= 0) {
                        detectedImage.setTitle("Detected Neurons - " + lastDetectedCount + " neurons");
                    } else {
                        detectedImage.setTitle("Detected Neurons");
                    }

                    detectedImage.show();

                    loadDetectionsAsEditableROIs(detectedImage);
                    RoiOverlayUpdater.updateOverlay(detectedImage);

                    IJ.showStatus("Detected neurons: " + lastDetectedCount);
                    IJ.showMessage("Neuron Detection", "Detected neurons: " + lastDetectedCount);

                    statusLabel.setText("Detection completed. You can now correct the detections.");
                    correctButton.setEnabled(true);
                    saveCorrectionsButton.setEnabled(true);
                });

            } catch (Exception ex) {
                IJ.handleException(ex);

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

        FileSaver saver = new FileSaver(imp);
        boolean saved = saver.saveAsPng(inputFile.getAbsolutePath());

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

        IJ.log("Running Python inference from assistant...");
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

        ImagePlus result = IJ.openImage(outputFile.getAbsolutePath());

        if (result == null) {
            throw new RuntimeException("The output image could not be opened.");
        }

        return new DetectionResult(result, neuronCount);
    }

    private void loadDetectionsAsEditableROIs(ImagePlus imp) {
        try {
            String tmpDir = System.getProperty("java.io.tmpdir");
            File csvFile = new File(tmpDir, "imagej_output.csv");

            if (!csvFile.exists()) {
                IJ.log("CSV not found: " + csvFile.getAbsolutePath());
                return;
            }

            RoiManager rm = RoiManager.getInstance();

            if (rm == null) {
                rm = new RoiManager();
            }

            // Remove previous automatic detections only.
            for (int i = rm.getCount() - 1; i >= 0; i--) {
                String roiName = rm.getName(i);

                if (roiName != null && roiName.startsWith("AUTO_")) {
                    rm.select(i);
                    rm.runCommand("Delete");
                }
            }

            try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
                String line = br.readLine(); // skip header
                int autoIndex = 1;

                while ((line = br.readLine()) != null) {
                    String[] parts = line.split(",");

                    if (parts.length < 3) {
                        IJ.log("Skipping malformed CSV line: " + line);
                        continue;
                    }

                    double cx = Double.parseDouble(parts[0]);
                    double cy = Double.parseDouble(parts[1]);
                    double r = Double.parseDouble(parts[2]);

                    double x = cx - r;
                    double y = cy - r;
                    double d = r * 2;

                    OvalRoi roi = new OvalRoi(x, y, d, d);
                    roi.setStrokeColor(Color.BLUE);
                    roi.setStrokeWidth(1.5);
                    roi.setName("AUTO_" + autoIndex);

                    rm.addRoi(roi);
                    autoIndex++;
                }
            }

            IJ.log("Automatic detections loaded as editable ROIs.");

        } catch (Exception e) {
            IJ.handleException(e);
        }
    }

    private void enableCorrectionMode() {
        ImagePlus imp = WindowManager.getCurrentImage();

        if (imp == null) {
            IJ.error("No image is open.");
            return;
        }

        RoiManager rm = RoiManager.getInstance();

        if (rm == null) {
            rm = new RoiManager();
        }

        IJ.setTool("oval");

        IJ.showMessage(
                "Correction mode",
                "Correction mode enabled.\n\n" +
                        "- Delete wrong detections from the ROI Manager.\n" +
                        "- Add missing neurons with the oval tool.\n" +
                        "- Press 't' after drawing each new neuron.\n\n" +
                        "When finished, click 'Save corrections for transfer learning'."
        );

        startManualRoiWatcher(rm);

        statusLabel.setText("Correction mode enabled.");
        saveCorrectionsButton.setEnabled(true);
    }

    private void startManualRoiWatcher(RoiManager rm) {
        new Thread(() -> {
            int lastCount = -1;

            while (frame != null && frame.isDisplayable()) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    return;
                }

                int currentCount = rm.getCount();

                if (currentCount != lastCount) {
                    for (int i = 0; i < rm.getCount(); i++) {
                        String roiName = rm.getName(i);

                        if (roiName == null ||
                                roiName.isBlank() ||
                                (!roiName.startsWith("AUTO_") && !roiName.startsWith("MANUAL_"))) {

                            rm.rename(i, "MANUAL_" + (i + 1));
                        }
                    }

                    lastCount = currentCount;

                    ImagePlus currentImage = WindowManager.getCurrentImage();

                    if (currentImage != null) {
                        RoiOverlayUpdater.updateOverlay(currentImage);
                    }
                }
            }
        }).start();
    }

    private void saveCorrections() {
        /*
         * Siguiente paso:
         * aquí guardaremos automáticamente la imagen + ROIs corregidas
         * para crear el dataset de transfer learning.
         */
        IJ.log("Save corrections step pending implementation.");

        statusLabel.setText("Corrections saved. Ready for transfer learning.");
        retrainButton.setEnabled(true);

        IJ.showMessage(
                "Corrections",
                "This step will save the corrected ROIs automatically for transfer learning.\n\n" +
                        "Current status: interface ready, export pending implementation."
        );
    }

    private void retrainModel() {
        IJ.showMessage(
                "Transfer Learning",
                "This step will retrain the model using the corrected detections.\n\n" +
                        "Current status: training pipeline pending implementation."
        );

        statusLabel.setText("Transfer learning pending implementation.");
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
}