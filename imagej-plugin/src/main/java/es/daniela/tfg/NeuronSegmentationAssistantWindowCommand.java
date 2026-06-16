package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.io.FileSaver;
import ij.ImageListener;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.Properties;
import java.util.Locale;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Neuron Segmentation Assistant Window")
public class NeuronSegmentationAssistantWindowCommand implements Command, ImageListener {

    private String pythonExe;
    private String scriptPath;
    private String retrainScriptPath;

    private JFrame frame;
    private DefaultListModel<String> roiListModel;
    private JList<String> roiList;
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

    private JButton importImageButton;
    private JButton deleteSelectedButton;
    private JButton addNeuronButton;
    private boolean addNeuronMode = false;
    private boolean correctionMode = false;
    private boolean detectionRunning = false;
    private boolean retrainingRunning = false;
    private String currentMessage = "Ready.";

    private ImagePanel imagePanel;

    private JButton detectButton;
    private JButton correctButton;
    private JButton saveButton;
    private JButton retrainButton;

    private ImagePlus sourceImage;
    private ImagePlus detectedImage;
    private final List<NeuronDetection> detections = new ArrayList<>();
    private int selectedDetectionIndex = -1;
    private final String sessionId = UUID.randomUUID().toString();
    private File sessionDir;

    // Images imported from the assistant should not be auto-loaded by other assistant windows.
    private static final Set<Integer> ASSISTANT_IMPORTED_IMAGE_IDS =
            Collections.synchronizedSet(new HashSet<>());

    @Override
    public void run() {
        SwingUtilities.invokeLater(this::createWindow);
    }

    private void createWindow() {
        loadConfiguration();

        sessionDir = new File(
                System.getProperty("java.io.tmpdir"),
                "neuron_assistant_" + sessionId
        );

        if (!sessionDir.exists() && !sessionDir.mkdirs()) {
            IJ.log("Warning: could not create session directory: " + sessionDir.getAbsolutePath());
        } else {
            IJ.log("Temporary session directory: " + sessionDir.getAbsolutePath());
        }

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

        ImagePlus.addImageListener(this);

        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                ImagePlus.removeImageListener(NeuronSegmentationAssistantWindowCommand.this);

                if (sessionDir != null) {
                    deleteDirectory(sessionDir);
                }
            }
        });

        if (sourceImage != null) {
            imageToDisplay = sourceImage;
            updateStatus("Selected image: " + sourceImage.getTitle());
            updateButtonState();
            updateDisplayedImage(false);
        } else {
            updateStatus("No Fiji image selected.");
            updateButtonState();
        }
    }

    private void loadConfiguration() {
        try {
            File configFile = new File(
                    System.getProperty("user.home"),
                    ".neuron-segmentation-assistant/config.properties"
            );

            if (!configFile.exists()) {
                IJ.error(
                        "Configuration not found",
                        "Config file not found:\n" +
                                configFile.getAbsolutePath() +
                                "\n\nPlease run the installer first."
                );
                return;
            }

            Properties properties = new Properties();

            try (FileInputStream fis = new FileInputStream(configFile)) {
                properties.load(fis);
            }

            pythonExe = properties.getProperty("python");
            scriptPath = properties.getProperty("script");
            retrainScriptPath = properties.getProperty("retrain_script");

            if (pythonExe == null || scriptPath == null) {
                IJ.error(
                        "Invalid configuration",
                        "The configuration file must contain 'python' and 'script' paths."
                );
                return;
            }

            IJ.log("Loaded Python executable: " + pythonExe);
            IJ.log("Loaded inference script: " + scriptPath);

            if (retrainScriptPath != null) {
                IJ.log("Loaded retraining script: " + retrainScriptPath);
            } else {
                IJ.log("Retraining script not configured yet.");
            }

        } catch (Exception e) {
            IJ.handleException(e);
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

    private void setSourceImage(ImagePlus image, String message) {
        if (image == null) {
            updateStatus("No image selected.");
            return;
        }

        sourceImage = image;
        imageToDisplay = sourceImage;
        detectedImage = null;

        detections.clear();
        roiListModel.clear();
        selectedDetectionIndex = -1;

        correctionMode = false;
        addNeuronMode = false;

        if (addNeuronButton != null) {
            addNeuronButton.setText("2.2 Add neurons");
        }

        fitToPanel = true;

        updateStatus(message + ": " + sourceImage.getTitle());
        updateButtonState();
        updateDisplayedImage(false);
    }

    private JPanel createLeftPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(180, 0));
        panel.setBorder(BorderFactory.createTitledBorder("ROI list"));

        roiListModel = new DefaultListModel<>();
        roiList = new JList<>(roiListModel);
        roiList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        roiList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                int index = roiList.getSelectedIndex();

                if (index >= 0 && index < detections.size()) {
                    selectedDetectionIndex = index;
                }

                if (imagePanel != null) {
                    imagePanel.repaint();
                }
            }
        });

        roiList.getInputMap(JComponent.WHEN_FOCUSED).put(
                KeyStroke.getKeyStroke("DELETE"),
                "deleteSelectedDetection"
        );

        roiList.getActionMap().put("deleteSelectedDetection", new AbstractAction() {
            @Override
            public void actionPerformed(java.awt.event.ActionEvent e) {
                deleteSelectedDetection();
            }
        });

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
        panel.setLayout(new GridLayout(7, 1, 8, 8));
        panel.setBorder(BorderFactory.createTitledBorder("Actions"));

        importImageButton = new JButton("0. Import image");
        detectButton = new JButton("1. Detect neurons");
        correctButton = new JButton("2. Correct detections");
        deleteSelectedButton = new JButton("2.1 Delete neuron");
        addNeuronButton = new JButton("2.2 Add neurons");
        saveButton = new JButton("2.3 Save corrections");
        retrainButton = new JButton("3. Retrain model");

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
                addNeuronButton.setText("2.2 Stop adding neurons");
                updateStatus("Add neuron mode enabled. Click and drag to add neurons. You can still delete the selected neuron.");
            } else {
                addNeuronButton.setText("2.2 Add neurons");
                updateStatus("Add neuron mode disabled. You can select, move or delete neurons.");
            }

            updateButtonState();

            if (imagePanel != null) {
                imagePanel.repaint();
            }
        });

        importImageButton.addActionListener(e -> importImageFromFile());

        panel.add(importImageButton);
        panel.add(detectButton);
        panel.add(correctButton);
        panel.add(deleteSelectedButton);
        panel.add(addNeuronButton);
        panel.add(saveButton);
        panel.add(retrainButton);

        return panel;
    }

    private void importImageFromFile() {
        JFileChooser chooser = new JFileChooser();

        int result = chooser.showOpenDialog(frame);

        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File selectedFile = chooser.getSelectedFile();

        ImagePlus importedImage = IJ.openImage(selectedFile.getAbsolutePath());

        if (importedImage == null) {
            IJ.error("Could not open selected image.");
            return;
        }

        ASSISTANT_IMPORTED_IMAGE_IDS.add(importedImage.getID());

        importedImage.show();
        setSourceImage(importedImage, "Imported image");
    }

    private JPanel createBottomPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        statusLabel = new JLabel("Detected neurons: 0");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
        panel.add(statusLabel, BorderLayout.WEST);
        return panel;
    }

    private void updateStatus(String message) {
        currentMessage = message;

        int count = detections.size();

        if (count > 0) {
            statusLabel.setText("Neurons: " + count + " | " + currentMessage);
        } else {
            statusLabel.setText(currentMessage);
        }
    }

    private void updateButtonState() {
        boolean hasDetections = !detections.isEmpty();
        boolean busy = detectionRunning || retrainingRunning;

        if (importImageButton != null) {
            importImageButton.setEnabled(!busy && !correctionMode && !addNeuronMode);
        }

        detectButton.setEnabled(
                sourceImage != null &&
                        !hasDetections &&
                        !correctionMode &&
                        !addNeuronMode &&
                        !busy
        );

        correctButton.setEnabled(
                hasDetections &&
                        !correctionMode &&
                        !addNeuronMode &&
                        !busy
        );

        deleteSelectedButton.setEnabled(
                correctionMode &&
                        hasDetections &&
                        !busy
        );

        addNeuronButton.setEnabled(
                correctionMode &&
                        !busy
        );

        saveButton.setEnabled(
                correctionMode &&
                        !addNeuronMode &&
                        !busy
        );

        retrainButton.setEnabled(
                !correctionMode &&
                        hasDetections &&
                        !busy
        );
    }

    private void detectNeurons() {
        if (pythonExe == null || scriptPath == null) {

            IJ.error("Plugin is not configured. Please run the installer first.");

            return;

        }

        ImagePlus currentImage = sourceImage;

        if (currentImage == null) {
            currentImage = getCurrentOrFirstImage();

            if (currentImage != null) {
                setSourceImage(currentImage, "Selected image");
            }
        }

        if (currentImage == null) {
            IJ.error("No image is open in Fiji.");
            return;
        }

        sourceImage = currentImage;
        final ImagePlus imageForDetection = currentImage;
        IJ.log("Assistant selected source image: " + sourceImage.getTitle());

        detectionRunning = true;

        correctionMode = false;
        addNeuronMode = false;
        selectedDetectionIndex = -1;

        detectButton.setEnabled(false);
        correctButton.setEnabled(false);
        saveButton.setEnabled(false);
        retrainButton.setEnabled(false);

        updateStatus("Detecting neurons... Please wait.");
        updateButtonState();

        roiListModel.clear();
        detections.clear();

        new Thread(() -> {
            try {
                DetectionResult result = runPythonDetection(imageForDetection);
                detectedImage = result.image;

                loadDetectionsFromCsv();

                imageToDisplay = detectedImage;
                fitToPanel = true;
                updateDisplayedImage(true);

                SwingUtilities.invokeLater(() -> {
                    detectionRunning = false;

                    updateStatus("Detection completed. Click 'Correct detections' to review the result.");
                    updateButtonState();

                    IJ.showStatus("Detected neurons: " + detections.size());
                    IJ.log("Assistant detection completed. Count: " + detections.size());
                });

            } catch (Exception e) {
                IJ.handleException(e);

                SwingUtilities.invokeLater(() -> {
                    detectionRunning = false;

                    updateStatus("Detection failed.");
                    updateButtonState();
                });
            }
        }).start();
    }

    private DetectionResult runPythonDetection(ImagePlus imp) throws Exception {
        if (sessionDir == null) {
            sessionDir = new File(
                    System.getProperty("java.io.tmpdir"),
                    "neuron_assistant_" + sessionId
            );
        }

        if (!sessionDir.exists() && !sessionDir.mkdirs()) {
            throw new RuntimeException("Could not create session directory: " + sessionDir.getAbsolutePath());
        }

        File inputFile = new File(sessionDir, "imagej_input.png");
        File outputFile = new File(sessionDir, "imagej_output.png");
        File csvFile = new File(sessionDir, "imagej_output.csv");

        deleteIfExists(outputFile);
        deleteIfExists(csvFile);

        boolean saved = new FileSaver(imp).saveAsPng(inputFile.getAbsolutePath());

        if (!saved) {
            throw new RuntimeException("Could not save the input image.");
        }

        if (pythonExe == null || scriptPath == null) {
            throw new RuntimeException("Plugin is not configured. Please run the installer first.");
        }

        File pythonFile = new File(pythonExe);
        File scriptFile = new File(scriptPath);

        if (!pythonFile.exists()) {
            throw new RuntimeException("Python executable not found: " + pythonExe);
        }

        if (!scriptFile.exists()) {
            throw new RuntimeException("Inference script not found: " + scriptPath);
        }

        ProcessBuilder pb = new ProcessBuilder(
                pythonExe,
                "-u",
                scriptPath,
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

    private void enableCorrectionMode() {
        correctionMode = true;
        addNeuronMode = false;
        selectedDetectionIndex = -1;

        addNeuronButton.setText("2.2 Add neurons");
        roiList.clearSelection();

        updateStatus("Correction mode enabled. Click a neuron to select it, drag to move it, delete it, or add a new one.");
        updateButtonState();

        if (imagePanel != null) {
            imagePanel.repaint();
        }

        IJ.showMessage(
                "Correction mode",
                "Correction mode enabled.\n\n" +
                        "- Click on a detected neuron to select it.\n" +
                        "- Drag the selected neuron to move it.\n" +
                        "- Click 'Delete selected neuron' to remove it.\n" +
                        "- Click 'Add neuron mode' and draw as many missing neurons as needed.\n" +
                        "- While adding neurons, right-click on a neuron to select it and delete it.\n" +
                        "- When finished, click 'Save corrections'."
        );
    }

    private void loadDetectionsFromCsv() throws Exception {
        File csvFile = new File(sessionDir, "imagej_output.csv");

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
                double diameter = radius * 2.0;

                String maskPolygon = null;

                if (parts.length >= 5) {
                    maskPolygon = parts[4].replace("\"", "").trim();

                    if (maskPolygon.isEmpty()) {
                        maskPolygon = null;
                    }
                }

                detections.add(new NeuronDetection(
                        "AUTO_" + index,
                        cx,
                        cy,
                        diameter,
                        diameter,
                        maskPolygon,
                        false
                ));
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

    private File getTransferLearningRootDir() {
        File root = new File(
                System.getProperty("user.home"),
                ".neuron-segmentation-assistant/transfer-learning-data"
        );

        File imagesTrain = new File(root, "images/train");
        File labelsTrain = new File(root, "labels/train");

        imagesTrain.mkdirs();
        labelsTrain.mkdirs();

        return root;
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    private List<Double> ellipseToYoloPolygon(
            NeuronDetection detection,
            int imageWidth,
            int imageHeight,
            int points
    ) {
        List<Double> coords = new ArrayList<>();

        double rx = detection.width / 2.0;
        double ry = detection.height / 2.0;

        for (int i = 0; i < points; i++) {
            double angle = 2.0 * Math.PI * i / points;

            double x = detection.cx + rx * Math.cos(angle);
            double y = detection.cy + ry * Math.sin(angle);

            x = clamp(x, 0, imageWidth - 1);
            y = clamp(y, 0, imageHeight - 1);

            coords.add(x / imageWidth);
            coords.add(y / imageHeight);
        }

        return coords;
    }

    private List<Double> maskPolygonToNormalizedYolo(
            String maskPolygon,
            int imageWidth,
            int imageHeight
    ) {
        List<Double> coords = new ArrayList<>();

        if (maskPolygon == null || maskPolygon.trim().isEmpty()) {
            return coords;
        }

        String[] values = maskPolygon.trim().split("\\s+");

        for (int i = 0; i + 1 < values.length; i += 2) {
            double x = Double.parseDouble(values[i]);
            double y = Double.parseDouble(values[i + 1]);

            x = clamp(x, 0, imageWidth - 1);
            y = clamp(y, 0, imageHeight - 1);

            coords.add(x / imageWidth);
            coords.add(y / imageHeight);
        }

        return coords;
    }

    private List<Double> detectionToYoloPolygon(
            NeuronDetection detection,
            int imageWidth,
            int imageHeight
    ) {
        boolean canUseOriginalMask =
                !detection.edited &&
                        detection.maskPolygon != null &&
                        !detection.maskPolygon.trim().isEmpty();

        if (canUseOriginalMask) {
            try {
                List<Double> maskCoords = maskPolygonToNormalizedYolo(
                        detection.maskPolygon,
                        imageWidth,
                        imageHeight
                );

                // YOLO segmentation needs at least 3 points = 6 values.
                if (maskCoords.size() >= 6) {
                    return maskCoords;
                }

            } catch (Exception e) {
                IJ.log("Could not use original mask polygon for " + detection.name + ". Falling back to ellipse.");
            }
        }

        return ellipseToYoloPolygon(
                detection,
                imageWidth,
                imageHeight,
                32
        );
    }

    private void writeYoloLabelFile(File labelFile, int imageWidth, int imageHeight) throws IOException {
        try (PrintWriter writer = new PrintWriter(new FileWriter(labelFile))) {
            for (NeuronDetection detection : detections) {
                List<Double> polygon = detectionToYoloPolygon(
                        detection,
                        imageWidth,
                        imageHeight
                );

                if (polygon.size() < 6) {
                    IJ.log("Skipping invalid polygon for detection: " + detection.name);
                    continue;
                }

                StringBuilder line = new StringBuilder();
                line.append("0");

                for (Double value : polygon) {
                    line.append(" ");
                    line.append(String.format(Locale.US, "%.6f", value));
                }

                writer.println(line);
            }
        }
    }

    private void writeDataYaml(File rootDir) throws IOException {
        File yamlFile = new File(rootDir, "data.yaml");

        try (PrintWriter writer = new PrintWriter(new FileWriter(yamlFile))) {
            writer.println("path: " + rootDir.getAbsolutePath().replace("\\", "/"));
            writer.println("train: images/train");
            writer.println("val: images/train");
            writer.println("names:");
            writer.println("  0: neuron");
        }
    }

    private void updateDisplayedImage(boolean flattenOverlay) {
        if (imageToDisplay == null) {
            return;
        }

        ImagePlus displayImage = imageToDisplay;

        Image awtImage = displayImage.getImage();

        int originalWidth = displayImage.getWidth();
        int originalHeight = displayImage.getHeight();

        BufferedImage bufferedImage = new BufferedImage(
                originalWidth,
                originalHeight,
                BufferedImage.TYPE_INT_RGB
        );

        Graphics2D g2 = bufferedImage.createGraphics();
        g2.drawImage(awtImage, 0, 0, null);
        g2.dispose();

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

    private void saveCorrections() {
        if (sourceImage == null) {
            IJ.error("No source image available.");
            return;
        }

        if (detections.isEmpty()) {
            IJ.error("There are no detections to save.");
            return;
        }

        try {
            File rootDir = getTransferLearningRootDir();

            File imagesTrainDir = new File(rootDir, "images/train");
            File labelsTrainDir = new File(rootDir, "labels/train");

            String baseName = sanitizeFileName(
                    sourceImage.getTitle().replaceFirst("[.][^.]+$", "")
            );

            String uniqueName = baseName + "_" + sessionId.substring(0, 8);

            File imageFile = new File(imagesTrainDir, uniqueName + ".png");
            File labelFile = new File(labelsTrainDir, uniqueName + ".txt");

            boolean savedImage = new FileSaver(sourceImage).saveAsPng(imageFile.getAbsolutePath());

            if (!savedImage) {
                throw new RuntimeException("Could not save corrected training image.");
            }

            writeYoloLabelFile(
                    labelFile,
                    sourceImage.getWidth(),
                    sourceImage.getHeight()
            );

            writeDataYaml(rootDir);

            correctionMode = false;
            addNeuronMode = false;
            selectedDetectionIndex = -1;

            addNeuronButton.setText("2.2 Add neurons");
            roiList.clearSelection();

            updateStatus("Corrections saved for transfer learning.");
            updateButtonState();

            if (imagePanel != null) {
                imagePanel.repaint();
            }

            IJ.log("Transfer learning image saved to: " + imageFile.getAbsolutePath());
            IJ.log("Transfer learning label saved to: " + labelFile.getAbsolutePath());
            IJ.log("Transfer learning data.yaml saved to: " + new File(rootDir, "data.yaml").getAbsolutePath());

            IJ.showMessage(
                    "Save corrections",
                    "Corrections saved for transfer learning.\n\n" +
                            "Image:\n" + imageFile.getAbsolutePath() + "\n\n" +
                            "Label:\n" + labelFile.getAbsolutePath()
            );

        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not save corrections.");
            updateButtonState();
        }
    }

    private void retrainModel() {
        if (pythonExe == null || retrainScriptPath == null) {
            IJ.error(
                    "Retraining not configured",
                    "The retraining script is not configured.\n\n" +
                            "Please run the installer again or check config.properties."
            );
            return;
        }

        File pythonFile = new File(pythonExe);
        File retrainScriptFile = new File(retrainScriptPath);

        if (!pythonFile.exists()) {
            IJ.error("Python executable not found:\n" + pythonExe);
            return;
        }

        if (!retrainScriptFile.exists()) {
            IJ.error("Retraining script not found:\n" + retrainScriptPath);
            return;
        }

        retrainingRunning = true;
        updateStatus("Retraining model... This may take several minutes. Please wait.");
        updateButtonState();

        new Thread(() -> {
            try {
                File rootDir = getTransferLearningRootDir();
                File dataYaml = new File(rootDir, "data.yaml");

                if (!dataYaml.exists()) {
                    throw new RuntimeException(
                            "Transfer learning data.yaml not found. Please save corrections before retraining."
                    );
                }

                ProcessBuilder pb = new ProcessBuilder(
                        pythonExe,
                        "-u",
                        retrainScriptPath,
                        dataYaml.getAbsolutePath()
                );

                pb.redirectErrorStream(true);

                IJ.log("Running model retraining...");
                IJ.log(String.join(" ", pb.command()));

                Process process = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {

                    String line;

                    while ((line = reader.readLine()) != null) {
                        IJ.log(line);
                    }
                }

                int exitCode = process.waitFor();

                if (exitCode != 0) {
                    throw new RuntimeException("Retraining failed with exit code " + exitCode);
                }

                SwingUtilities.invokeLater(() -> {
                    retrainingRunning = false;
                    updateStatus("Retraining completed successfully.");
                    updateButtonState();

                    IJ.showMessage(
                            "Retraining completed",
                            "The model was retrained successfully.\n\n" +
                                    "The adapted model has been generated and will be used in future detections."
                    );
                });

            } catch (Exception e) {
                IJ.handleException(e);

                SwingUtilities.invokeLater(() -> {
                    retrainingRunning = false;
                    updateStatus("Retraining failed.");
                    updateButtonState();
                });
            }
        }).start();
    }

    private void deleteSelectedDetection() {
        int selectedIndex = selectedDetectionIndex;

        if (selectedIndex < 0 || selectedIndex >= detections.size()) {
            selectedIndex = roiList.getSelectedIndex();
        }

        if (selectedIndex < 0 || selectedIndex >= detections.size()) {
            IJ.showMessage(
                    "Delete neuron",
                    "Please select a neuron first by clicking on the image or selecting it from the ROI list."
            );
            return;
        }

        detections.remove(selectedIndex);

        selectedDetectionIndex = -1;
        roiList.clearSelection();

        refreshAfterEditing();

        updateStatus("Deleted selected neuron.");
        updateButtonState();
    }

    private void refreshAfterEditing() {
        roiListModel.clear();

        for (NeuronDetection detection : detections) {
            roiListModel.addElement(detection.name);
        }

        if (selectedDetectionIndex >= 0 && selectedDetectionIndex < detections.size()) {
            roiList.setSelectedIndex(selectedDetectionIndex);
            roiList.ensureIndexIsVisible(selectedDetectionIndex);
        } else {
            roiList.clearSelection();
        }

        if (imagePanel != null) {
            imagePanel.repaint();
        }

        updateButtonState();
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

    private void deleteDirectory(File directory) {
        if (directory == null || !directory.exists()) {
            return;
        }

        File[] files = directory.listFiles();

        if (files != null) {
            for (File file : files) {
                if (file.isDirectory()) {
                    deleteDirectory(file);
                } else {
                    if (!file.delete()) {
                        IJ.log("Warning: could not delete temp file: " + file.getAbsolutePath());
                    }
                }
            }
        }

        if (!directory.delete()) {
            IJ.log("Warning: could not delete temp directory: " + directory.getAbsolutePath());
        } else {
            IJ.log("Temporary session directory deleted: " + directory.getAbsolutePath());
        }
    }

    @Override
    public void imageOpened(ImagePlus imp) {
        SwingUtilities.invokeLater(() -> {
            if (imp == null) {
                return;
            }

            // If the image was imported from one assistant window,
            // other assistant windows should not auto-load it.
            if (ASSISTANT_IMPORTED_IMAGE_IDS.contains(imp.getID())) {
                return;
            }

            // Do not change the image while the user is editing corrections.
            if (correctionMode || addNeuronMode) {
                return;
            }

            // If this assistant window already has an image assigned,
            // do not replace it automatically.
            if (sourceImage != null) {
                return;
            }

            setSourceImage(imp, "Image opened in Fiji");
        });
    }

    @Override
    public void imageClosed(ImagePlus imp) {
        SwingUtilities.invokeLater(() -> {
            if (imp != null) {
                ASSISTANT_IMPORTED_IMAGE_IDS.remove(imp.getID());
            }

            if (imp != null && imp == sourceImage) {
                sourceImage = null;
                imageToDisplay = null;
                detectedImage = null;

                detections.clear();
                roiListModel.clear();
                selectedDetectionIndex = -1;

                updateStatus("Selected image was closed.");
                updateButtonState();

                if (imagePanel != null) {
                    imagePanel.setImage(null, 640, 480);
                    imagePanel.repaint();
                }
            }
        });
    }

    @Override
    public void imageUpdated(ImagePlus imp) {
        // Not needed for now.
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
        private String name;
        private double cx;
        private double cy;
        private double width;
        private double height;

        // Original YOLO mask polygon, exported by infer_one.py.
        // Stored in absolute image coordinates: x1 y1 x2 y2 ...
        private String maskPolygon;

        // True when the user moves, edits or creates the detection manually.
        // If true, the detection is exported as an ellipse polygon.
        private boolean edited;

        private NeuronDetection(String name, double cx, double cy, double width, double height) {
            this(name, cx, cy, width, height, null, false);
        }

        private NeuronDetection(
                String name,
                double cx,
                double cy,
                double width,
                double height,
                String maskPolygon,
                boolean edited
        ) {
            this.name = name;
            this.cx = cx;
            this.cy = cy;
            this.width = width;
            this.height = height;
            this.maskPolygon = maskPolygon;
            this.edited = edited;
        }
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private void addManualDetection(double cx, double cy, double width, double height) {
        int manualIndex = 1;

        for (NeuronDetection detection : detections) {
            if (detection.name.startsWith("MANUAL_")) {
                manualIndex++;
            }
        }

        detections.add(new NeuronDetection(
                "MANUAL_" + manualIndex,
                cx,
                cy,
                width,
                height,
                null,
                true
        ));

        selectedDetectionIndex = detections.size() - 1;

        refreshAfterEditing();

        updateStatus("Manual neuron added. Keep drawing or click 'Stop adding neurons' to finish.");
        updateButtonState();
    }

    private class ImagePanel extends JPanel {

        private Image image;
        private int imageWidth;
        private int imageHeight;

        private int offsetX;
        private int offsetY;

        private boolean drawing = false;
        private int dragStartX;
        private int dragStartY;
        private int dragCurrentX;
        private int dragCurrentY;

        private boolean movingSelected = false;
        private double lastMoveImageX;
        private double lastMoveImageY;

        private ImagePanel() {
            setPreferredSize(new Dimension(640, 480));
            setBackground(Color.DARK_GRAY);

            java.awt.event.MouseAdapter mouseAdapter = new java.awt.event.MouseAdapter() {

                @Override
                public void mousePressed(java.awt.event.MouseEvent e) {
                    if (imageToDisplay == null) {
                        return;
                    }

                    if (!isInsideDisplayedImage(e.getX(), e.getY())) {
                        return;
                    }

                    double imageX = screenToImageX(e.getX());
                    double imageY = screenToImageY(e.getY());

                    if (!correctionMode && !addNeuronMode) {
                        return;
                    }

                    if (addNeuronMode) {
                        int clickedIndex = findDetectionAt(imageX, imageY);

                        if (clickedIndex >= 0) {
                            selectedDetectionIndex = clickedIndex;

                            roiList.setSelectedIndex(clickedIndex);
                            roiList.ensureIndexIsVisible(clickedIndex);

                            NeuronDetection selected = detections.get(clickedIndex);
                            updateStatus("Selected " + selected.name + ". Click 'Delete selected neuron' to remove it, or drag empty space to add another.");

                            repaint();
                            return;
                        }

                        selectedDetectionIndex = -1;
                        roiList.clearSelection();

                        drawing = true;
                        dragStartX = e.getX();
                        dragStartY = e.getY();
                        dragCurrentX = e.getX();
                        dragCurrentY = e.getY();

                        updateStatus("Drawing new neuron...");
                        repaint();
                        return;
                    }

                    int clickedIndex = findDetectionAt(imageX, imageY);

                    if (clickedIndex >= 0) {
                        selectedDetectionIndex = clickedIndex;

                        roiList.setSelectedIndex(clickedIndex);
                        roiList.ensureIndexIsVisible(clickedIndex);

                        NeuronDetection selected = detections.get(clickedIndex);

                        movingSelected = true;
                        lastMoveImageX = imageX;
                        lastMoveImageY = imageY;

                        updateStatus("Selected " + selected.name + ". Drag to move or click Delete selected ROI.");

                        repaint();
                    } else {
                        selectedDetectionIndex = -1;
                        roiList.clearSelection();
                        updateStatus("No ROI selected.");
                        repaint();
                    }
                }

                @Override
                public void mouseDragged(java.awt.event.MouseEvent e) {
                    if (drawing) {
                        dragCurrentX = e.getX();
                        dragCurrentY = e.getY();
                        repaint();
                        return;
                    }

                    if (movingSelected) {
                        int selectedIndex = selectedDetectionIndex;

                        if (selectedIndex < 0 || selectedIndex >= detections.size()) {
                            selectedIndex = roiList.getSelectedIndex();
                        }

                        if (selectedIndex < 0 || selectedIndex >= detections.size()) {
                            return;
                        }

                        selectedDetectionIndex = selectedIndex;
                        roiList.ensureIndexIsVisible(selectedIndex);

                        double imageX = screenToImageX(e.getX());
                        double imageY = screenToImageY(e.getY());

                        NeuronDetection selected = detections.get(selectedIndex);

                        double dx = imageX - lastMoveImageX;
                        double dy = imageY - lastMoveImageY;

                        selected.cx = clamp(selected.cx + dx, 0, imageToDisplay.getWidth() - 1);
                        selected.cy = clamp(selected.cy + dy, 0, imageToDisplay.getHeight() - 1);
                        selected.edited = true;

                        lastMoveImageX = imageX;
                        lastMoveImageY = imageY;

                        repaint();
                    }
                }

                @Override
                public void mouseReleased(java.awt.event.MouseEvent e) {
                    if (drawing) {
                        drawing = false;

                        dragCurrentX = e.getX();
                        dragCurrentY = e.getY();

                        addManualDetectionFromDrag();

                        repaint();
                        return;
                    }

                    if (movingSelected) {
                        movingSelected = false;
                        updateStatus("Corrected neurons: " + detections.size());
                        repaint();
                    }
                }
            };

            addMouseListener(mouseAdapter);
            addMouseMotionListener(mouseAdapter);
        }

        private void setImage(Image image, int width, int height) {
            this.image = image;
            this.imageWidth = width;
            this.imageHeight = height;

            setPreferredSize(new Dimension(width, height));
            revalidate();
            repaint();
        }

        private boolean isInsideDisplayedImage(int x, int y) {
            return x >= offsetX &&
                    y >= offsetY &&
                    x <= offsetX + imageWidth &&
                    y <= offsetY + imageHeight;
        }

        private double screenToImageX(int screenX) {
            return (screenX - offsetX) / zoomFactor;
        }

        private double screenToImageY(int screenY) {
            return (screenY - offsetY) / zoomFactor;
        }

        private boolean isPointInsideDetection(double imageX, double imageY, NeuronDetection detection) {
            double rx = detection.width / 2.0;
            double ry = detection.height / 2.0;

            if (rx <= 0 || ry <= 0) {
                return false;
            }

            double normalizedX = (imageX - detection.cx) / rx;
            double normalizedY = (imageY - detection.cy) / ry;

            return normalizedX * normalizedX + normalizedY * normalizedY <= 1.0;
        }

        private void addManualDetectionFromDrag() {
            if (imageToDisplay == null) {
                return;
            }

            int x1Screen = Math.min(dragStartX, dragCurrentX);
            int y1Screen = Math.min(dragStartY, dragCurrentY);
            int x2Screen = Math.max(dragStartX, dragCurrentX);
            int y2Screen = Math.max(dragStartY, dragCurrentY);

            int widthScreen = x2Screen - x1Screen;
            int heightScreen = y2Screen - y1Screen;

            if (widthScreen < 4 || heightScreen < 4) {
                updateStatus("ROI too small. Drag a larger region.");
                return;
            }

            double x1Image = screenToImageX(x1Screen);
            double y1Image = screenToImageY(y1Screen);
            double x2Image = screenToImageX(x2Screen);
            double y2Image = screenToImageY(y2Screen);

            x1Image = clamp(x1Image, 0, imageToDisplay.getWidth() - 1);
            y1Image = clamp(y1Image, 0, imageToDisplay.getHeight() - 1);
            x2Image = clamp(x2Image, 0, imageToDisplay.getWidth() - 1);
            y2Image = clamp(y2Image, 0, imageToDisplay.getHeight() - 1);

            double cx = (x1Image + x2Image) / 2.0;
            double cy = (y1Image + y2Image) / 2.0;

            double width = Math.abs(x2Image - x1Image);
            double height = Math.abs(y2Image - y1Image);

            addManualDetection(cx, cy, width, height);

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

            drawDetections((Graphics2D) g);

            if (drawing) {
                drawTemporaryRoi((Graphics2D) g);
            }
        }

        private void drawDetections(Graphics2D g2) {
            int selectedIndex = selectedDetectionIndex;

            for (int i = 0; i < detections.size(); i++) {
                NeuronDetection detection = detections.get(i);

                Color roiColor;

                if (i == selectedIndex) {
                    roiColor = Color.ORANGE;
                } else if (detection.name.startsWith("MANUAL_")) {
                    roiColor = Color.GREEN;
                } else {
                    roiColor = Color.BLUE;
                }

                int x = (int) Math.round(offsetX + (detection.cx - detection.width / 2.0) * zoomFactor);
                int y = (int) Math.round(offsetY + (detection.cy - detection.height / 2.0) * zoomFactor);
                int w = (int) Math.round(detection.width * zoomFactor);
                int h = (int) Math.round(detection.height * zoomFactor);

                g2.setColor(roiColor);
                g2.setStroke(new BasicStroke(i == selectedIndex ? 2.5f : 1.5f));
                g2.drawOval(x, y, w, h);

            }
        }

        private void drawTemporaryRoi(Graphics2D g2) {
            g2.setColor(Color.GREEN);
            g2.setStroke(new BasicStroke(2.0f));

            int x = Math.min(dragStartX, dragCurrentX);
            int y = Math.min(dragStartY, dragCurrentY);
            int w = Math.abs(dragCurrentX - dragStartX);
            int h = Math.abs(dragCurrentY - dragStartY);

            g2.drawOval(x, y, w, h);

        }

        private int findDetectionAt(double imageX, double imageY) {
            for (int i = detections.size() - 1; i >= 0; i--) {
                NeuronDetection detection = detections.get(i);

                if (isPointInsideDetection(imageX, imageY, detection)) {
                    return i;
                }
            }

            return -1;
        }
    }
}