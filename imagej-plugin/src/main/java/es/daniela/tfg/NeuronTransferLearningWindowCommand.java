package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.io.FileSaver;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Locale;
import java.util.UUID;

@Plugin(type = Command.class, menuPath = "Plugins>Neuron Analysis>Transfer Learning Assistant")
public class NeuronTransferLearningWindowCommand implements Command {

    private String pythonExe;
    private String scriptPath;
    private String retrainScriptPath;

    private JFrame frame;

    private DefaultListModel<ImageEntry> imageListModel;
    private JList<ImageEntry> imageList;

    private DefaultListModel<String> roiListModel;
    private JList<String> roiList;

    private JLabel statusLabel;
    private JLabel folderLabel;
    private JLabel modelLabel;
    private JLabel zoomLabel;

    private JScrollPane imageScrollPane;
    private ImagePanel imagePanel;

    private JButton selectFolderButton;
    private JButton detectImagesButton;
    private JButton correctButton;
    private JButton deleteSelectedButton;
    private JButton addNeuronButton;
    private JButton saveToTrainingSetButton;
    private JButton retrainButton;
    private JButton changeModelButton;
    private JButton zoomInButton;
    private JButton zoomOutButton;
    private JButton fitButton;
    private JButton actualSizeButton;

    private File selectedFolder;
    private File workDir;
    private File detectionsDir;
    private File annotationsDir;
    private File yoloDatasetDir;
    private File modelsDir;
    private File activeModelFile;

    private File currentImageFile;
    private ImagePlus sourceImage;
    private ImagePlus imageToDisplay;

    private final List<NeuronDetection> detections = new ArrayList<>();
    private int selectedDetectionIndex = -1;

    private boolean correctionMode = false;
    private boolean addNeuronMode = false;
    private boolean detectionRunning = false;
    private boolean retrainingRunning = false;

    private double zoomFactor = 1.0;
    private boolean fitToPanel = true;

    private final String sessionId = UUID.randomUUID().toString();
    private File sessionDir;

    @Override
    public void run() {
        SwingUtilities.invokeLater(this::createWindow);
    }

    private void createWindow() {
        loadConfiguration();

        sessionDir = new File(
                System.getProperty("java.io.tmpdir"),
                "neuron_transfer_learning_" + sessionId
        );

        if (!sessionDir.exists()) {
            sessionDir.mkdirs();
        }

        frame = new JFrame("Neuron Transfer Learning");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(1200, 760);
        frame.setLayout(new BorderLayout(8, 8));

        frame.add(createLeftPanel(), BorderLayout.WEST);
        frame.add(createCenterPanel(), BorderLayout.CENTER);
        frame.add(createRightPanel(), BorderLayout.EAST);
        frame.add(createBottomPanel(), BorderLayout.SOUTH);

        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosed(java.awt.event.WindowEvent e) {
                if (sessionDir != null) {
                    deleteDirectory(sessionDir);
                }
            }
        });

        updateStatus("Ready. Select an image folder to start.");
        updateButtonState();

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private JPanel createLeftPanel() {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setPreferredSize(new Dimension(260, 0));
        panel.setBorder(BorderFactory.createTitledBorder("Image folder"));

        folderLabel = new JLabel("No folder selected");
        folderLabel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));

        imageListModel = new DefaultListModel<>();
        imageList = new JList<>(imageListModel);
        imageList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        imageList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                ImageEntry entry = imageList.getSelectedValue();

                if (entry != null) {
                    loadImageEntry(entry);
                }
            }
        });

        panel.add(folderLabel, BorderLayout.NORTH);
        panel.add(new JScrollPane(imageList), BorderLayout.CENTER);

        return panel;
    }

    private JPanel createCenterPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setBorder(BorderFactory.createTitledBorder("Review / Correction"));

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
            updateDisplayedImage();
        });

        zoomInButton.addActionListener(e -> {
            fitToPanel = false;
            zoomFactor = Math.min(5.0, zoomFactor + 0.25);
            updateDisplayedImage();
        });

        fitButton.addActionListener(e -> {
            fitToPanel = true;
            updateDisplayedImage();
        });

        actualSizeButton.addActionListener(e -> {
            fitToPanel = false;
            zoomFactor = 1.0;
            updateDisplayedImage();
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
        JPanel wrapper = new JPanel(new BorderLayout());
        wrapper.setPreferredSize(new Dimension(230, 0));

        JPanel actionsPanel = new JPanel();
        actionsPanel.setLayout(new GridLayout(8, 1, 8, 8));
        actionsPanel.setBorder(BorderFactory.createTitledBorder("Actions"));

        selectFolderButton = new JButton("1. Select folder");
        detectImagesButton = new JButton("2. Detect images");
        correctButton = new JButton("3. Correct image");
        deleteSelectedButton = new JButton("3.1 Delete neuron");
        addNeuronButton = new JButton("3.2 Add neurons");
        saveToTrainingSetButton = new JButton("4. Save to training set");
        retrainButton = new JButton("5. Retrain model");
        changeModelButton = new JButton("Change model");

        selectFolderButton.addActionListener(e -> selectFolder());
        detectImagesButton.addActionListener(e -> detectImages());
        correctButton.addActionListener(e -> enableCorrectionMode());
        deleteSelectedButton.addActionListener(e -> deleteSelectedDetection());
        saveToTrainingSetButton.addActionListener(e -> saveToTrainingSet());
        retrainButton.addActionListener(e -> retrainModel());
        changeModelButton.addActionListener(e -> changeActiveModel());

        addNeuronButton.addActionListener(e -> {
            addNeuronMode = !addNeuronMode;

            if (addNeuronMode) {
                addNeuronButton.setText("3.2 Stop adding");
                updateStatus("Add neuron mode enabled. Click and drag to add missing neurons.");
            } else {
                addNeuronButton.setText("3.2 Add neurons");
                updateStatus("Add neuron mode disabled.");
            }

            updateButtonState();
            imagePanel.repaint();
        });

        actionsPanel.add(selectFolderButton);
        actionsPanel.add(detectImagesButton);
        actionsPanel.add(correctButton);
        actionsPanel.add(deleteSelectedButton);
        actionsPanel.add(addNeuronButton);
        actionsPanel.add(saveToTrainingSetButton);
        actionsPanel.add(retrainButton);
        actionsPanel.add(changeModelButton);

        JPanel roiPanel = new JPanel(new BorderLayout());
        roiPanel.setBorder(BorderFactory.createTitledBorder("ROI list"));

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

        roiPanel.add(new JScrollPane(roiList), BorderLayout.CENTER);

        JPanel modelPanel = new JPanel(new BorderLayout());
        modelPanel.setBorder(BorderFactory.createTitledBorder("Model"));
        modelLabel = new JLabel("Base model");
        modelLabel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        modelPanel.add(modelLabel, BorderLayout.CENTER);

        wrapper.add(actionsPanel, BorderLayout.NORTH);
        wrapper.add(roiPanel, BorderLayout.CENTER);
        wrapper.add(modelPanel, BorderLayout.SOUTH);

        return wrapper;
    }

    private JPanel createBottomPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        statusLabel = new JLabel("Ready.");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));
        panel.add(statusLabel, BorderLayout.WEST);
        return panel;
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

            IJ.log("Loaded config file: " + configFile.getAbsolutePath());
            IJ.log("Loaded Python executable: " + pythonExe);
            IJ.log("Loaded inference script: " + scriptPath);
            IJ.log("Loaded retraining script: " + retrainScriptPath);

            if (pythonExe == null || pythonExe.trim().isEmpty()) {
                IJ.error("Configuration error", "Missing property: python");
            }

            if (scriptPath == null || scriptPath.trim().isEmpty()) {
                IJ.error("Configuration error", "Missing property: script");
            }

            if (retrainScriptPath == null || retrainScriptPath.trim().isEmpty()) {
                IJ.error("Configuration error", "Missing property: retrain_script");
            }

        } catch (Exception e) {
            IJ.handleException(e);
        }
    }

    private void selectFolder() {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

        int result = chooser.showOpenDialog(frame);

        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }

        selectedFolder = chooser.getSelectedFile();

        try {
            initializeWorkDirForSelectedFolder();
            loadImageListFromFolder();

            folderLabel.setText(selectedFolder.getName());
            updateModelLabel();
            updateStatus("Folder selected: " + selectedFolder.getAbsolutePath());

        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not load selected folder.");
        }

        updateButtonState();
    }

    private void initializeWorkDirForSelectedFolder() throws IOException {
        workDir = new File(selectedFolder, "neuron_transfer_learning");

        detectionsDir = new File(workDir, "detections");
        annotationsDir = new File(workDir, "annotations");
        yoloDatasetDir = new File(workDir, "yolo_dataset");
        modelsDir = new File(workDir, "models");

        detectionsDir.mkdirs();
        annotationsDir.mkdirs();
        new File(yoloDatasetDir, "images/train").mkdirs();
        new File(yoloDatasetDir, "labels/train").mkdirs();
        modelsDir.mkdirs();

        writeDataYaml(yoloDatasetDir);

        activeModelFile = readActiveModelForFolder();

        if (activeModelFile == null || !activeModelFile.exists()) {
            activeModelFile = getGlobalActiveModelOrBaseModel();
        }
    }

    private void loadImageListFromFolder() {
        imageListModel.clear();

        File[] files = selectedFolder.listFiles();

        if (files == null) {
            return;
        }

        for (File file : files) {
            if (!file.isFile() || !isSupportedImage(file)) {
                continue;
            }

            ImageEntry entry = new ImageEntry(file);
            entry.status = computeImageStatus(file);
            imageListModel.addElement(entry);
        }

        if (!imageListModel.isEmpty()) {
            imageList.setSelectedIndex(0);
        }
    }

    private String computeImageStatus(File imageFile) {
        File latestAnnotation = getLatestAnnotationFile(imageFile);

        if (latestAnnotation != null && latestAnnotation.exists()) {
            return "annotated";
        }

        File detectionFile = getDetectionFile(imageFile);

        if (detectionFile.exists()) {
            return "detected";
        }

        return "pending";
    }

    private void loadImageEntry(ImageEntry entry) {
        try {
            currentImageFile = entry.file;
            sourceImage = IJ.openImage(currentImageFile.getAbsolutePath());

            if (sourceImage == null) {
                throw new RuntimeException("Could not open image: " + currentImageFile.getAbsolutePath());
            }

            imageToDisplay = sourceImage;
            fitToPanel = true;

            detections.clear();
            roiListModel.clear();
            selectedDetectionIndex = -1;
            correctionMode = false;
            addNeuronMode = false;

            File latestAnnotation = getLatestAnnotationFile(currentImageFile);
            File detectionFile = getDetectionFile(currentImageFile);

            if (latestAnnotation != null && latestAnnotation.exists()) {
                loadDetectionsFromYoloLabel(latestAnnotation, sourceImage.getWidth(), sourceImage.getHeight());
                updateStatus("Loaded latest annotation for: " + currentImageFile.getName());
            } else if (detectionFile.exists()) {
                loadDetectionsFromCsv(detectionFile);
                updateStatus("Loaded previous detection for: " + currentImageFile.getName());
            } else {
                updateStatus("Loaded image without detections: " + currentImageFile.getName());
            }

            updateDisplayedImage();
            updateButtonState();

        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not load selected image.");
        }
    }

    private void detectImages() {
        if (workDir == null || selectedFolder == null) {
            IJ.error("Please select a folder first.");
            return;
        }

        if (pythonExe == null || scriptPath == null) {
            IJ.error("Plugin is not configured. Please run the installer first.");
            return;
        }

        detectionRunning = true;
        updateStatus("Detecting images... Please wait.");
        updateButtonState();

        new Thread(() -> {
            try {
                File[] files = selectedFolder.listFiles();

                if (files == null) {
                    throw new RuntimeException("No images found.");
                }

                int processed = 0;

                for (File imageFile : files) {
                    if (!imageFile.isFile() || !isSupportedImage(imageFile)) {
                        continue;
                    }

                    File outputImage = getDetectionPreviewFile(imageFile);
                    File outputCsv = getDetectionFile(imageFile);

                    runPythonDetectionForFile(imageFile, outputImage, outputCsv);
                    processed++;
                }

                final int processedCount = processed;

                SwingUtilities.invokeLater(() -> {
                    detectionRunning = false;
                    loadImageListFromFolder();
                    updateStatus("Detection completed for " + processedCount + " images.");
                    updateButtonState();
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

    private void runPythonDetectionForFile(File imageFile, File outputImage, File expectedCsv) throws Exception {
        deleteIfExists(outputImage);
        deleteIfExists(expectedCsv);

        File pythonFile = new File(pythonExe);
        File scriptFile = new File(scriptPath);

        if (!pythonFile.exists()) {
            throw new RuntimeException("Python executable not found: " + pythonExe);
        }

        if (!scriptFile.exists()) {
            throw new RuntimeException("Inference script not found: " + scriptPath);
        }

        if (sessionDir == null) {
            sessionDir = new File(
                    System.getProperty("java.io.tmpdir"),
                    "neuron_transfer_learning_" + sessionId
            );
        }

        if (!sessionDir.exists() && !sessionDir.mkdirs()) {
            throw new RuntimeException("Could not create session directory: " + sessionDir.getAbsolutePath());
        }

        String baseName = getBaseName(imageFile);

        File tempInputPng = new File(sessionDir, baseName + "_input.png");
        File tempOutputPng = new File(sessionDir, baseName + "_output.png");
        File tempCsv = new File(sessionDir, "imagej_output.csv");

        deleteIfExists(tempInputPng);
        deleteIfExists(tempOutputPng);
        deleteIfExists(tempCsv);

        ImagePlus imp = IJ.openImage(imageFile.getAbsolutePath());

        if (imp == null) {
            throw new RuntimeException("Could not open image with ImageJ: " + imageFile.getAbsolutePath());
        }

        boolean savedAsPng = new FileSaver(imp).saveAsPng(tempInputPng.getAbsolutePath());

        if (!savedAsPng || !tempInputPng.exists()) {
            throw new RuntimeException("Could not save temporary PNG input for: " + imageFile.getName());
        }

        File model = getActiveModelFile();

        ProcessBuilder pb = new ProcessBuilder(
                pythonExe,
                "-u",
                scriptPath,
                tempInputPng.getAbsolutePath(),
                tempOutputPng.getAbsolutePath(),
                model.getAbsolutePath()
        );

        pb.redirectErrorStream(true);

        IJ.log("Running batch inference using temporary PNG created by ImageJ:");
        IJ.log("Original image: " + imageFile.getAbsolutePath());
        IJ.log("Temporary input: " + tempInputPng.getAbsolutePath());
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
            throw new RuntimeException("Python inference failed with exit code " + exitCode);
        }

        if (!tempOutputPng.exists()) {
            throw new RuntimeException("Detection output image was not created: " + tempOutputPng.getAbsolutePath());
        }

        if (!tempCsv.exists()) {
            throw new RuntimeException("Detection CSV was not created for: " + imageFile.getName());
        }

        Files.copy(tempOutputPng.toPath(), outputImage.toPath(), StandardCopyOption.REPLACE_EXISTING);
        Files.copy(tempCsv.toPath(), expectedCsv.toPath(), StandardCopyOption.REPLACE_EXISTING);

        deleteIfExists(tempInputPng);
        deleteIfExists(tempOutputPng);
        deleteIfExists(tempCsv);
    }

    private void enableCorrectionMode() {
        if (sourceImage == null) {
            IJ.error("Please select an image first.");
            return;
        }

        if (detections.isEmpty()) {
            IJ.error("No detections available for this image. Run detection first.");
            return;
        }

        correctionMode = true;
        addNeuronMode = false;
        selectedDetectionIndex = -1;
        roiList.clearSelection();

        addNeuronButton.setText("3.2 Add neurons");

        updateStatus("Correction mode enabled. Select, move, delete or add neurons.");
        updateButtonState();
        imagePanel.repaint();
    }

    private void saveToTrainingSet() {
        if (currentImageFile == null || sourceImage == null) {
            IJ.error("Please select an image first.");
            return;
        }

        if (detections.isEmpty()) {
            IJ.error("There are no detections to save.");
            return;
        }

        try {
            String imageBaseName = getBaseName(currentImageFile);

            File annotationFolder = new File(annotationsDir, imageBaseName);
            annotationFolder.mkdirs();

            int nextVersion = getNextAnnotationVersion(annotationFolder);

            File versionedAnnotation = new File(
                    annotationFolder,
                    String.format(Locale.US, "annotation_%03d.txt", nextVersion)
            );

            writeYoloLabelFile(
                    versionedAnnotation,
                    sourceImage.getWidth(),
                    sourceImage.getHeight()
            );

            writeDataYaml(yoloDatasetDir);

            correctionMode = false;
            addNeuronMode = false;
            selectedDetectionIndex = -1;
            roiList.clearSelection();

            refreshImageStatus(currentImageFile, "annotated");

            updateStatus("Saved corrected annotation: " + currentImageFile.getName());
            updateButtonState();
            imagePanel.repaint();

            IJ.log("Versioned annotation saved to: " + versionedAnnotation.getAbsolutePath());
            IJ.log("Training dataset will be rebuilt automatically before retraining.");

        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not save annotation to training set.");
        }
    }

    private void retrainModel() {
        if (workDir == null || yoloDatasetDir == null) {
            IJ.error("Please select a folder first.");
            return;
        }

        if (pythonExe == null || retrainScriptPath == null) {
            IJ.error("Retraining script is not configured.");
            return;
        }

        File dataYaml = new File(yoloDatasetDir, "data.yaml");

        int annotatedCount;

        try {
            annotatedCount = prepareTrainingSetFromAvailableData();
        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not prepare training set.");
            return;
        }

        if (annotatedCount == 0) {
            IJ.error(
                    "No training data found.\n\n" +
                            "Run detection first, or save at least one corrected annotation."
            );
            return;
        }

        String modelName = JOptionPane.showInputDialog(
                frame,
                "Model name:",
                "retrained_model_" + System.currentTimeMillis()
        );

        if (modelName == null || modelName.trim().isEmpty()) {
            return;
        }

        modelName = sanitizeFileName(modelName.trim());

        if (!modelName.endsWith(".pt")) {
            modelName += ".pt";
        }

        File outputModel = new File(modelsDir, modelName);
        File startModel = getActiveModelFile();

        retrainingRunning = true;
        updateStatus("Retraining with " + annotatedCount + " available labelled images...");
        updateButtonState();

        String finalModelName = modelName;

        new Thread(() -> {
            try {
                ProcessBuilder pb = new ProcessBuilder(
                        pythonExe,
                        "-u",
                        retrainScriptPath,
                        dataYaml.getAbsolutePath(),
                        outputModel.getAbsolutePath(),
                        startModel.getAbsolutePath()
                );

                pb.redirectErrorStream(true);

                IJ.log("Running transfer learning:");
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

                writeFolderActiveModel(outputModel);
                activeModelFile = outputModel;

                int setGlobal = JOptionPane.showConfirmDialog(
                        frame,
                        "Model generated:\n" + outputModel.getAbsolutePath() +
                                "\n\nSet this model as the global active model for future individual detections?",
                        "Set active model",
                        JOptionPane.YES_NO_OPTION
                );

                if (setGlobal == JOptionPane.YES_OPTION) {
                    writeGlobalActiveModel(outputModel);
                }

                SwingUtilities.invokeLater(() -> {
                    retrainingRunning = false;
                    updateModelLabel();
                    updateStatus("Retraining completed: " + finalModelName);
                    updateButtonState();

                    IJ.showMessage(
                            "Retraining completed",
                            "Model saved to:\n" + outputModel.getAbsolutePath()
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

    private int prepareTrainingSetFromAvailableData() throws IOException {
        if (selectedFolder == null || yoloDatasetDir == null) {
            return 0;
        }

        File yoloImagesTrain = new File(yoloDatasetDir, "images/train");
        File yoloLabelsTrain = new File(yoloDatasetDir, "labels/train");

        yoloImagesTrain.mkdirs();
        yoloLabelsTrain.mkdirs();

        clearDirectory(yoloImagesTrain);
        clearDirectory(yoloLabelsTrain);

        File[] files = selectedFolder.listFiles();

        if (files == null) {
            return 0;
        }

        int count = 0;

        for (File imageFile : files) {
            if (!imageFile.isFile() || !isSupportedImage(imageFile)) {
                continue;
            }

            String imageBaseName = getBaseName(imageFile);

            File latestAnnotation = getLatestAnnotationFile(imageFile);
            File detectionFile = getDetectionFile(imageFile);

            File yoloImage = new File(yoloImagesTrain, imageBaseName + ".png");
            File yoloLabel = new File(yoloLabelsTrain, imageBaseName + ".txt");

            if (latestAnnotation != null && latestAnnotation.exists()) {
                createTrainingImageForYolo(imageFile, yoloImage);
                Files.copy(latestAnnotation.toPath(), yoloLabel.toPath(), StandardCopyOption.REPLACE_EXISTING);

                IJ.log("Training sample from corrected annotation: " + imageFile.getName());
                count++;
                continue;
            }

            if (detectionFile.exists()) {
                ImagePlus imp = IJ.openImage(imageFile.getAbsolutePath());

                if (imp == null) {
                    IJ.log("Skipping image because ImageJ could not open it: " + imageFile.getAbsolutePath());
                    continue;
                }

                try {
                    List<NeuronDetection> automaticDetections = readDetectionsFromCsv(detectionFile);

                    if (automaticDetections.isEmpty()) {
                        IJ.log("Skipping image without detections: " + imageFile.getName());
                        continue;
                    }

                    createTrainingImageForYolo(imageFile, yoloImage);

                    writeYoloLabelFile(
                            yoloLabel,
                            imp.getWidth(),
                            imp.getHeight(),
                            automaticDetections
                    );

                    IJ.log("Training sample from automatic detection: " + imageFile.getName());
                    count++;

                } finally {
                    imp.close();
                }
            }
        }

        writeDataYaml(yoloDatasetDir);

        IJ.log("Prepared training set with " + count + " images.");
        return count;
    }

    private List<NeuronDetection> readDetectionsFromCsv(File csvFile) throws IOException {
        List<NeuronDetection> result = new ArrayList<>();

        if (!csvFile.exists()) {
            return result;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line = br.readLine();
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

                result.add(new NeuronDetection(
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

        return result;
    }

    private void createTrainingImageForYolo(File sourceImageFile, File targetImageFile) throws IOException {
        Files.deleteIfExists(targetImageFile.toPath());

        ImagePlus imp = IJ.openImage(sourceImageFile.getAbsolutePath());

        if (imp == null) {
            throw new IOException("Could not open image for training: " + sourceImageFile.getAbsolutePath());
        }

        try {
            targetImageFile.getParentFile().mkdirs();

            boolean saved = new FileSaver(imp).saveAsPng(targetImageFile.getAbsolutePath());

            if (!saved || !targetImageFile.exists() || targetImageFile.length() == 0) {
                throw new IOException("Could not save training PNG: " + targetImageFile.getAbsolutePath());
            }

            IJ.log("Created normalized training PNG: " + targetImageFile.getAbsolutePath());

        } finally {
            imp.close();
        }
    }

    private void clearDirectory(File directory) throws IOException {
        if (directory == null || !directory.exists()) {
            return;
        }

        File[] files = directory.listFiles();

        if (files == null) {
            return;
        }

        for (File file : files) {
            if (file.isDirectory()) {
                clearDirectory(file);

                if (!file.delete()) {
                    IJ.log("Warning: could not delete directory: " + file.getAbsolutePath());
                }
            } else {
                Files.deleteIfExists(file.toPath());
            }
        }
    }

    private void changeActiveModel() {
        JFileChooser chooser = new JFileChooser();

        if (modelsDir != null && modelsDir.exists()) {
            chooser.setCurrentDirectory(modelsDir);
        }

        int result = chooser.showOpenDialog(frame);

        if (result != JFileChooser.APPROVE_OPTION) {
            return;
        }

        File selectedModel = chooser.getSelectedFile();

        if (!selectedModel.exists() || !selectedModel.getName().endsWith(".pt")) {
            IJ.error("Please select a valid .pt model file.");
            return;
        }

        activeModelFile = selectedModel;

        try {
            if (workDir != null) {
                writeFolderActiveModel(selectedModel);
            }

            updateModelLabel();
            updateStatus("Active model changed: " + selectedModel.getName());

        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not change active model.");
        }
    }

    private File getActiveModelFile() {
        if (activeModelFile != null && activeModelFile.exists()) {
            return activeModelFile;
        }

        activeModelFile = getGlobalActiveModelOrBaseModel();
        return activeModelFile;
    }

    private File getGlobalActiveModelOrBaseModel() {
        File globalActive = new File(
                System.getProperty("user.home"),
                ".neuron-segmentation-assistant/active_model.txt"
        );

        if (globalActive.exists()) {
            try {
                String path = new String(Files.readAllBytes(globalActive.toPath())).trim();
                File model = new File(path);

                if (model.exists()) {
                    return model;
                }

            } catch (Exception e) {
                IJ.log("Could not read global active model. Falling back to base model.");
            }
        }

        if (scriptPath == null || scriptPath.trim().isEmpty()) {
            IJ.error(
                    "Configuration error",
                    "Inference script path is missing.\n\n" +
                            "Please check this file:\n" +
                            new File(
                                    System.getProperty("user.home"),
                                    ".neuron-segmentation-assistant/config.properties"
                            ).getAbsolutePath() +
                            "\n\nIt must contain a line like:\n" +
                            "script=/path/to/infer_one.py"
            );

            return null;
        }

        File scriptFile = new File(scriptPath);

        if (!scriptFile.exists()) {
            IJ.error(
                    "Configuration error",
                    "Inference script not found:\n" + scriptFile.getAbsolutePath()
            );

            return null;
        }

        File pythonRoot = scriptFile.getParentFile();

        if (pythonRoot == null) {
            IJ.error("Configuration error", "Could not resolve inference script folder.");
            return null;
        }

        return new File(pythonRoot, "models/best.pt");
    }

    private File readActiveModelForFolder() {
        if (workDir == null) {
            return null;
        }

        File file = new File(workDir, "active_model.txt");

        if (!file.exists()) {
            return null;
        }

        try {
            String path = new String(Files.readAllBytes(file.toPath())).trim();
            File model = new File(path);

            if (model.exists()) {
                return model;
            }

        } catch (Exception e) {
            IJ.log("Could not read folder active model.");
        }

        return null;
    }

    private void writeFolderActiveModel(File modelFile) throws IOException {
        File activeFile = new File(workDir, "active_model.txt");

        try (PrintWriter writer = new PrintWriter(new FileWriter(activeFile))) {
            writer.println(modelFile.getAbsolutePath());
        }
    }

    private void writeGlobalActiveModel(File modelFile) throws IOException {
        File activeFile = new File(
                System.getProperty("user.home"),
                ".neuron-segmentation-assistant/active_model.txt"
        );

        activeFile.getParentFile().mkdirs();

        try (PrintWriter writer = new PrintWriter(new FileWriter(activeFile))) {
            writer.println(modelFile.getAbsolutePath());
        }
    }

    private void updateModelLabel() {
        File model = getActiveModelFile();

        if (model != null && model.exists()) {
            modelLabel.setText(model.getName());
        } else {
            modelLabel.setText("No model selected");
        }
    }

    private File getDetectionPreviewFile(File imageFile) {
        return new File(detectionsDir, getBaseName(imageFile) + "_preview.png");
    }

    private File getDetectionFile(File imageFile) {
        return new File(detectionsDir, getBaseName(imageFile) + ".csv");
    }

    private File getLatestAnnotationFile(File imageFile) {
        if (annotationsDir == null) {
            return null;
        }

        File annotationFolder = new File(annotationsDir, getBaseName(imageFile));

        if (!annotationFolder.exists()) {
            return null;
        }

        File[] files = annotationFolder.listFiles((dir, name) ->
                name.startsWith("annotation_") && name.endsWith(".txt")
        );

        if (files == null || files.length == 0) {
            return null;
        }

        File latest = files[0];

        for (File file : files) {
            if (file.getName().compareTo(latest.getName()) > 0) {
                latest = file;
            }
        }

        return latest;
    }

    private int getNextAnnotationVersion(File annotationFolder) {
        File[] files = annotationFolder.listFiles((dir, name) ->
                name.startsWith("annotation_") && name.endsWith(".txt")
        );

        if (files == null || files.length == 0) {
            return 1;
        }

        int max = 0;

        for (File file : files) {
            String name = file.getName()
                    .replace("annotation_", "")
                    .replace(".txt", "");

            try {
                int value = Integer.parseInt(name);
                max = Math.max(max, value);
            } catch (NumberFormatException ignored) {
            }
        }

        return max + 1;
    }

    private void refreshImageStatus(File imageFile, String status) {
        for (int i = 0; i < imageListModel.size(); i++) {
            ImageEntry entry = imageListModel.getElementAt(i);

            if (entry.file.equals(imageFile)) {
                entry.status = status;
                imageListModel.set(i, entry);
                imageList.setSelectedIndex(i);
                break;
            }
        }
    }

    private boolean isSupportedImage(File file) {
        String name = file.getName().toLowerCase(Locale.ROOT);

        return name.endsWith(".png") ||
                name.endsWith(".jpg") ||
                name.endsWith(".jpeg") ||
                name.endsWith(".tif") ||
                name.endsWith(".tiff");
    }

    private String getBaseName(File file) {
        String name = file.getName();
        int dot = name.lastIndexOf(".");

        if (dot > 0) {
            name = name.substring(0, dot);
        }

        return sanitizeFileName(name);
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[^a-zA-Z0-9._-]", "_");
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private void updateStatus(String message) {
        int count = detections.size();

        if (count > 0) {
            statusLabel.setText("Neurons: " + count + " | " + message);
        } else {
            statusLabel.setText(message);
        }

        IJ.log("[Transfer Learning] " + message);
    }

    private void updateButtonState() {
        boolean hasFolder = workDir != null;
        boolean hasImage = sourceImage != null;
        boolean hasDetections = !detections.isEmpty();
        boolean busy = detectionRunning || retrainingRunning;

        selectFolderButton.setEnabled(!busy);
        detectImagesButton.setEnabled(hasFolder && !busy);
        correctButton.setEnabled(hasImage && hasDetections && !correctionMode && !busy);
        deleteSelectedButton.setEnabled(correctionMode && hasDetections && !busy);
        addNeuronButton.setEnabled(correctionMode && !busy);
        saveToTrainingSetButton.setEnabled(hasImage && hasDetections && !addNeuronMode && !busy);
        retrainButton.setEnabled(hasFolder && !busy);
        changeModelButton.setEnabled(!busy);
    }

    private void updateDisplayedImage() {
        if (imageToDisplay == null) {
            return;
        }

        Image awtImage = imageToDisplay.getImage();

        int originalWidth = imageToDisplay.getWidth();
        int originalHeight = imageToDisplay.getHeight();

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

    private void loadDetectionsFromCsv(File csvFile) throws Exception {
        detections.clear();
        detections.addAll(readDetectionsFromCsv(csvFile));
        refreshRoiList();
    }

    private void loadDetectionsFromYoloLabel(File labelFile, int imageWidth, int imageHeight) throws Exception {
        detections.clear();

        try (BufferedReader br = new BufferedReader(new FileReader(labelFile))) {
            String line;
            int index = 1;

            while ((line = br.readLine()) != null) {
                String[] parts = line.trim().split("\\s+");

                if (parts.length < 7) {
                    continue;
                }

                List<Double> xs = new ArrayList<>();
                List<Double> ys = new ArrayList<>();

                for (int i = 1; i + 1 < parts.length; i += 2) {
                    double x = Double.parseDouble(parts[i]) * imageWidth;
                    double y = Double.parseDouble(parts[i + 1]) * imageHeight;

                    xs.add(x);
                    ys.add(y);
                }

                if (xs.isEmpty()) {
                    continue;
                }

                double minX = xs.stream().mapToDouble(v -> v).min().orElse(0);
                double maxX = xs.stream().mapToDouble(v -> v).max().orElse(0);
                double minY = ys.stream().mapToDouble(v -> v).min().orElse(0);
                double maxY = ys.stream().mapToDouble(v -> v).max().orElse(0);

                double cx = (minX + maxX) / 2.0;
                double cy = (minY + maxY) / 2.0;
                double width = Math.max(4, maxX - minX);
                double height = Math.max(4, maxY - minY);

                detections.add(new NeuronDetection(
                        "ANNOTATION_" + index,
                        cx,
                        cy,
                        width,
                        height,
                        null,
                        true
                ));

                index++;
            }
        }

        refreshRoiList();
    }

    private void refreshRoiList() {
        roiListModel.clear();

        for (NeuronDetection detection : detections) {
            roiListModel.addElement(detection.name);
        }

        selectedDetectionIndex = -1;
        roiList.clearSelection();

        if (imagePanel != null) {
            imagePanel.repaint();
        }
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

    private void writeYoloLabelFile(
            File labelFile,
            int imageWidth,
            int imageHeight,
            List<NeuronDetection> detectionsToWrite
    ) throws IOException {
        labelFile.getParentFile().mkdirs();

        try (PrintWriter writer = new PrintWriter(new FileWriter(labelFile))) {
            for (NeuronDetection detection : detectionsToWrite) {
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

    private void writeYoloLabelFile(File labelFile, int imageWidth, int imageHeight) throws IOException {
        labelFile.getParentFile().mkdirs();

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

    private void deleteSelectedDetection() {
        int selectedIndex = selectedDetectionIndex;

        if (selectedIndex < 0 || selectedIndex >= detections.size()) {
            selectedIndex = roiList.getSelectedIndex();
        }

        if (selectedIndex < 0 || selectedIndex >= detections.size()) {
            IJ.showMessage(
                    "Delete neuron",
                    "Please select a neuron first."
            );
            return;
        }

        detections.remove(selectedIndex);
        selectedDetectionIndex = -1;
        refreshAfterEditing();
        updateStatus("Deleted selected neuron.");
    }

    private void refreshAfterEditing() {
        roiListModel.clear();

        for (NeuronDetection detection : detections) {
            roiListModel.addElement(detection.name);
        }

        roiList.clearSelection();

        if (imagePanel != null) {
            imagePanel.repaint();
        }

        updateButtonState();
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
        updateStatus("Manual neuron added.");
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
        }
    }

    private static class ImageEntry {
        private final File file;
        private String status = "pending";

        private ImageEntry(File file) {
            this.file = file;
        }

        @Override
        public String toString() {
            return file.getName() + " [" + status + "]";
        }
    }

    private static class NeuronDetection {
        private String name;
        private double cx;
        private double cy;
        private double width;
        private double height;
        private String maskPolygon;
        private boolean edited;

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

                        repaint();
                        return;
                    }

                    int clickedIndex = findDetectionAt(imageX, imageY);

                    if (clickedIndex >= 0) {
                        selectedDetectionIndex = clickedIndex;
                        roiList.setSelectedIndex(clickedIndex);
                        roiList.ensureIndexIsVisible(clickedIndex);

                        movingSelected = true;
                        lastMoveImageX = imageX;
                        lastMoveImageY = imageY;

                        repaint();
                    } else {
                        selectedDetectionIndex = -1;
                        roiList.clearSelection();
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
                g.drawString("Select a folder and choose an image.", 30, 40);
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
                } else if (detection.name.startsWith("ANNOTATION_")) {
                    roiColor = Color.MAGENTA;
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