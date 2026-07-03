package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.io.FileSaver;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.Polygon;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Locale;
import java.util.UUID;
import java.text.SimpleDateFormat;
import java.util.Date;

@Plugin(type = Command.class, menuPath = "Plugins>Neuron Segmentation>Transfer Learning Assistant")
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
    private JTextField modelLabel;
    private JLabel zoomLabel;

    private JTextArea logTextArea;
    private File logFile;
    private PrintWriter logWriter;

    private JScrollPane imageScrollPane;
    private ImagePanel imagePanel;

    private JButton selectFolderButton;
    private JButton detectImagesButton;
    private JButton correctButton;
    private JButton deleteSelectedButton;
    private JButton addNeuronButton;
    private JButton undoButton;
    private JButton saveToTrainingSetButton;
    private JButton retrainButton;
    private JButton changeModelButton;
    private JCheckBox useHardwareAccelerationCheckBox;
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
    private final List<Integer> selectedDetectionIndices = new ArrayList<>();
    private List<NeuronDetection> undoState = null;

    private boolean correctionMode = false;
    private boolean addNeuronMode = false;
    private boolean detectionRunning = false;
    private boolean retrainingRunning = false;
    private boolean useHardwareAccelerationForRetraining = false;

    private double zoomFactor = 1.0;
    private boolean fitToPanel = true;

    private final String sessionId = UUID.randomUUID().toString();
    private File sessionDir;

    private static final Color COLOR_AUTOMATIC = new Color(0, 170, 255);      // automatic YOLO detections
    private static final Color COLOR_MANUAL = new Color(255, 190, 0);         // user-added neurons
    private static final Color COLOR_ANNOTATION = new Color(190, 90, 255);    // saved corrected annotations
    private static final Color COLOR_SELECTED = new Color(255, 80, 80);       // selected ROI
    private static final Color COLOR_LASSO = new Color(255, 140, 0);          // lasso selection
    private static final Color COLOR_TEMPORARY = new Color(0, 220, 160);      // ROI being drawn
    private static final Color COLOR_OUTLINE_SHADOW = new Color(0, 0, 0, 100);

    private static final String CONFIG_DIR_NAME = ".neuron-segmentation-assistant";
    private static final String CONFIG_FILE_NAME = "config.properties";
    private static final String CONFIG_USE_HARDWARE_ACCELERATION = "use_hardware_acceleration";

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

        frame = new JFrame("Neuron Transfer Learning Assistant");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(1250, 850);
        frame.setLayout(new BorderLayout(8, 8));

        JSplitPane leftCenterSplit = new JSplitPane(
                JSplitPane.HORIZONTAL_SPLIT,
                createLeftPanel(),
                createCenterPanel()
        );

        leftCenterSplit.setResizeWeight(0.25);
        leftCenterSplit.setDividerLocation(340);
        leftCenterSplit.setOneTouchExpandable(true);

        JSplitPane mainSplit = new JSplitPane(
                JSplitPane.HORIZONTAL_SPLIT,
                leftCenterSplit,
                createRightPanel()
        );

        mainSplit.setResizeWeight(0.82);
        mainSplit.setDividerLocation(970);
        mainSplit.setOneTouchExpandable(true);

        JSplitPane verticalSplit = new JSplitPane(
                JSplitPane.VERTICAL_SPLIT,
                mainSplit,
                createBottomPanel()
        );

        verticalSplit.setResizeWeight(0.85);
        verticalSplit.setDividerLocation(650);
        verticalSplit.setOneTouchExpandable(true);

        frame.add(verticalSplit, BorderLayout.CENTER);

        frame.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosed(java.awt.event.WindowEvent e) {
                appendLog("Closing transfer learning assistant.");

                if (sessionDir != null) {
                    deleteDirectory(sessionDir);
                }

                closeLogFile();
            }
        });

        updateStatus("Ready. Select an image folder to start.");
        updateButtonState();

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private JPanel createLeftPanel() {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setPreferredSize(new Dimension(340, 0));
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
        JPanel wrapper = new JPanel(new BorderLayout(6, 6));
        wrapper.setPreferredSize(new Dimension(285, 0));
        wrapper.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));

        JPanel topPanel = new JPanel();
        topPanel.setLayout(new BoxLayout(topPanel, BoxLayout.Y_AXIS));

        JPanel infoPanel = new JPanel(new BorderLayout(4, 4));
        infoPanel.setBorder(BorderFactory.createTitledBorder("Mode"));

        JLabel infoLabel = new JLabel(
                "<html>" +
                        "Process a full image folder.<br><br>" +
                        "Detect neurons, review corrections, save annotations " +
                        "and retrain the active model." +
                        "</html>"
        );

        infoLabel.setFont(infoLabel.getFont().deriveFont(Font.PLAIN, 11f));
        infoLabel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));

        infoPanel.add(infoLabel, BorderLayout.CENTER);

        JPanel modelPanel = new JPanel(new BorderLayout(4, 4));
        modelPanel.setBorder(BorderFactory.createTitledBorder("Active model"));

        modelLabel = new JTextField("No model selected");
        modelLabel.setEditable(false);
        modelLabel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        modelLabel.setToolTipText("Current model used for folder detection and retraining");
        modelLabel.setHorizontalAlignment(JTextField.LEFT);

        changeModelButton = new JButton("Change model");
        styleActionButton(changeModelButton, false);
        changeModelButton.addActionListener(e -> changeActiveModel());

        useHardwareAccelerationCheckBox = new JCheckBox("Use hardware acceleration");
        useHardwareAccelerationCheckBox.setSelected(useHardwareAccelerationForRetraining);
        useHardwareAccelerationCheckBox.setFont(useHardwareAccelerationCheckBox.getFont().deriveFont(Font.PLAIN, 11f));
        useHardwareAccelerationCheckBox.setToolTipText(
                "If enabled, retraining will request the best available hardware acceleration."
        );

        useHardwareAccelerationCheckBox.addActionListener(e -> {
            saveHardwareAccelerationPreference(useHardwareAccelerationCheckBox.isSelected());

            if (useHardwareAccelerationCheckBox.isSelected()) {
                updateStatus("Hardware acceleration enabled for retraining.");
            } else {
                updateStatus("CPU retraining selected.");
            }
        });

        JPanel modelBottomPanel = new JPanel();
        modelBottomPanel.setLayout(new BoxLayout(modelBottomPanel, BoxLayout.Y_AXIS));

        changeModelButton.setAlignmentX(Component.CENTER_ALIGNMENT);
        useHardwareAccelerationCheckBox.setAlignmentX(Component.LEFT_ALIGNMENT);

        modelBottomPanel.add(changeModelButton);
        modelBottomPanel.add(Box.createVerticalStrut(4));
        modelBottomPanel.add(useHardwareAccelerationCheckBox);

        modelPanel.add(modelLabel, BorderLayout.CENTER);
        modelPanel.add(modelBottomPanel, BorderLayout.SOUTH);

        JPanel actionsPanel = new JPanel();
        actionsPanel.setLayout(new GridLayout(5, 1, 8, 8));
        actionsPanel.setBorder(BorderFactory.createTitledBorder("Actions"));

        selectFolderButton = new JButton("1. Select folder");
        detectImagesButton = new JButton("2. Detect images");
        correctButton = new JButton("3. Correct image");
        saveToTrainingSetButton = new JButton("4. Save corrections");
        retrainButton = new JButton("5. Retrain model");

        styleActionButton(selectFolderButton, false);
        styleActionButton(detectImagesButton, false);
        styleActionButton(correctButton, false);
        styleActionButton(saveToTrainingSetButton, false);
        styleActionButton(retrainButton, true);

        selectFolderButton.addActionListener(e -> selectFolder());
        detectImagesButton.addActionListener(e -> detectImages());
        correctButton.addActionListener(e -> enableCorrectionMode());
        saveToTrainingSetButton.addActionListener(e -> saveToTrainingSet());
        retrainButton.addActionListener(e -> retrainModel());

        actionsPanel.add(selectFolderButton);
        actionsPanel.add(detectImagesButton);
        actionsPanel.add(correctButton);
        actionsPanel.add(saveToTrainingSetButton);
        actionsPanel.add(retrainButton);

        topPanel.add(infoPanel);
        topPanel.add(Box.createVerticalStrut(8));
        topPanel.add(modelPanel);
        topPanel.add(Box.createVerticalStrut(8));
        topPanel.add(actionsPanel);

        JPanel neuronPanel = new JPanel(new BorderLayout());
        neuronPanel.setBorder(BorderFactory.createTitledBorder("Neuron list"));

        roiListModel = new DefaultListModel<>();
        roiList = new JList<>(roiListModel);
        roiList.setSelectionMode(ListSelectionModel.MULTIPLE_INTERVAL_SELECTION);

        roiList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                selectedDetectionIndices.clear();

                for (int index : roiList.getSelectedIndices()) {
                    if (index >= 0 && index < detections.size()) {
                        selectedDetectionIndices.add(index);
                    }
                }

                if (imagePanel != null) {
                    imagePanel.repaint();
                }

                updateButtonState();
            }
        });

        deleteSelectedButton = new JButton("Delete");
        addNeuronButton = new JButton("Add");
        undoButton = new JButton("↶ Undo");

        styleActionButton(deleteSelectedButton, false);
        styleActionButton(addNeuronButton, false);
        styleActionButton(undoButton, false);

        deleteSelectedButton.addActionListener(e -> deleteSelectedDetection());
        undoButton.addActionListener(e -> undoLastChange());

        addNeuronButton.addActionListener(e -> {
            addNeuronMode = !addNeuronMode;

            if (addNeuronMode) {
                addNeuronButton.setText("Stop add");
                updateStatus("Add neuron mode enabled. Click and drag to add missing neurons.");
            } else {
                addNeuronButton.setText("Add");
                updateStatus("Add neuron mode disabled.");
            }

            updateButtonState();
            imagePanel.repaint();
        });

        JPanel neuronActionsPanel = new JPanel(new GridLayout(1, 3, 4, 4));
        neuronActionsPanel.setBorder(BorderFactory.createEmptyBorder(2, 2, 4, 2));

        neuronActionsPanel.add(deleteSelectedButton);
        neuronActionsPanel.add(addNeuronButton);
        neuronActionsPanel.add(undoButton);

        neuronPanel.add(neuronActionsPanel, BorderLayout.NORTH);
        neuronPanel.add(new JScrollPane(roiList), BorderLayout.CENTER);

        wrapper.add(topPanel, BorderLayout.NORTH);
        wrapper.add(neuronPanel, BorderLayout.CENTER);

        return wrapper;
    }

    private void saveHardwareAccelerationPreference(boolean useHardwareAcceleration) {
        useHardwareAccelerationForRetraining = useHardwareAcceleration;

        File configFile = new File(
                System.getProperty("user.home"),
                CONFIG_DIR_NAME + "/" + CONFIG_FILE_NAME
        );

        try {
            Properties properties = new Properties();

            if (configFile.exists()) {
                try (FileInputStream fis = new FileInputStream(configFile)) {
                    properties.load(fis);
                }
            }

            properties.setProperty(
                    CONFIG_USE_HARDWARE_ACCELERATION,
                    Boolean.toString(useHardwareAcceleration)
            );

            configFile.getParentFile().mkdirs();

            try (FileOutputStream fos = new FileOutputStream(configFile)) {
                properties.store(fos, "Neuron Segmentation Assistant configuration");
            }

            appendLog("Hardware acceleration preference saved: " + useHardwareAcceleration);

        } catch (Exception e) {
            logError("Could not save hardware acceleration preference: " + e.getMessage());
            IJ.handleException(e);
        }
    }

    private void styleActionButton(JButton button, boolean primary) {
        button.setAlignmentX(Component.CENTER_ALIGNMENT);
        button.setMaximumSize(new Dimension(Integer.MAX_VALUE, 34));
        button.setFocusPainted(false);
        button.setMargin(new Insets(6, 10, 6, 10));

        if (primary) {
            button.setFont(button.getFont().deriveFont(Font.BOLD));
        }
    }

    private JPanel createBottomPanel() {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setBorder(BorderFactory.createEmptyBorder(4, 6, 4, 6));

        statusLabel = new JLabel("Ready.");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(4, 6, 4, 6));

        logTextArea = new JTextArea(4, 80);
        logTextArea.setEditable(false);
        logTextArea.setLineWrap(false);
        logTextArea.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 11));

        JScrollPane logScrollPane = new JScrollPane(logTextArea);
        logScrollPane.setBorder(BorderFactory.createTitledBorder("Log"));

        panel.add(statusLabel, BorderLayout.NORTH);
        panel.add(logScrollPane, BorderLayout.CENTER);

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

            useHardwareAccelerationForRetraining = Boolean.parseBoolean(
                    properties.getProperty(CONFIG_USE_HARDWARE_ACCELERATION, "false")
            );

            appendLog("Loaded config file: " + configFile.getAbsolutePath());
            appendLog("Loaded Python executable: " + pythonExe);
            appendLog("Loaded inference script: " + scriptPath);
            appendLog("Loaded retraining script: " + retrainScriptPath);

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
            logError("Could not load configuration: " + e.getMessage());
            IJ.handleException(e);
        }
    }

    private boolean isDetectionSelected(int index) {
        return selectedDetectionIndices.contains(index);
    }

    private void clearDetectionSelection() {
        selectedDetectionIndices.clear();

        if (roiList != null) {
            roiList.clearSelection();
        }

        if (imagePanel != null) {
            imagePanel.repaint();
        }
    }

    private void setSingleDetectionSelection(int index) {
        selectedDetectionIndices.clear();

        if (index >= 0 && index < detections.size()) {
            selectedDetectionIndices.add(index);
            roiList.setSelectedIndex(index);
            roiList.ensureIndexIsVisible(index);
        } else {
            roiList.clearSelection();
        }

        if (imagePanel != null) {
            imagePanel.repaint();
        }
    }

    private void toggleDetectionSelection(int index) {
        if (index < 0 || index >= detections.size()) {
            return;
        }

        if (selectedDetectionIndices.contains(index)) {
            selectedDetectionIndices.remove(Integer.valueOf(index));
            roiList.removeSelectionInterval(index, index);
        } else {
            selectedDetectionIndices.add(index);
            roiList.addSelectionInterval(index, index);
            roiList.ensureIndexIsVisible(index);
        }

        if (imagePanel != null) {
            imagePanel.repaint();
        }

        updateButtonState();
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
            initializeLogFile();

            appendLog("Loaded Python executable: " + pythonExe);
            appendLog("Loaded inference script: " + scriptPath);
            appendLog("Loaded retraining script: " + retrainScriptPath);

            loadImageListFromFolder();

            folderLabel.setText(selectedFolder.getName());
            updateModelLabel();
            updateStatus("Folder selected: " + selectedFolder.getAbsolutePath());

        } catch (Exception e) {
            logError("Could not load selected folder: " + e.getMessage());
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
            selectedDetectionIndices.clear();
            undoState = null;
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
            logError("Could not load selected image: " + e.getMessage());
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

        File activeModel = getActiveModelFile();

        if (activeModel == null || !activeModel.exists()) {
            logError("No active model found. Detection cancelled.");
            IJ.error("No active model found. Please check the configuration or select a valid model.");
            detectionRunning = false;
            updateButtonState();
            return;
        }

        appendLogSeparator();
        appendLog("Detection process started.");
        appendLog("Active model: " + activeModel.getAbsolutePath());
        appendLog("Input folder: " + selectedFolder.getAbsolutePath());
        appendLog("Detections folder: " + detectionsDir.getAbsolutePath());
        appendLogSeparator();

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
                logError("Detection failed: " + e.getMessage());
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

        try {
            boolean savedAsPng = new FileSaver(imp).saveAsPng(tempInputPng.getAbsolutePath());

            if (!savedAsPng || !tempInputPng.exists()) {
                throw new RuntimeException("Could not save temporary PNG input for: " + imageFile.getName());
            }
        } finally {
            imp.close();
        }

        File model = getActiveModelFile();

        if (model == null || !model.exists()) {
            throw new RuntimeException("No valid active model found for detection.");
        }

        ProcessBuilder pb = new ProcessBuilder(
                pythonExe,
                "-u",
                scriptPath,
                tempInputPng.getAbsolutePath(),
                tempOutputPng.getAbsolutePath(),
                model.getAbsolutePath()
        );

        pb.redirectErrorStream(true);

        appendLog("Running batch inference using temporary PNG created by ImageJ:");
        appendLog("Original image: " + imageFile.getAbsolutePath());
        appendLog("Temporary input: " + tempInputPng.getAbsolutePath());
        appendLog(String.join(" ", pb.command()));

        Process process = pb.start();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {

            String line;

            while ((line = reader.readLine()) != null) {
                logPython(line);
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
        selectedDetectionIndices.clear();
        roiList.clearSelection();

        addNeuronButton.setText("Add");

        updateStatus("Correction mode enabled. Select, move, delete or add neurons.");
        updateButtonState();
        imagePanel.repaint();
    }

    private void saveToTrainingSet() {
        if (!correctionMode) {
            IJ.error("Enable correction mode before saving corrections.");
            return;
        }

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
            undoState = null;
            selectedDetectionIndices.clear();
            roiList.clearSelection();

            addNeuronButton.setText("Add");

            refreshImageStatus(currentImageFile, "annotated");

            updateStatus("Saved corrected annotation: " + currentImageFile.getName());
            updateButtonState();
            imagePanel.repaint();

            appendLog("Versioned annotation saved to: " + versionedAnnotation.getAbsolutePath());
            appendLog("Training dataset will be rebuilt automatically before retraining.");

        } catch (Exception e) {
            logError("Could not save annotation to training set: " + e.getMessage());
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

        if (startModel == null || !startModel.exists()) {
            logError("No valid starting model found. Retraining cancelled.");
            IJ.error("No valid starting model found. Please check the configuration or select a valid model.");
            return;
        }

        appendLogSeparator();
        appendLog("Retraining process requested.");
        appendLog("Training data YAML: " + dataYaml.getAbsolutePath());
        appendLog("Starting model: " + startModel.getAbsolutePath());
        appendLog("Output model: " + outputModel.getAbsolutePath());
        appendLog("Training samples: " + annotatedCount);
        appendLogSeparator();

        retrainingRunning = true;
        updateStatus("Retraining with " + annotatedCount + " available labelled images...");
        updateButtonState();

        boolean useHardwareAcceleration =
                useHardwareAccelerationCheckBox != null && useHardwareAccelerationCheckBox.isSelected();

        String finalModelName = modelName;

        new Thread(() -> {
            try {
                List<String> command = new ArrayList<>();

                command.add(pythonExe);
                command.add("-u");
                command.add(retrainScriptPath);
                command.add(dataYaml.getAbsolutePath());
                command.add(outputModel.getAbsolutePath());
                command.add(startModel.getAbsolutePath());

                String selectedDevice;

                if (useHardwareAcceleration) {
                    selectedDevice = "auto_acceleration";
                } else {
                    selectedDevice = "cpu";
                }

                command.add(selectedDevice);

                ProcessBuilder pb = new ProcessBuilder(command);

                pb.redirectErrorStream(true);

                appendLog("Running transfer learning:");
                appendLog(String.join(" ", pb.command()));

                appendLog("Requested training device: " + selectedDevice);

                Process process = pb.start();

                try (BufferedReader reader = new BufferedReader(
                        new InputStreamReader(process.getInputStream()))) {

                    String line;

                    while ((line = reader.readLine()) != null) {
                        logPython(line);
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

                    appendLog("Retraining completed successfully.");
                    appendLog("Generated model: " + outputModel.getAbsolutePath());

                    if (logFile != null) {
                        appendLog("Session log saved to: " + logFile.getAbsolutePath());
                    }

                    IJ.showMessage(
                            "Retraining completed",
                            "Model saved to:\n" + outputModel.getAbsolutePath()
                    );
                });

            } catch (Exception e) {
                logError("Retraining failed: " + e.getMessage());
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

                appendLog("Training sample from corrected annotation: " + imageFile.getName());
                count++;
                continue;
            }

            if (detectionFile.exists()) {
                ImagePlus imp = IJ.openImage(imageFile.getAbsolutePath());

                if (imp == null) {
                    logWarning("Skipping image because ImageJ could not open it: " + imageFile.getAbsolutePath());
                    continue;
                }

                try {
                    List<NeuronDetection> automaticDetections = readDetectionsFromCsv(detectionFile);

                    if (automaticDetections.isEmpty()) {
                        logWarning("Skipping image without detections: " + imageFile.getName());
                        continue;
                    }

                    createTrainingImageForYolo(imageFile, yoloImage);

                    writeYoloLabelFile(
                            yoloLabel,
                            imp.getWidth(),
                            imp.getHeight(),
                            automaticDetections
                    );

                    appendLog("Training sample from automatic detection: " + imageFile.getName());
                    count++;

                } finally {
                    imp.close();
                }
            }
        }

        writeDataYaml(yoloDatasetDir);

        appendLog("Prepared training set with " + count + " images.");
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
                    logWarning("Skipping malformed CSV line: " + line);
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

            appendLog("Created normalized training PNG: " + targetImageFile.getAbsolutePath());

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
                    logWarning("Could not delete directory: " + file.getAbsolutePath());
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
            logError("Could not change active model: " + e.getMessage());
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
                logWarning("Could not read global active model. Falling back to base model.");
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
            logWarning("Could not read folder active model.");
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
        if (modelLabel == null) {
            return;
        }

        File model = getActiveModelFile();

        if (model != null && model.exists()) {
            modelLabel.setText(model.getName());
            modelLabel.setCaretPosition(0);
            modelLabel.setToolTipText(model.getAbsolutePath());
        } else {
            modelLabel.setText("No model selected");
            modelLabel.setToolTipText(null);
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

    private void initializeLogFile() {
        closeLogFile();

        if (workDir == null) {
            return;
        }

        try {
            File logsDir = new File(workDir, "logs");

            if (!logsDir.exists() && !logsDir.mkdirs()) {
                logWarning("Could not create logs directory: " + logsDir.getAbsolutePath());
                return;
            }

            String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());

            logFile = new File(
                    logsDir,
                    "transfer_learning_session_" + timestamp + ".log"
            );

            logWriter = new PrintWriter(new FileWriter(logFile, true), true);

            appendLogSeparator();
            appendLog("Log file created: " + logFile.getAbsolutePath());
            appendLog("Session id: " + sessionId);
            appendLog("Selected folder: " + selectedFolder.getAbsolutePath());
            appendLog("Work directory: " + workDir.getAbsolutePath());
            appendLogSeparator();

        } catch (Exception e) {
            logError("Could not initialize log file: " + e.getMessage());
            IJ.handleException(e);
        }
    }

    private void closeLogFile() {
        if (logWriter != null) {
            logWriter.flush();
            logWriter.close();
            logWriter = null;
        }
    }

    private String getCurrentTimestampForLog() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    }

    private void appendLogSeparator() {
        appendLog("SYSTEM", "------------------------------------------------------------");
    }

    private void updateStatus(String message) {
        int count = detections.size();

        String fullMessage;

        if (count > 0) {
            fullMessage = "Neurons: " + count + " | " + message;
        } else {
            fullMessage = message;
        }

        if (statusLabel != null) {
            statusLabel.setText(fullMessage);
        }

        appendLog("STATUS", message);
    }

    private synchronized void appendLog(String message) {
        appendLog("INFO", message);
    }

    private synchronized void appendLog(String level, String message) {
        String timestampedMessage =
                "[" + getCurrentTimestampForLog() + "] " +
                        "[" + level + "] " +
                        message;

        if (logTextArea != null) {
            SwingUtilities.invokeLater(() -> {
                logTextArea.append(timestampedMessage + "\n");
                logTextArea.setCaretPosition(logTextArea.getDocument().getLength());
            });
        }

        if (logWriter != null) {
            logWriter.println(timestampedMessage);
            logWriter.flush();
        }
    }

    private void logInfo(String message) {
        appendLog("INFO", message);
    }

    private void logWarning(String message) {
        appendLog("WARNING", message);
    }

    private void logError(String message) {
        appendLog("ERROR", message);
    }

    private void logPython(String message) {
        appendLog("PYTHON", message);
    }

    private void updateButtonState() {
        boolean hasFolder = workDir != null;
        boolean hasImage = sourceImage != null;
        boolean hasDetections = !detections.isEmpty();
        boolean busy = detectionRunning || retrainingRunning;

        selectFolderButton.setEnabled(!busy);
        detectImagesButton.setEnabled(hasFolder && !busy);
        correctButton.setEnabled(hasImage && hasDetections && !correctionMode && !busy);

        deleteSelectedButton.setEnabled(correctionMode && !selectedDetectionIndices.isEmpty() && !busy);
        addNeuronButton.setEnabled(correctionMode && !busy);
        undoButton.setEnabled(correctionMode && undoState != null && !busy);

        saveToTrainingSetButton.setEnabled(correctionMode && hasImage && hasDetections && !busy);

        retrainButton.setEnabled(hasFolder && !busy);
        changeModelButton.setEnabled(!busy);

        if (useHardwareAccelerationCheckBox != null) {
            useHardwareAccelerationCheckBox.setEnabled(!busy);
        }
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

        selectedDetectionIndices.clear();
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
                logWarning("Could not use original mask polygon for " + detection.name + ". Falling back to ellipse.");
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
                    logWarning("Skipping invalid polygon for detection: " + detection.name);
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
                    logWarning("Skipping invalid polygon for detection: " + detection.name);
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

    private List<NeuronDetection> copyDetections(List<NeuronDetection> source) {
        List<NeuronDetection> copy = new ArrayList<>();

        for (NeuronDetection detection : source) {
            copy.add(new NeuronDetection(
                    detection.name,
                    detection.cx,
                    detection.cy,
                    detection.width,
                    detection.height,
                    detection.maskPolygon,
                    detection.edited
            ));
        }

        return copy;
    }

    private void saveUndoState() {
        undoState = copyDetections(detections);
        updateButtonState();
    }

    private void undoLastChange() {
        if (undoState == null) {
            updateStatus("No changes to undo.");
            return;
        }

        detections.clear();
        detections.addAll(copyDetections(undoState));

        undoState = null;
        selectedDetectionIndices.clear();
        addNeuronMode = false;
        addNeuronButton.setText("Add");

        refreshAfterEditing();

        updateStatus("Last change undone.");
        appendLog("Undo applied to current image.");
    }

    private void deleteSelectedDetection() {
        if (selectedDetectionIndices.isEmpty()) {
            IJ.showMessage(
                    "Delete neuron",
                    "Please select one or more neurons first."
            );
            return;
        }

        saveUndoState();

        selectedDetectionIndices.sort((a, b) -> Integer.compare(b, a));

        int deletedCount = 0;

        for (int index : selectedDetectionIndices) {
            if (index >= 0 && index < detections.size()) {
                detections.remove(index);
                deletedCount++;
            }
        }

        selectedDetectionIndices.clear();
        refreshAfterEditing();

        updateStatus("Deleted " + deletedCount + " selected neuron(s).");
    }

    private void refreshAfterEditing() {
        roiListModel.clear();

        for (NeuronDetection detection : detections) {
            roiListModel.addElement(detection.name);
        }

        selectedDetectionIndices.clear();
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

        saveUndoState();

        detections.add(new NeuronDetection(
                "MANUAL_" + manualIndex,
                cx,
                cy,
                width,
                height,
                null,
                true
        ));

        refreshAfterEditing();
        setSingleDetectionSelection(detections.size() - 1);
        updateStatus("Manual neuron added.");
    }

    private void deleteIfExists(File file) {
        if (file.exists() && !file.delete()) {
            logWarning("Previous file could not be deleted: " + file.getAbsolutePath());
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
                        logWarning("Could not delete temp file: " + file.getAbsolutePath());
                    }
                }
            }
        }

        if (!directory.delete()) {
            logWarning("Could not delete temp directory: " + directory.getAbsolutePath());
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
            String statusPrefix;

            switch (status) {
                case "annotated":
                    statusPrefix = "[annotated] ";
                    break;
                case "detected":
                    statusPrefix = "[detected] ";
                    break;
                default:
                    statusPrefix = "[pending] ";
                    break;
            }

            return statusPrefix + file.getName();
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

        private boolean strokeSelecting = false;
        private final List<Point> selectionStrokePoints = new ArrayList<>();

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
                            setSingleDetectionSelection(clickedIndex);
                            repaint();
                            return;
                        }

                        selectedDetectionIndices.clear();
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
                        boolean multiSelect =
                                e.isControlDown() ||
                                        e.isMetaDown() ||
                                        e.isShiftDown();

                        if (multiSelect) {
                            toggleDetectionSelection(clickedIndex);
                            movingSelected = false;
                        } else {
                            setSingleDetectionSelection(clickedIndex);

                            saveUndoState();

                            movingSelected = true;
                            lastMoveImageX = imageX;
                            lastMoveImageY = imageY;
                        }

                        repaint();
                    } else {
                        boolean strokeSelect =
                                e.isShiftDown() ||
                                        e.isControlDown() ||
                                        e.isMetaDown();

                        if (strokeSelect) {
                            strokeSelecting = true;
                            selectionStrokePoints.clear();
                            selectionStrokePoints.add(new Point(e.getX(), e.getY()));

                            updateStatus("Lasso selection enabled. Draw around neurons to select them.");
                            repaint();
                            return;
                        }

                        clearDetectionSelection();
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

                    if (strokeSelecting) {
                        selectionStrokePoints.add(new Point(e.getX(), e.getY()));
                        repaint();
                        return;
                    }

                    if (movingSelected) {
                        int selectedIndex = roiList.getSelectedIndex();

                        if (selectedDetectionIndices.size() == 1) {
                            selectedIndex = selectedDetectionIndices.get(0);
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

                    if (strokeSelecting) {
                        strokeSelecting = false;

                        selectDetectionsInsideLasso();

                        selectionStrokePoints.clear();

                        updateButtonState();
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

            if (strokeSelecting) {
                drawSelectionStroke((Graphics2D) g);
            }
        }

        private void drawSelectionStroke(Graphics2D g2) {
            if (selectionStrokePoints.size() < 2) {
                return;
            }

            g2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            g2.setColor(COLOR_LASSO);
            g2.setStroke(new BasicStroke(
                    2.5f,
                    BasicStroke.CAP_ROUND,
                    BasicStroke.JOIN_ROUND
            ));

            for (int i = 1; i < selectionStrokePoints.size(); i++) {
                Point previous = selectionStrokePoints.get(i - 1);
                Point current = selectionStrokePoints.get(i);

                g2.drawLine(previous.x, previous.y, current.x, current.y);
            }
        }

        private void drawDetections(Graphics2D g2) {
            g2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            for (int i = 0; i < detections.size(); i++) {
                NeuronDetection detection = detections.get(i);

                Color roiColor;
                Stroke colorStroke;
                Stroke shadowStroke;

                if (isDetectionSelected(i)) {
                    roiColor = COLOR_SELECTED;
                    colorStroke = new BasicStroke(3.2f);
                    shadowStroke = new BasicStroke(4.6f);
                } else if (detection.name.startsWith("MANUAL_")) {
                    roiColor = COLOR_MANUAL;
                    colorStroke = new BasicStroke(
                            2.4f,
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                            10.0f,
                            new float[]{8.0f, 5.0f},
                            0.0f
                    );
                    shadowStroke = new BasicStroke(
                            3.8f,
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                            10.0f,
                            new float[]{8.0f, 5.0f},
                            0.0f
                    );
                } else if (detection.name.startsWith("ANNOTATION_")) {
                    roiColor = COLOR_ANNOTATION;
                    colorStroke = new BasicStroke(
                            2.1f,
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                            10.0f,
                            new float[]{2.5f, 4.0f},
                            0.0f
                    );
                    shadowStroke = new BasicStroke(
                            3.6f,
                            BasicStroke.CAP_ROUND,
                            BasicStroke.JOIN_ROUND,
                            10.0f,
                            new float[]{2.5f, 4.0f},
                            0.0f
                    );
                } else {
                    roiColor = COLOR_AUTOMATIC;
                    colorStroke = new BasicStroke(2.0f);
                    shadowStroke = new BasicStroke(3.2f);
                }

                int x = (int) Math.round(offsetX + (detection.cx - detection.width / 2.0) * zoomFactor);
                int y = (int) Math.round(offsetY + (detection.cy - detection.height / 2.0) * zoomFactor);
                int w = (int) Math.round(detection.width * zoomFactor);
                int h = (int) Math.round(detection.height * zoomFactor);

                g2.setColor(COLOR_OUTLINE_SHADOW);
                g2.setStroke(shadowStroke);
                g2.drawOval(x, y, w, h);

                g2.setColor(roiColor);
                g2.setStroke(colorStroke);
                g2.drawOval(x, y, w, h);
            }
        }

        private void drawTemporaryRoi(Graphics2D g2) {
            g2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            g2.setColor(COLOR_TEMPORARY);
            g2.setStroke(new BasicStroke(2.5f));

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

        private void selectDetectionsInsideLasso() {
            if (selectionStrokePoints.size() < 3) {
                return;
            }

            Polygon lasso = new Polygon();

            for (Point point : selectionStrokePoints) {
                lasso.addPoint(point.x, point.y);
            }

            int selectedCountBefore = selectedDetectionIndices.size();

            for (int i = 0; i < detections.size(); i++) {
                NeuronDetection detection = detections.get(i);

                int screenX = (int) Math.round(offsetX + detection.cx * zoomFactor);
                int screenY = (int) Math.round(offsetY + detection.cy * zoomFactor);

                if (lasso.contains(screenX, screenY)) {
                    if (!selectedDetectionIndices.contains(i)) {
                        selectedDetectionIndices.add(i);
                        roiList.addSelectionInterval(i, i);
                    }
                }
            }

            int selectedNow = selectedDetectionIndices.size() - selectedCountBefore;
            updateStatus("Lasso selected " + selectedNow + " neuron(s). Total selected: " + selectedDetectionIndices.size());
            updateButtonState();
        }
    }
}