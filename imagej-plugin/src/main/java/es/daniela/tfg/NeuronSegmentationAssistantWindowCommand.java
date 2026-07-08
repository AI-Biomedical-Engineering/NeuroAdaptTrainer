package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.ImageListener;
import ij.WindowManager;
import ij.io.FileSaver;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;

@Plugin(type = Command.class, menuPath = "Plugins>Neuron Segmentation>Single Image Segmentation")
public class NeuronSegmentationAssistantWindowCommand implements Command, ImageListener {

    private String pythonExe;
    private String scriptPath;

    private JFrame frame;

    private DefaultListModel<String> roiListModel;
    private JList<String> roiList;

    private JLabel statusLabel;
    private JLabel zoomLabel;
    private JTextField modelLabel;

    private JScrollPane imageScrollPane;
    private ImagePanel imagePanel;

    private JButton importImageButton;
    private JButton changeModelButton;
    private JButton detectButton;
    private JButton openTransferLearningButton;

    private JButton zoomInButton;
    private JButton zoomOutButton;
    private JButton fitButton;
    private JButton actualSizeButton;

    private JCheckBox useHardwareAccelerationCheckBox;
    private boolean useHardwareAccelerationForInference = false;
    private static final String CONFIG_USE_HARDWARE_ACCELERATION = "use_hardware_acceleration";

    private ImagePlus sourceImage;
    private ImagePlus imageToDisplay;

    private File activeModelFile;

    private final List<NeuronDetection> detections = new ArrayList<>();

    private boolean detectionRunning = false;

    private double zoomFactor = 1.0;
    private boolean fitToPanel = true;

    private final String sessionId = UUID.randomUUID().toString();
    private File sessionDir;

    private static final Set<Integer> ASSISTANT_IMPORTED_IMAGE_IDS =
            Collections.synchronizedSet(new HashSet<>());

    private static final Color COLOR_AUTOMATIC = new Color(0, 170, 255);
    private static final Color COLOR_SELECTED = new Color(255, 80, 80);
    private static final Color COLOR_OUTLINE_SHADOW = new Color(0, 0, 0, 100);

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
            logToConsole("Warning: could not create session directory: " + sessionDir.getAbsolutePath());
        } else {
            logToConsole("Temporary session directory: " + sessionDir.getAbsolutePath());
        }

        sourceImage = getCurrentOrFirstImage();

        frame = new JFrame("Single Image Neuron Segmentation");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(1150, 760);
        frame.setLayout(new BorderLayout(8, 8));

        JSplitPane leftCenterSplit = new JSplitPane(
                JSplitPane.HORIZONTAL_SPLIT,
                createLeftPanel(),
                createCenterPanel()
        );

        leftCenterSplit.setResizeWeight(0.20);
        leftCenterSplit.setDividerLocation(220);
        leftCenterSplit.setOneTouchExpandable(true);

        JSplitPane mainSplit = new JSplitPane(
                JSplitPane.HORIZONTAL_SPLIT,
                leftCenterSplit,
                createRightPanel()
        );

        mainSplit.setResizeWeight(0.78);
        mainSplit.setDividerLocation(850);
        mainSplit.setOneTouchExpandable(true);

        frame.add(mainSplit, BorderLayout.CENTER);
        frame.add(createBottomPanel(), BorderLayout.SOUTH);

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
            updateDisplayedImage();
        } else {
            updateStatus("No Fiji image selected.");
        }

        updateModelLabel();
        updateButtonState();

        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
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

            useHardwareAccelerationForInference = Boolean.parseBoolean(
                    properties.getProperty(CONFIG_USE_HARDWARE_ACCELERATION, "false")
            );

            if (pythonExe == null || pythonExe.trim().isEmpty()) {
                IJ.error("Configuration error", "Missing property: python");
                return;
            }

            if (scriptPath == null || scriptPath.trim().isEmpty()) {
                IJ.error("Configuration error", "Missing property: script");
                return;
            }

            logToConsole("Loaded Python executable: " + pythonExe);
            logToConsole("Loaded inference script: " + scriptPath);
            logToConsole("Use hardware acceleration: " + useHardwareAccelerationForInference);

            activeModelFile = getGlobalActiveModelOrBaseModel();

        } catch (Exception e) {
            IJ.handleException(e);
        }
    }

    private void logToConsole(String message) {
        System.out.println("[Single Image Assistant] " + message);
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

        detections.clear();

        if (roiListModel != null) {
            roiListModel.clear();
        }

        fitToPanel = true;

        updateStatus(message + ": " + sourceImage.getTitle());
        updateButtonState();
        updateDisplayedImage();
    }

    private JPanel createLeftPanel() {
        JPanel panel = new JPanel(new BorderLayout(4, 4));
        panel.setPreferredSize(new Dimension(190, 0));
        panel.setBorder(BorderFactory.createTitledBorder("Detections"));

        roiListModel = new DefaultListModel<>();
        roiList = new JList<>(roiListModel);
        roiList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        roiList.setVisibleRowCount(20);

        roiList.addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                int index = roiList.getSelectedIndex();

                if (index >= 0 && index < detections.size()) {
                    updateStatus("Selected detection: " + detections.get(index).name);
                }

                if (imagePanel != null) {
                    imagePanel.repaint();
                }
            }
        });

        JScrollPane scrollPane = new JScrollPane(roiList);
        panel.add(scrollPane, BorderLayout.CENTER);

        JLabel hintLabel = new JLabel("Click a detection or a neuron");
        hintLabel.setFont(hintLabel.getFont().deriveFont(Font.PLAIN, 11f));
        hintLabel.setBorder(BorderFactory.createEmptyBorder(2, 4, 4, 4));

        panel.add(hintLabel, BorderLayout.SOUTH);

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

        JPanel contentPanel = new JPanel();
        contentPanel.setLayout(new BoxLayout(contentPanel, BoxLayout.Y_AXIS));

        JPanel infoPanel = new JPanel(new BorderLayout(4, 4));
        infoPanel.setBorder(BorderFactory.createTitledBorder("Mode"));

        JLabel infoLabel = new JLabel(
                "<html>" +
                        "Detect neurons in the current image.<br><br>" +
                        "Use <b>Transfer Learning</b> to process a full folder, " +
                        "review corrections, save annotations and retrain the model." +
                        "</html>"
        );

        infoLabel.setFont(infoLabel.getFont().deriveFont(Font.PLAIN, 11f));
        infoLabel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));

        openTransferLearningButton = new JButton("Open transfer learning");
        styleActionButton(openTransferLearningButton, false);
        openTransferLearningButton.addActionListener(e -> new NeuronTransferLearningWindowCommand().run());

        infoPanel.add(infoLabel, BorderLayout.CENTER);
        infoPanel.add(openTransferLearningButton, BorderLayout.SOUTH);

        JPanel modelPanel = new JPanel(new BorderLayout(4, 4));
        modelPanel.setBorder(BorderFactory.createTitledBorder("Active model"));

        modelLabel = new JTextField("No model selected");
        modelLabel.setEditable(false);
        modelLabel.setBorder(BorderFactory.createEmptyBorder(4, 4, 4, 4));
        modelLabel.setToolTipText("Current model used for single image detection");
        modelLabel.setHorizontalAlignment(JTextField.LEFT);

        changeModelButton = new JButton("Change model");
        styleActionButton(changeModelButton, false);
        changeModelButton.addActionListener(e -> changeActiveModel());

        useHardwareAccelerationCheckBox = new JCheckBox("Use hardware acceleration");
        useHardwareAccelerationCheckBox.setSelected(useHardwareAccelerationForInference);
        useHardwareAccelerationCheckBox.setFont(
                useHardwareAccelerationCheckBox.getFont().deriveFont(Font.PLAIN, 11f)
        );
        useHardwareAccelerationCheckBox.setToolTipText(
                "If enabled, inference will request CUDA or Apple MPS when available."
        );

        useHardwareAccelerationCheckBox.addActionListener(e -> {
            saveHardwareAccelerationPreference(useHardwareAccelerationCheckBox.isSelected());

            if (useHardwareAccelerationCheckBox.isSelected()) {
                updateStatus("Hardware acceleration enabled for inference.");
            } else {
                updateStatus("CPU inference selected.");
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
        actionsPanel.setLayout(new BoxLayout(actionsPanel, BoxLayout.Y_AXIS));
        actionsPanel.setBorder(BorderFactory.createTitledBorder("Actions"));

        importImageButton = new JButton("Import image");
        detectButton = new JButton("Detect neurons");

        styleActionButton(importImageButton, false);
        styleActionButton(detectButton, true);

        importImageButton.addActionListener(e -> importImageFromFile());
        detectButton.addActionListener(e -> detectNeurons());

        actionsPanel.add(importImageButton);
        actionsPanel.add(Box.createVerticalStrut(8));
        actionsPanel.add(detectButton);

        contentPanel.add(infoPanel);
        contentPanel.add(Box.createVerticalStrut(8));
        contentPanel.add(modelPanel);
        contentPanel.add(Box.createVerticalStrut(8));
        contentPanel.add(actionsPanel);

        wrapper.add(contentPanel, BorderLayout.NORTH);

        return wrapper;
    }

    private void saveHardwareAccelerationPreference(boolean useHardwareAcceleration) {
        useHardwareAccelerationForInference = useHardwareAcceleration;

        File configFile = new File(
                System.getProperty("user.home"),
                ".neuron-segmentation-assistant/config.properties"
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

            logToConsole("Hardware acceleration preference saved: " + useHardwareAcceleration);

        } catch (Exception e) {
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
        JPanel panel = new JPanel(new BorderLayout());

        statusLabel = new JLabel("Ready.");
        statusLabel.setBorder(BorderFactory.createEmptyBorder(6, 12, 6, 12));

        panel.add(statusLabel, BorderLayout.WEST);

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

        setSourceImage(importedImage, "Imported image");
    }

    private void updateStatus(String message) {
        int count = detections.size();

        if (statusLabel == null) {
            return;
        }

        if (count > 0) {
            statusLabel.setText("Neurons: " + count + " | " + message);
        } else {
            statusLabel.setText(message);
        }
    }

    private void updateButtonState() {
        boolean hasImage = sourceImage != null;
        boolean busy = detectionRunning;

        if (importImageButton != null) {
            importImageButton.setEnabled(!busy);
        }

        if (changeModelButton != null) {
            changeModelButton.setEnabled(!busy);
        }

        if (detectButton != null) {
            detectButton.setEnabled(hasImage && !busy);
        }

        if (openTransferLearningButton != null) {
            openTransferLearningButton.setEnabled(!busy);
        }

        if (useHardwareAccelerationCheckBox != null) {
            useHardwareAccelerationCheckBox.setEnabled(!busy);
        }
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

        File activeModel = getActiveModelFile();

        if (activeModel == null || !activeModel.exists()) {
            IJ.error("No active model found. Please select a valid model.");
            return;
        }

        sourceImage = currentImage;
        imageToDisplay = sourceImage;

        detectionRunning = true;

        detections.clear();
        roiListModel.clear();

        updateStatus("Detecting neurons... Please wait.");
        updateButtonState();

        if (imagePanel != null) {
            imagePanel.repaint();
        }

        final ImagePlus imageForDetection = currentImage;

        new Thread(() -> {
            try {
                int detectedCount = runPythonDetection(imageForDetection);

                loadDetectionsFromCsv();

                imageToDisplay = sourceImage;
                fitToPanel = true;
                updateDisplayedImage();

                SwingUtilities.invokeLater(() -> {
                    detectionRunning = false;

                    updateStatus("Detection completed. Select a detection from the list or click a neuron.");
                    updateButtonState();

                    IJ.showStatus("Detected neurons: " + detections.size());
                    logToConsole("Assistant detection completed. Count: " + detections.size());

                    if (detectedCount >= 0) {
                        logToConsole("Python reported neuron count: " + detectedCount);
                    }
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

    private int runPythonDetection(ImagePlus imp) throws Exception {
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

        deleteIfExists(inputFile);
        deleteIfExists(outputFile);
        deleteIfExists(csvFile);

        boolean saved = new FileSaver(imp).saveAsPng(inputFile.getAbsolutePath());

        if (!saved || !inputFile.exists()) {
            throw new RuntimeException("Could not save the input image.");
        }

        File pythonFile = new File(pythonExe);
        File scriptFile = new File(scriptPath);
        File modelFile = getActiveModelFile();

        if (!pythonFile.exists()) {
            throw new RuntimeException("Python executable not found: " + pythonExe);
        }

        if (!scriptFile.exists()) {
            throw new RuntimeException("Inference script not found: " + scriptPath);
        }

        if (modelFile == null || !modelFile.exists()) {
            throw new RuntimeException("Active model not found.");
        }

        String selectedDevice;

        if (useHardwareAccelerationCheckBox != null && useHardwareAccelerationCheckBox.isSelected()) {
            selectedDevice = "auto_acceleration";
        } else {
            selectedDevice = "cpu";
        }

        ProcessBuilder pb = new ProcessBuilder(
                pythonExe,
                "-u",
                scriptPath,
                inputFile.getAbsolutePath(),
                outputFile.getAbsolutePath(),
                modelFile.getAbsolutePath(),
                selectedDevice
        );

        logToConsole("Requested inference device: " + selectedDevice);

        pb.redirectErrorStream(true);

        logToConsole("Running Python inference from single image assistant...");
        logToConsole("Active model: " + modelFile.getAbsolutePath());
        logToConsole(String.join(" ", pb.command()));

        Process process = pb.start();

        int neuronCount = -1;

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {

            String line;

            while ((line = reader.readLine()) != null) {
                logToConsole(line);

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

        if (!csvFile.exists()) {
            throw new RuntimeException("The detection CSV was not created.");
        }

        return neuronCount;
    }

    private void loadDetectionsFromCsv() throws Exception {
        File csvFile = new File(sessionDir, "imagej_output.csv");

        if (!csvFile.exists()) {
            throw new RuntimeException("CSV not found: " + csvFile.getAbsolutePath());
        }

        detections.clear();

        try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
            String line = br.readLine();
            int index = 1;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");

                if (parts.length < 3) {
                    logToConsole("Skipping malformed CSV line: " + line);
                    continue;
                }

                double cx = Double.parseDouble(parts[0]);
                double cy = Double.parseDouble(parts[1]);
                double radius = Double.parseDouble(parts[2]);
                double diameter = radius * 2.0;

                detections.add(new NeuronDetection(
                        "AUTO_" + index,
                        cx,
                        cy,
                        diameter,
                        diameter
                ));

                index++;
            }
        }

        SwingUtilities.invokeLater(() -> {
            roiListModel.clear();

            for (NeuronDetection detection : detections) {
                roiListModel.addElement(detection.name);
            }

            if (!detections.isEmpty()) {
                roiList.setSelectedIndex(0);
                roiList.ensureIndexIsVisible(0);
            }

            if (imagePanel != null) {
                imagePanel.repaint();
            }
        });
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
                logToConsole("Could not read global active model. Falling back to base model.");
            }
        }

        if (scriptPath == null || scriptPath.trim().isEmpty()) {
            return null;
        }

        File scriptFile = new File(scriptPath);

        if (!scriptFile.exists()) {
            return null;
        }

        File pythonRoot = scriptFile.getParentFile();

        if (pythonRoot == null) {
            return null;
        }

        return new File(pythonRoot, "models/best.pt");
    }

    private void changeActiveModel() {
        JFileChooser chooser = new JFileChooser();

        File currentModel = getActiveModelFile();

        if (currentModel != null && currentModel.getParentFile() != null) {
            chooser.setCurrentDirectory(currentModel.getParentFile());
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
            writeGlobalActiveModel(selectedModel);
            updateModelLabel();
            updateStatus("Active model changed: " + selectedModel.getName());

        } catch (Exception e) {
            IJ.handleException(e);
            updateStatus("Could not change active model.");
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

    private String shortenTextMiddle(String text, int maxLength) {
        if (text == null || text.length() <= maxLength) {
            return text;
        }

        int keepStart = Math.max(4, maxLength / 2 - 2);
        int keepEnd = Math.max(4, maxLength - keepStart - 3);

        return text.substring(0, keepStart) + "..." + text.substring(text.length() - keepEnd);
    }

    private void updateModelLabel() {
        if (modelLabel == null) {
            return;
        }

        File model = getActiveModelFile();

        if (model != null && model.exists()) {
            String name = model.getName();

            modelLabel.setText(name);
            modelLabel.setCaretPosition(0);
            modelLabel.setToolTipText(model.getAbsolutePath());

        } else {
            modelLabel.setText("No model selected");
            modelLabel.setToolTipText(null);
        }
    }

    private int parseNeuronCount(String line) {
        try {
            String[] parts = line.split(":");

            if (parts.length > 1) {
                return Integer.parseInt(parts[1].trim());
            }

        } catch (Exception e) {
            logToConsole("Could not parse neuron count from line: " + line);
        }

        return -1;
    }

    private void deleteIfExists(File file) {
        if (file.exists() && !file.delete()) {
            logToConsole("Warning: previous file could not be deleted: " + file.getAbsolutePath());
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
                } else if (!file.delete()) {
                    logToConsole("Warning: could not delete temp file: " + file.getAbsolutePath());
                }
            }
        }

        if (!directory.delete()) {
            logToConsole("Warning: could not delete temp directory: " + directory.getAbsolutePath());
        } else {
            logToConsole("Temporary session directory deleted: " + directory.getAbsolutePath());
        }
    }

    @Override
    public void imageOpened(ImagePlus imp) {
        SwingUtilities.invokeLater(() -> {
            if (imp == null) {
                return;
            }

            if (ASSISTANT_IMPORTED_IMAGE_IDS.contains(imp.getID())) {
                return;
            }

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

                detections.clear();

                if (roiListModel != null) {
                    roiListModel.clear();
                }

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

    private static class NeuronDetection {
        private final String name;
        private final double cx;
        private final double cy;
        private final double width;
        private final double height;

        private NeuronDetection(
                String name,
                double cx,
                double cy,
                double width,
                double height
        ) {
            this.name = name;
            this.cx = cx;
            this.cy = cy;
            this.width = width;
            this.height = height;
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
            setBackground(Color.DARK_GRAY);

            java.awt.event.MouseAdapter mouseAdapter = new java.awt.event.MouseAdapter() {
                @Override
                public void mousePressed(java.awt.event.MouseEvent e) {
                    if (image == null || detections.isEmpty()) {
                        return;
                    }

                    if (!isInsideDisplayedImage(e.getX(), e.getY())) {
                        return;
                    }

                    double imageX = screenToImageX(e.getX());
                    double imageY = screenToImageY(e.getY());

                    int clickedIndex = findDetectionAt(imageX, imageY);

                    if (clickedIndex >= 0) {
                        roiList.setSelectedIndex(clickedIndex);
                        roiList.ensureIndexIsVisible(clickedIndex);
                        updateStatus("Selected detection: " + detections.get(clickedIndex).name);
                    } else {
                        roiList.clearSelection();
                        updateStatus("No detection selected.");
                    }

                    repaint();
                }
            };

            addMouseListener(mouseAdapter);
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

        private int findDetectionAt(double imageX, double imageY) {
            for (int i = detections.size() - 1; i >= 0; i--) {
                NeuronDetection detection = detections.get(i);

                if (isPointInsideDetection(imageX, imageY, detection)) {
                    return i;
                }
            }

            return -1;
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
        }

        private void drawDetections(Graphics2D g2) {
            g2.setRenderingHint(
                    RenderingHints.KEY_ANTIALIASING,
                    RenderingHints.VALUE_ANTIALIAS_ON
            );

            int selectedIndex = roiList != null ? roiList.getSelectedIndex() : -1;

            for (int i = 0; i < detections.size(); i++) {
                NeuronDetection detection = detections.get(i);

                int x = (int) Math.round(offsetX + (detection.cx - detection.width / 2.0) * zoomFactor);
                int y = (int) Math.round(offsetY + (detection.cy - detection.height / 2.0) * zoomFactor);
                int w = (int) Math.round(detection.width * zoomFactor);
                int h = (int) Math.round(detection.height * zoomFactor);

                boolean selected = i == selectedIndex;

                Stroke colorStroke;
                Stroke shadowStroke;
                Color roiColor;

                if (selected) {
                    roiColor = COLOR_SELECTED;
                    colorStroke = new BasicStroke(3.2f);
                    shadowStroke = new BasicStroke(4.6f);
                } else {
                    roiColor = COLOR_AUTOMATIC;
                    colorStroke = new BasicStroke(2.0f);
                    shadowStroke = new BasicStroke(3.2f);
                }

                g2.setColor(COLOR_OUTLINE_SHADOW);
                g2.setStroke(shadowStroke);
                g2.drawOval(x, y, w, h);

                g2.setColor(roiColor);
                g2.setStroke(colorStroke);
                g2.drawOval(x, y, w, h);

                if (selected) {
                    int centerX = (int) Math.round(offsetX + detection.cx * zoomFactor);
                    int centerY = (int) Math.round(offsetY + detection.cy * zoomFactor);

                    g2.setColor(COLOR_SELECTED);
                }
            }
        }
    }
}