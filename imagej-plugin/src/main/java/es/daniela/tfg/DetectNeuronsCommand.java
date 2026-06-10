package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.io.FileSaver;
import ij.gui.OvalRoi;
import ij.gui.Overlay;

import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.io.FileReader;
import java.awt.Color;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Detect Neurons (YOLO)")
public class DetectNeuronsCommand implements Command {

    private static final String PYTHON_EXE = "/Users/danielaerasocasas/tfg/venv/bin/python3";
    private static final String SCRIPT_PATH = "/Users/danielaerasocasas/Documents/gitHub/fiji-yolo-neuron-segmentation/yolo-inference/infer_one.py";

    @Override
    public void run() {
        try {
            ImagePlus imp = WindowManager.getCurrentImage();
            if (imp == null) {
                IJ.error("No image is open.");
                return;
            }

            String tmpDir = System.getProperty("java.io.tmpdir");
            File inputFile = new File(tmpDir, "imagej_input.png");
            File outputFile = new File(tmpDir, "imagej_output.png");

            if (outputFile.exists() && !outputFile.delete()) {
                IJ.log("Warning: previous output file could not be deleted.");
            }

            FileSaver saver = new FileSaver(imp);
            boolean saved = saver.saveAsPng(inputFile.getAbsolutePath());
            if (!saved) {
                IJ.error("Could not save the input image.");
                return;
            }

            ProcessBuilder pb = new ProcessBuilder(
                    PYTHON_EXE,
                    "-u",
                    SCRIPT_PATH,
                    inputFile.getAbsolutePath(),
                    outputFile.getAbsolutePath()
            );

            pb.redirectErrorStream(true);

            IJ.log("Running Python inference...");
            IJ.log(String.join(" ", pb.command()));

            Process process = pb.start();

            int neuronCount = -1;

            try (BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    IJ.log(line);

                    if (line.startsWith("Detected neurons")) {
                        String[] parts = line.split(":");
                        if (parts.length > 1) {
                            neuronCount = Integer.parseInt(parts[1].trim());
                        }
                    }
                }
            }

            int exitCode = process.waitFor();

            if (exitCode != 0) {
                IJ.error("Python process failed with exit code " + exitCode);
                return;
            }

            if (!outputFile.exists()) {
                IJ.error("The output image was not created.");
                return;
            }

            ImagePlus result = IJ.openImage(outputFile.getAbsolutePath());
            if (result == null) {
                IJ.error("The output image could not be opened.");
                return;
            }

            int detectedFromCsv = loadDetectionsAsOverlay(result);

            int finalCount = neuronCount >= 0 ? neuronCount : detectedFromCsv;

            if (finalCount >= 0) {
                result.setTitle("Detected Neurons - " + finalCount + " neurons");
            } else {
                result.setTitle("Detected Neurons");
            }

            result.show();

            if (finalCount >= 0) {
                IJ.showStatus("Detected neurons: " + finalCount);
                IJ.showMessage("Neuron Detection", "Detected neurons: " + finalCount);
            }
        } catch (Exception e) {
            IJ.handleException(e);
        }
    }

    private int loadDetectionsAsOverlay(ImagePlus imp) {
        int count = 0;

        try {
            String tmpDir = System.getProperty("java.io.tmpdir");
            File csvFile = new File(tmpDir, "imagej_output.csv");

            if (!csvFile.exists()) {
                IJ.log("CSV not found: " + csvFile.getAbsolutePath());
                return -1;
            }

            Overlay overlay = new Overlay();

            try (BufferedReader br = new BufferedReader(new FileReader(csvFile))) {
                String line = br.readLine(); // skip header

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

                    // Circle around the detected neuron
                    OvalRoi circle = new OvalRoi(x, y, d, d);
                    circle.setStrokeColor(Color.BLUE);
                    circle.setStrokeWidth(1.5);
                    circle.setName("AUTO_" + (count + 1));
                    overlay.add(circle);

                    // Small center dot
                    double dotRadius = 1.8;
                    OvalRoi centerDot = new OvalRoi(
                            cx - dotRadius,
                            cy - dotRadius,
                            dotRadius * 2,
                            dotRadius * 2
                    );
                    centerDot.setStrokeColor(Color.BLUE);
                    centerDot.setFillColor(Color.BLUE);
                    centerDot.setName("CENTER_" + (count + 1));
                    overlay.add(centerDot);

                    count++;
                }
            }

            imp.setOverlay(overlay);
            imp.updateAndDraw();

            IJ.log("Automatic detections loaded as overlay. Count: " + count);

        } catch (Exception e) {
            IJ.handleException(e);
            return -1;
        }

        return count;
    }

}