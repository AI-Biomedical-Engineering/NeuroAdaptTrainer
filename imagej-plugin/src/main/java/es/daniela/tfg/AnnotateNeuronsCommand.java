package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.OvalRoi;
import ij.plugin.frame.RoiManager;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Annotate Neurons")
public class AnnotateNeuronsCommand implements Command {

    @Override
    public void run() {

        ImagePlus imp = WindowManager.getCurrentImage();

        if (imp == null) {
            IJ.error("No image open");
            return;
        }

        RoiManager rm = RoiManager.getInstance();
        if (rm == null) {
            rm = new RoiManager();
        }

        IJ.setTool("oval");

        IJ.showMessage(
                "Neuron Annotation",
                "Draw neurons using the oval tool.\n" +
                        "Press 't' to add them to the ROI Manager.\n\n" +
                        "Manual annotations will appear automatically in green."
        );

        // Show Auto detection
        loadDetectionsAsROIs(imp);

        RoiManager finalRm = rm;

        new Thread(() -> {
            int lastCount = -1;

            while (true) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    return;
                }

                int currentCount = finalRm.getCount();

                if (currentCount != lastCount) {
                    boolean updated = false;
                    // Rename new ROIs as MANUAL_x if they do not already have a type
                    for (int i = 0; i < finalRm.getCount(); i++) {
                        String roiName = finalRm.getName(i);

                        if (roiName == null ||
                                roiName.isBlank() ||
                                (!roiName.startsWith("AUTO_") && !roiName.startsWith("MANUAL_"))) {

                            finalRm.rename(i, "MANUAL_" + (i + 1));
                            updated = true;
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

    private void loadDetectionsAsROIs(ImagePlus imp) {
        try {
            String tmpDir = System.getProperty("java.io.tmpdir");
            File csvFile = new File(tmpDir, "imagej_output.csv");

            if (!csvFile.exists()) {
                IJ.log("CSV not found: " + csvFile.getAbsolutePath());
                return;
            }

            RoiManager rm = RoiManager.getInstance();
            if (rm == null) rm = new RoiManager();

            // Remove previous automatic ROIs detections only
            for (int i = rm.getCount() - 1; i >= 0; i--) {
                String roiName = rm.getName(i);
                if (roiName != null && roiName.startsWith("AUTO_")) {
                    rm.select(i);
                    rm.runCommand("Delete");
                }
            }

            BufferedReader br = new BufferedReader(new FileReader(csvFile));
            String line = br.readLine(); // skip header

            int autoIndex = 1;

            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");

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

            br.close();

            RoiOverlayUpdater.updateOverlay(imp);

            IJ.log("Automatic detections loaded from CSV");

        } catch (Exception e) {
            IJ.handleException(e);
        }
    }
}