package es.daniela.tfg;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.Roi;
import ij.io.SaveDialog;
import ij.plugin.frame.RoiManager;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

import java.io.FileWriter;
import java.io.PrintWriter;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Export Annotations")
public class ExportAnnotationsCommand implements Command {

    @Override
    public void run() {
        try {
            ImagePlus imp = WindowManager.getCurrentImage();
            if (imp == null) {
                IJ.error("No image is open.");
                return;
            }

            RoiManager rm = RoiManager.getInstance();
            if (rm == null || rm.getCount() == 0) {
                IJ.error("No ROIs found in ROI Manager.");
                return;
            }

            SaveDialog sd = new SaveDialog("Save annotations", imp.getTitle() + "_annotations", ".csv");
            String dir = sd.getDirectory();
            String name = sd.getFileName();

            if (dir == null || name == null) {
                return;
            }

            Roi[] rois = rm.getRoisAsArray();

            try (PrintWriter pw = new PrintWriter(new FileWriter(dir + name))) {
//                pw.println("image_name,roi_index,roi_name,cx,cy,radius");
//
//                for (int i = 0; i < rois.length; i++) {
//
//                    Roi roi = rois[i];
//                    var bounds = roi.getBounds();
//                    String roiName = roi.getName() != null ? roi.getName() : "";
//
//                    double cx = bounds.x + bounds.width / 2.0;
//                    double cy = bounds.y + bounds.height / 2.0;
//                    double r = (bounds.width + bounds.height) / 4.0;
//
//                    //pw.printf("%s,%d,%s,%.2f,%.2f,%.2f%n",
//                    pw.printf(java.util.Locale.US,
//                            "%s,%d,%s,%.2f,%.2f,%.2f%n",
//                            imp.getTitle(),
//                            i,
//                            roiName,
//                            cx,
//                            cy,
//                            r);
//                }

                pw.println("image_name,roi_index,roi_name,cx,cy,width,height");

                for (int i = 0; i < rois.length; i++) {

                    Roi roi = rois[i];
                    String roiName = roi.getName() != null ? roi.getName() : "";

                    var bounds = roi.getBounds();

                    double cx = bounds.x + bounds.width / 2.0;
                    double cy = bounds.y + bounds.height / 2.0;
                    double w = bounds.width;
                    double h = bounds.height;

                    pw.printf(java.util.Locale.US,
                            "%s,%d,%s,%.2f,%.2f,%.2f,%.2f%n",
                            imp.getTitle(),
                            i,
                            roiName,
                            cx,
                            cy,
                            w,
                            h);
                }
            }

            IJ.showMessage("Export completed", "Annotations exported successfully.");

        } catch (Exception e) {
            IJ.handleException(e);
        }
    }
}