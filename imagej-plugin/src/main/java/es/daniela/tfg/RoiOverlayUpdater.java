package es.daniela.tfg;

import ij.ImagePlus;
import ij.gui.Overlay;
import ij.gui.Roi;
import ij.plugin.frame.RoiManager;

import java.awt.Color;

public class RoiOverlayUpdater {

    public static void updateOverlay(ImagePlus imp) {

        RoiManager rm = RoiManager.getInstance();
        if (rm == null) return;

        Overlay overlay = new Overlay();
        Roi[] rois = rm.getRoisAsArray();

        for (Roi roi : rois) {
            //Roi clone = (Roi) roi.clone();

            String name = roi.getName();
            if (name != null && name.startsWith("AUTO_")) {
                roi.setStrokeColor(Color.BLUE);
            } else if (name != null && name.startsWith("MANUAL_")) {
                roi.setStrokeColor(Color.GREEN);
            } else {
                roi.setStrokeColor(Color.YELLOW);
            }

            roi.setStrokeWidth(1.5);
            overlay.add(roi);
        }

        imp.setOverlay(overlay);
        imp.updateAndDraw();
    }
}