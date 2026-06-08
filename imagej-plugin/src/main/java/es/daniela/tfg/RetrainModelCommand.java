package es.daniela.tfg;

import ij.IJ;
import org.scijava.command.Command;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, menuPath = "Plugins>TFG>Retrain Model (Transfer Learning)")
public class RetrainModelCommand implements Command {

    @Override
    public void run() {
        IJ.showMessage(
                "Transfer Learning",
                "This module will retrain the neuron segmentation model\n" +
                        "using user-corrected annotations exported from Fiji.\n\n" +
                        "Current status: interface prepared, training pipeline pending implementation."
        );
    }
}