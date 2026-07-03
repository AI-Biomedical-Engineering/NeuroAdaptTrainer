# Installers

This folder contains the installers and the compiled Fiji/ImageJ plugin `.jar` file.

## Contents

```text
installers/
├── mac/
│   ├── install_mac.sh
│   └── NeuronSegmentationInstaller.app
├── windows/
│   ├── install_windows.ps1
│   └── NeuronSegmentationInstaller.exe
└── imagej-plugin-0.1.0-SNAPSHOT.jar
```

## Recommended installation

The recommended option is to use the installer for the corresponding operating system.

### macOS

Run the macOS installer application:

```text
mac/NeuronSegmentationInstaller.app
```

Alternatively, run the installation script manually:

```bash
cd mac
chmod +x install_mac.sh
./install_mac.sh
```

### Windows

Run the Windows installer executable:

```text
windows/NeuronSegmentationInstaller.exe
```

Alternatively, run the PowerShell script manually:

```powershell
cd windows
.\install_windows.ps1
```

The installer prepares the Python environment, installs the required dependencies and writes the configuration file used by the plugin.

## Manual plugin installation

The compiled Fiji/ImageJ plugin is:

```text
imagej-plugin-0.1.0-SNAPSHOT.jar
```

To install the plugin manually, copy this `.jar` file into the Fiji/ImageJ `plugins` folder.

Example on macOS:

```text
Fiji.app/plugins/
```

Example on Windows:

```text
Fiji.app\plugins\
```

After copying the `.jar`, restart Fiji/ImageJ.

The plugin should appear in the Fiji/ImageJ menu under:

```text
Plugins > Neuron Segmentation
```

## Important note

Manual installation only copies the Java plugin into Fiji/ImageJ.

The Python environment, required dependencies and configuration file must still be prepared separately if the installer is not used.