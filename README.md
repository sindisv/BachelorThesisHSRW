Thesis: Design and Development of a Cost-Effective Multi-Sensor Wearable System for Real-Time Biomechanical Analysis in Boxing

This repository contains the files and assets for the thesis. The top-level folders hold CAD models, electrical schematics, firmware, training data and a packaged software application. The application distributed with this repository is provided as a zip file.



Folder summary

* Software Arduino Sketches: Arduino/embedded firmware source code for the different device variants (sketch\_device1..4). Use the Arduino IDE or compatible toolchain to open and upload the .ino sketches to the target boards.
* Software Application: Packaged application for the project. The file My Application.zip contains the compiled application or installers/resources. Unzip the archive (for example with Explorer, 7-Zip, or unzip) and follow the included instructions or run the extracted executable where appropriate.
* CAD Files Fusion360: Fusion360 3D model files, STEP files and drawing exports. Subfolders separate 3D files and drawing sheets for each device/unit. These are used for enclosure and mechanical design.
* Electrical Schematics KiCad: KiCad schematic files for each device (Device 1..4) and custom symbols. Open these with KiCad to view or modify the electrical designs and generate PCBs.
* Training Sessions CSV and XLSX Files: CSV & XLSX datasets collected during training sessions. These contain recorded sensor data for specific motions and can be used for analysis, training models or evaluation.
* Software Machine Learning: python scripts for running, training and processing the machine learning.


Notes

* The repository does not currently include an installed application — the application is provided zipped under Software Application/My Application.zip. Extract that archive to access the application files.
* Inspect each folder for README or instruction files specific to that component (for example CAD export notes or KiCad project details).



Important files and expanded per-folder details

1. Software Arduino Sketches
* sketch\_device1/sketch\_device1.ino — firmware for device 1
* sketch\_device2/sketch\_device2.ino — firmware for device 2
* sketch\_device3/sketch\_device3.ino — firmware for device 3
* sketch\_device4/sketch\_device4.ino — firmware for device 4
Notes: open the appropriate .ino in the Arduino IDE or PlatformIO. Check board and library requirements in the top comments of each sketch (if present) before uploading.
2. Software Application



* My Application.zip (located in Software Application/) — packaged application bundle. Extract to view installers, binaries or resource files.



3. CAD Files Fusion360
* 3D Files Fusion360/ (contains .step STEP files for enclosures, batteries, PCB models and assembly files)
* Drawings Fusion360/ (contains .dwg and .pdf drawing exports for manufacturing and assembly)
Examples: "Enclosure Unit 1.step", "Assembly Unit 1.step", and the exported PDF drawings such as "Assembly Drawing and Parts List Unit 1.pdf".
Notes: STEP files can be opened in Fusion360, FreeCAD or other CAD tools. DWG files require a CAD viewer that supports AutoCAD formats.



4. Electrical Schematics KiCad
* Circuit Device 1.kicad\_sch
* Circuit Device 2.kicad\_sch
* Circuit Device 3 and Device 4.kicad\_sch
* XIAO\_Series\_SCH\_Symbols/.../Seeed\_Studio\_XIAO\_Series.kicad\_sym (custom symbols)
Notes: Open these in KiCad to inspect or export BOM/PCB layouts. Be aware of the KiCad version used to create the project if you encounter compatibility warnings.



5. Training Sessions CSV Files
* Training Sessions CSV Files/Session 2 (Specific Motions)/Device1.csv
* Device1Uppercut.csv, Device1hookfront.csv, Device3Jab.csv, Device3uppercut.csv, Boxinghookdevice3.csv
Notes: These CSV files contain recorded sensor traces from devices during motion sessions. They can be loaded into Python, MATLAB, or Excel for analysis and model training.

6. Software Machine Learning
There is a dedicated Machine Learning workspace under "Software Machine Learning/Software Machine Learning" and processed ML outputs under "Training Sessions CSV and XLSX Files/ml_complete". This area contains scripts to label data, extract features, train classifiers and produce evaluation outputs used in the thesis.
Important scripts (Software Machine Learning/Software Machine Learning):
 - label_boxing_data.py — helpers for labeling events/windows in raw sensor traces
 - preprocess.py — feature extraction and dataset assembly (produces ml_data/features_all.csv)
 - train_model.py — training pipeline: trains multiple classifiers, evaluates with cross-validation, generates plots and saves the best model
 - quick_analysis.py — small exploratory analysis and plotting utilities
 - boxing_analysis_complete6.py — more comprehensive analysis script used for experiments
