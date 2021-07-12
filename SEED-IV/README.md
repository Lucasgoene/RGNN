# How to fill this folder?

This folder should represent the [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/seed-iv.html) dataset as downloaded and de-compressed from the official website.

The 3 files already present in the folder are our addition:
- the two **JSON** file represents a dump of the final Python dictionary as computed by the two training functions
  - the **Subject-Dependent** one is trained over all 15 subjects for 100 epochs
  - the **Subject-Independent** one is trained by cross-validation on a single model (not all 15, for time constraints), leaving the first subject out for testing
- the **channel_locations.txt** is a required file, used to correctly compute the *Adjacency Matrix*; it needs a double-check on the correct origin of the file, intended to be a representation of the EEG headset's datasheet.