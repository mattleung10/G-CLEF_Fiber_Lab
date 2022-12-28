April 2022

Exploratory code for how to continuously acquire images using mvIMPACT Python library.
Useful for reference.
FINAL WORKING SCRIPT: mv_acquire_test_cts_ac.py (April 12, 2022)

------------------------------------------------------------------------------------------------

KEY CONCLUSIONS for mvIMPACT Python library with mvBlueCOUGAR-X102kG:
    - Do not use Device Specific Interface Layout
    - Use GenICam Interface Layout instead
    - Use acquire.AcquisitionControl, acquire.AnalogControl, and acquire.ImageFormatControl
      instead of acquire.CameraSettingsBlueCOUGAR
    - Under no circumstances should you run wxPropView before running the code
