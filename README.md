# vkr
HSE Master's Deegree Diploma

Requirements

1. BigARTM version 0.7.1
2. Linux or Windows (was tested on Ubuntu 14.04 LTS x64 and Windows 8.1 x64 )

How to use?

1. git clone
2. checkout branch dataProcessing TODO may be changed in release
    - run init.py
3. Download ARFF data (texts.zip from Google Drive and extract it)
4. For example mkdir -p ./rawdata/OHSUMED/arff/ and cp OHSUMED arff files in it e.g. quant_OHSUMED_test_87.txt
5. Launch arff2uci.py to convert ARFF to BigARTM UCI format
    - if sklearn error occures - comment unused includes or update/reinstall it
6. Launch multimodal_simple.py for checking BigARTM work
    - if you have such message

      libartm.so: cannot open shared object file: No such file or directory,
      fall back to ARTM_SHARED_LIBRARY environment variable

      you should check env that ARTM_SHARED_LIBRARY path is correct, and if yes,
      probably you have old version of BigARTM, so you need 0.7.1

    - other errors in code probably occures due to old version of BigARTM too, so update it

    - Get BigARTM 0.7.1 from here https://github.com/bigartm/bigartm/releases/tag/v0.7.1
7. Launch multimodal_uci.py - base module for multimodal experiments with collections