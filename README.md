# center-finder-2.0

To test run the voting, sampling and blobbing procedures, call the --test or -t argument followed by the desired test radius:
```
python cf.py cf_mock_catalog_667C_15R_randoms_added.fits --test 108
```

Add the --save or -s argument to save outputs in a new file called "saves" (created automatically inside current directory):
```
python cf.py cf_mock_catalog_83C_200R.fits --test 108 --save
```

Add the --params_file or -p argument to change the default file from which the cosmological parameters are loaded to a new file whose name is entered after the argument:
```
python cf.py cf_set_8_mock_1_46C.fits --test 108 --save --params_file set_8_params.json
```
