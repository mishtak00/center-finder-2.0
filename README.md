# center-finder-2.0

Clone this repository by running:
```
git clone https://github.com/mishtak00/center-finder-2.0
```

Create, activate and configure a virtual environment that's ready for center-finder-2.0 by running the following from the terminal while inside the center-finder-2.0 directory:
```
python -m venv virtualenv
source virtualenv/bin/activate
pip install -r requirements.txt
```

To test run the voting, sampling, blobbing and refinement procedures, call the --test or -t argument followed by the desired test radius:
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

Add the --printout or -o argument to have sanity checks and feedback on the running process printed to standard output as the program runs in exchange for *slightly* degraded performance (some of these checks are sums over large grids):
```
python cf.py cf_mock_catalog_83C_200R.fits --test 108 --printout
```
