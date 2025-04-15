## How to Run

### Generate sub-dataset for ece cluster
Requirement: Must have the original dataset in the root folder named "imagenette2_full" (Download from https://github.com/fastai/imagenette)
```
# update the value of NUM_IMAGES to target value
python subset.py
```

### Generate shuffled image list
```
python makelist.py
```

### Compile
```
make clean && make all
```

### Run
```
./alexnet train -batchsize 40 -epochs 10
# can save temp weights with: -save ./temp.weights 
```