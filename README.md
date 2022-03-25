# ball3d

# Installation
```
cd ball3d
```

## Create a dedicate environment (recommended)
```
conda create --name ball3d python=3.8
```

## Intsall dependencies (required)
```
pip install -e .
```

## Set env (required)
```
cp .env.template .env
sed 's/TODO/`pwd`/g' .env
```
You should update `.env` file based on your environment


# Dataset



# Training
```
python -m experimentator configs/ballsize.py --epochs 101 --kwargs "eval_epochs=range(0,101,20)"
```


# Evaluation
comming soon