# EA-DETR: Event-Aware DEtection TRansformer

<p align="center">
  <img src="assets/ea-detr.png" width="700" height="auto">
</p>
<p align="center">
  <img src="assets/res_qual.png" width="700" height="auto">
</p>

Official implementation of EA-DETR: Event-Aware DEtection TRansformer : [Link of paper](https://hal.science/hal-05047456v1/document)

## Video

Here is the link of our video : [Link of video](https://www.youtube.com/watch?v=MgFWLxA0IeE)

## Citation

```
@inproceedings{rossi2025event,
  title={Event-Aware Distilled DETR for Object Detection in an Automotive Context},
  author={Rossi, Djessy and Vasseur, Pascal and Morbidi, Fabio and Demonceaux, C{\'e}dric and Rameau, Fran{\c{c}}ois},
  booktitle={IEEE INTELLIGENT VEHICLES SYMPOSIUM},
  year={2025}
}
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/djessy1998/EA-DETR.git
cd ea-detr
```

2. Use Conda to create the environment with the provided environment.yml file:

```bash
conda env create -f environment.yml
```

3. Activate the environment:

```bash
conda activate ea-detr
```

## Hard-DSEC-DET

Hard-DSEC-DET is available on Zenodo: [Hard-DSEC-DET](https://zenodo.org/records/20108586?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjU4Njc1MmQ1LWZmMTQtNGJkNS1hYmM5LTc4MmZlZmUzZGNjZSIsImRhdGEiOnt9LCJyYW5kb20iOiI3NGI4YzBhOTk3OTYzZDBhYjVhNzNiZmJhMDlmMGMyNyJ9.R4cAiUP0eXhDTHZ22m1co4n0Dun7PI0jbMmuDlRrDGxYCSmBNuW4LtGGibubQB0EpL5m0LhtzFYYWIS41NB85A)

## Database needed

To train or test EA-DETR, please follow the installation instructions provided on the DSEC-DET GitHub page: [DSEC-DET Installation Guide](https://github.com/uzh-rpg/dsec-det).

The official website for the database : https://dsec.ifi.uzh.ch/dsec-detection/

Then you can set the path of the DSEC-DET dataset using the option : 

```bash
python object_detection_detr.py -train -distillation -root_dsec path -dsec_det
```

## Training

You have several options for training the models, with example commands provided below:

1. **Simple DETR**: Train a basic DETR model on the DSEC-DET dataset.
   ```bash
   python object_detection_detr.py -train -dsec_det -epochs 50
   ```

2. **EA-DETR** (our proposed model): Train our Event-Aware DETR model with full distillation on the DSEC-DET dataset.
   ```bash
   python object_detection_detr.py -train -dsec-det -distillation -features -distill_logits -epochs 50
   ```

3. **EA-DETR Backbone Distillation Only**: Train the model using only backbone distillation.
   ```bash
   python object_detection_detr.py -train -dsec-det -distillation -features -epochs 50
   ```
4. **EA-DETR Logits Distillation Only**: Train the model using only logits distillation.
   ```bash
   python object_detection_detr.py -train -dsec-det -distillation -distill_logits -epochs 50
   ```

## Testing

 ```bash
 python object_detection_detr.py -test -dsec-det
 ```

