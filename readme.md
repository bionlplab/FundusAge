### Installation
Set up the environment with **Python 3.12**:
```bash
conda create -n xxx python=3.12
```
Install additional dependencies:
```bash
pip install -r rq.txt
```
###  Verify the intalled environment
Here is the 5-fold cross validation results on AREDS Dataset
| Backbone | Fold   | Phenotypes | Demographic | Age (Loss)        | C‑Index |
|----------|--------|------------|-------------|-------------------|---------|
| VMamba   | fold‑1 | Yes        | smk         | Pred (MAE + CoxPH) | 0.908   |
| VMamba   | fold‑2 | Yes        | smk         | Pred (MAE + CoxPH) | 0.909   |
| VMamba   | fold‑3 | Yes        | smk         | Pred (MAE + CoxPH) | 0.910   |
| VMamba   | fold‑4 | Yes        | smk         | Pred (MAE + CoxPH) | 0.911   |
| VMamba   | fold‑5 | Yes        | smk         | Pred (MAE + CoxPH) | 0.911   |
| VMamba   | all    | Yes        | smk         | Pred (MAE + CoxPH) | 0.922   |
###  Start with AREDS2
![Model Architecture](./Picture1.png)
### Data Preparation

To evaluate the **AREDS2 test set** with the Cox model  
(`d_age_cox.py`, `cox` function — see *line 423*), the input
table **must** contain the following columns:

| Category       | Field(s)                               | Notes / Expected format                    |
|----------------|----------------------------------------|-------------------------------------------|
| **Demographics** | `age`, `smk`, `gender`, `race`, `school` | - `age`: numerical (predicted or chronological)  <br>- `smk`: smoking status  <br>- `gender`: `1` / `0` (optional)  <br>- `race`: categorical code (optional) <br>- `school`: education‑level code (optional) |
| **AMD Phenotypes** | `DRSZWI`, `INCPWI`, `RPEDWI`, `GEOAWI`, `DRSOFT`, `DRARWI` | Six AMD phenotypes required by the model |

> Ensure all required fields are present and free of missing values; otherwise  
> the Cox routine will raise a validation error.

You can first build a Cox model on the AREDS2 dataset to validate the effects of these factors (using chronological age). The following table presents the results on AREDS:

| Backbone      | Param (M) | Training Time (h) | Phenotypes | Demographic       | Age          | C‑Index |
|---------------|-----------|-------------------|------------|-------------------|--------------|---------|
| —             | —         | —                 | Yes        | smk + race + school | Real Age     | 0.899 |
| —             | —         | —                 | Yes        | smk               | Real Age     | 0.891 |
| —             | —         | —                 | Yes        | —                 | Real Age     | 0.876 |

### Inference: Predicted Age

To generate predicted age values:

1. **Prepare Checkpoints**  
   Place all model checkpoints in the `./result_p` directory.

2. **Run Prediction**  
   Execute the following command to get predictions for each fold (X = 1–5):

   ```bash
   python d_age_cox.py --train_fold X
   ```
   After successful execution, a file named `fold_X_test.csv` will be generated in the root directory.
3. **AREDS2 Data Matching by ID**
    The image loading is based on patient/image IDs. Please ensure you prepare the correct CSV file with IDs matching the image filenames(see `AMD_gen_52new.csv` & `patient_gen_52.csv`), I get the info using **get_id()** function in `data_id_age_cox.py`

    Modify the Following Lines If Needed. To ensure correct image loading paths, you may need to adjust:

    - `d_age_cox.py`: lines 562, 566

    - `dataset_age_cox.py`: line 33

    For example, image loading is defined as:
    ```python
    X = Image.open('/prj0129/mil4012/AREDS/AMD_224/' + ID[:-4] + '.jpg')
    ```
    Make sure this path and file format match your local data setup.

4. **Run Cox Evaluation on Predicted Results**

After generating all 5 folds of test results, run `cox_test.py` to evaluate each fold's Cox performance.

- Use the `test_fold` variable at **line 17** of `cox_test.py` to specify which fold to evaluate:
  - `test_fold = 1` to `5`: evaluate individual folds
  - `test_fold = 0`: evaluate the **ensemble result** by combining predictions from all five folds

```python
# Example (cox_test.py, line 17)
test_fold = 0  # use 0 for ensemble prediction
```