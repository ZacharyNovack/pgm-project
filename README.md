The Code for this project mostly comes from the open source code from "FlipTest: Fairness Testing via Optimal Transport" and "Toward generating actionable counterfactual explanations via posterior inference" (repo not public).

Reproducing results from the paper can be done as follows:

First, in the recourse/src/ directory, run 
```
python gen_splitdata.py --config dataset.config
```
With the appropriate config file to generate the data needed for the OT. 

Then in the ot/, run through the matching .ipynb file for each dataset to run the optimal transport approximation and save the model.

After that, you can go back to the recourse/src/ directory and run
```
python generate_paired_counterf.py --config dataset.config
```
for SSL or Lipton and
```
python generate_paired_counterf_german.py --config german.config
```
for german to run the disparity analyses and write out all the information.

To generate the final plots, you can go down the bottom of the requisite get_idxs_dataset.ipynb file and run the final few chunks.