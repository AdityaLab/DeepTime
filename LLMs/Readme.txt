This folder contains instructions on how to use LLM to reason about time-series, include VL-Time and traditional numerical modeling methods.

Step 1:
Install the environment by running:

pip3 install -r ENV_PROMPT.txt
Step 2:

Place the processed data in the Dataset folder. Currently, data for visual modeling is provided.

Step 3:
Run the LLM to perform time-series inference and evaluation.
Please refer to the Scripts folder:

When modal="L", traditional numerical modeling is used.
When modal="V", VL-Time visual modeling is used.

Set --model to use differentt LLM
Set --num_shot_per_class=0 use zero-shot reasoning setting; >0 to use few-shot setting
Set ----hint="Please solve this proble step by step" to use chain-of-thought reasoning setting.

