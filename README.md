# llm-momomood

# How to run

This serve as a guide to try out the LLM DP project. This guides show you how to (1) create the prompts from your own data and (2) run inference with different prompt strategies.

Under `data` folder, there are 4 sets of prompts available. The filenames are self-explanatory, so use them at you like.


## Run inference

This project uses the llama-3.1 8b. Inference is implemented in `fine_tuning_llama3_phq.py`. 

A bash script is ready for use (look into `run.sh`). Run it on Triton using the command `sbatch run.sh`. 
