# Diploma Thesis Code: "Sarcasm Detection with Transfer Learning from Multiple Sources" in the Slovene Language

In this repository, we present the code used in the diploma thesis **"Sarcasm detection with transfer learning from multiple sources"**. The focus of this thesis was sarcasm detection in the Slovene language. Various large language models were used to classify the iSarcasmEval dataset, exploring models of differing sizes and architectures. Additionally, individual models were combined into ensembles to achieve higher performance.

## Repository Structure

The repository is organized as follows:

### 1. `datasets/`
- Contains the original iSarcasmEval dataset and translations obtained using MADLAD-T5 and ChatGPT-4o models.
- Includes code for analyzing and plotting text length distribution in the dataset.
- Contains scripts for splitting the dataset into training, validation, and test sets.

### 2. `small_models/`
- Contains the code for testing small BERT and RoBERTa-like models.
- Utilizes Hugging Face's *transformers* library for fine-tuning the models.
- Includes scripts for calculating model scores using *scikit-learn*.

### 3. `openai_api/`
- Contains code for accessing OpenAI's services, fine-tuning GPT-3 and GPT-4o models.
- Includes preprocessing scripts to format the data for the API.

### 4. `large_models/`
- Stores code for inferencing and fine-tuning generative Llama-based models.
- Utilizes the `SFTTrainer` class from the *transformers* library for fine-tuning.
- Contains preprocessing scripts and SLURM sbatch files used for running training in a SLURM environment.

### 5. `ensembles/`
- Contains code for combining individual models into ensembles.
- We tried hard, soft, and mixed voting ensembles, as well as stacking models using L2 logistic regression as the meta-model.

### Results
- The `small_models/`, `openai_api/`, and `large_models/` directories contain a `results/` subfolder with predictions from individual models.
- The `ensembles/combined_model_predictions/` directory stores the combined predictions from the individual models.

## License

This work is licensed under the *Creative Commons Zero v1.0 Universal* LICENSE.

## Citation

If you wish to cite this work, please use the following citation:

*TBD*