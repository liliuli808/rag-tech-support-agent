---
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- dense
- generated_from_trainer
- dataset_size:20
- loss:MultipleNegativesRankingLoss
base_model: BAAI/bge-small-en-v1.5
widget:
- source_sentence: Who do I call for an urgent outage?
  sentences:
  - 'Level 1 handles password resets, account unlocks, VPN client guidance, portal

    navigation and documented error codes. Level 2 handles server-side incidents and

    data-level corrections. Level 3 handles vendor escalations and release defects.'
  - 'User data on managed laptops is synchronised continuously to the corporate cloud

    drive. Local backup copies are taken every 4 hours by the backup agent.


    To restore a file, right-click the file or folder in the cloud drive web

    interface, choose "Version history", select the required timestamp and click

    Restore. Deleted items are retained in the recycle bin for 90 days.'
  - 'The password must contain at least 12 characters and include characters from
    at

    least three of the following categories: uppercase letters, lowercase letters,

    digits and special symbols. Passwords expire every 180 days. The last 10

    passwords cannot be reused. A password may not contain the username, the

    employee ID, or any of the company names.'
- source_sentence: What counts as a Priority 1 incident?
  sentences:
  - 'The outgoing mail server is smtp.northwind.example on port 587 with STARTTLS.

    Port 25 is blocked for all clients. The incoming server is

    imap.northwind.example on port 993 with SSL.


    If the customer reports "550 5.7.1 Message rejected as spam", ask them to check

    that the message does not contain an unusually large number of external links

    and that the attachment size is below the 25 MB limit.'
  - 'The supported VPN client is Northwind Connect 3.x for Windows and macOS. Legacy

    clients 2.x are no longer supported because they use an obsolete cipher suite.


    The most frequent VPN error is "Authentication failed (E-1042)". This means the

    MFA challenge was not completed within 60 seconds. Ask the customer to start the

    connection again and to approve the push notification promptly.'
  - 'Level 1 handles password resets, account unlocks, VPN client guidance, portal

    navigation and documented error codes. Level 2 handles server-side incidents and

    data-level corrections. Level 3 handles vendor escalations and release defects.'
- source_sentence: My account got locked out
  sentences:
  - 'Every employee receives a single corporate identity (SSO) in the format

    firstname.lastname@northwind.example. This identity is used for the intranet

    portal, the email system, the VPN and all cloud applications. Accounts are

    provisioned automatically 24 hours after the HR onboarding record is approved.'
  - 'User data on managed laptops is synchronised continuously to the corporate cloud

    drive. Local backup copies are taken every 4 hours by the backup agent.


    To restore a file, right-click the file or folder in the cloud drive web

    interface, choose "Version history", select the required timestamp and click

    Restore. Deleted items are retained in the recycle bin for 90 days.'
  - 'Standard software is deployed from the Company Portal. Applications that are
    not

    in the Company Portal require a software exception request approved by the

    line manager and by information security.


    A licence activation error "LIC-007 Seat limit reached" means all purchased seats

    are in use. Ask the customer to sign out of unused devices, or open a ticket for

    the procurement team to add seats.'
- source_sentence: I cannot log in to the intranet portal
  sentences:
  - 'Every employee receives a single corporate identity (SSO) in the format

    firstname.lastname@northwind.example. This identity is used for the intranet

    portal, the email system, the VPN and all cloud applications. Accounts are

    provisioned automatically 24 hours after the HR onboarding record is approved.'
  - 'A 400 Bad Request means the server could not understand the request because of

    invalid syntax. Ask the customer to clear the browser cache and cookies, and to

    verify that the URL has no trailing spaces.


    A 401 Unauthorized means the credentials were missing or invalid. A 403

    Forbidden means the credentials were valid but the account lacks the required

    permission; raise an access request in the IAM portal.'
  - 'A 400 Bad Request means the server could not understand the request because of

    invalid syntax. Ask the customer to clear the browser cache and cookies, and to

    verify that the URL has no trailing spaces.


    A 401 Unauthorized means the credentials were missing or invalid. A 403

    Forbidden means the credentials were valid but the account lacks the required

    permission; raise an access request in the IAM portal.'
- source_sentence: How often must I change my password?
  sentences:
  - 'The supported VPN client is Northwind Connect 3.x for Windows and macOS. Legacy

    clients 2.x are no longer supported because they use an obsolete cipher suite.


    The most frequent VPN error is "Authentication failed (E-1042)". This means the

    MFA challenge was not completed within 60 seconds. Ask the customer to start the

    connection again and to approve the push notification promptly.'
  - 'The outgoing mail server is smtp.northwind.example on port 587 with STARTTLS.

    Port 25 is blocked for all clients. The incoming server is

    imap.northwind.example on port 993 with SSL.


    If the customer reports "550 5.7.1 Message rejected as spam", ask them to check

    that the message does not contain an unusually large number of external links

    and that the attachment size is below the 25 MB limit.'
  - 'The password must contain at least 12 characters and include characters from
    at

    least three of the following categories: uppercase letters, lowercase letters,

    digits and special symbols. Passwords expire every 180 days. The last 10

    passwords cannot be reused. A password may not contain the username, the

    employee ID, or any of the company names.'
pipeline_tag: sentence-similarity
library_name: sentence-transformers
---

# SentenceTransformer based on BAAI/bge-small-en-v1.5

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). It maps inputs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) <!-- at revision 5c38ec7c405ec4b44b94cc5a9bb96e735b38267a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 dimensions
- **Similarity Function:** Cosine Similarity
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'transformer_task': 'feature-extraction', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'last_hidden_state'}}, 'module_output_name': 'token_embeddings', 'architecture': 'BertModel'})
  (1): Pooling({'embedding_dimension': 384, 'pooling_mode': 'cls', 'include_prompt': True})
  (2): Normalize({'module_input_name': 'sentence_embedding', 'module_output_name': 'sentence_embedding'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```
Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'How often must I change my password?',
    'The password must contain at least 12 characters and include characters from at\nleast three of the following categories: uppercase letters, lowercase letters,\ndigits and special symbols. Passwords expire every 180 days. The last 10\npasswords cannot be reused. A password may not contain the username, the\nemployee ID, or any of the company names.',
    'The supported VPN client is Northwind Connect 3.x for Windows and macOS. Legacy\nclients 2.x are no longer supported because they use an obsolete cipher suite.\n\nThe most frequent VPN error is "Authentication failed (E-1042)". This means the\nMFA challenge was not completed within 60 seconds. Ask the customer to start the\nconnection again and to approve the push notification promptly.',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7883, 0.5325],
#         [0.7883, 1.0000, 0.5040],
#         [0.5325, 0.5040, 1.0000]])
```
<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 20 training samples
* Columns: <code>sentence_0</code> and <code>sentence_1</code>
* Approximate statistics based on the first 20 samples:
  |          | sentence_0                                                                      | sentence_1                                                                         |
  |:---------|:--------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------|
  | type     | string                                                                          | string                                                                             |
  | modality | text                                                                            | text                                                                               |
  | details  | <ul><li>min: 6 tokens</li><li>mean: 9.7 tokens</li><li>max: 15 tokens</li></ul> | <ul><li>min: 49 tokens</li><li>mean: 81.6 tokens</li><li>max: 110 tokens</li></ul> |
* Samples:
  | sentence_0                                               | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  |:---------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
  | <code>Who do I call for an urgent outage?</code>         | <code>Level 1 handles password resets, account unlocks, VPN client guidance, portal<br>navigation and documented error codes. Level 2 handles server-side incidents and<br>data-level corrections. Level 3 handles vendor escalations and release defects.</code>                                                                                                                                                                          |
  | <code>Email from external senders is not arriving</code> | <code>The outgoing mail server is smtp.northwind.example on port 587 with STARTTLS.<br>Port 25 is blocked for all clients. The incoming server is<br>imap.northwind.example on port 993 with SSL.<br><br>If the customer reports "550 5.7.1 Message rejected as spam", ask them to check<br>that the message does not contain an unusually large number of external links<br>and that the attachment size is below the 25 MB limit.</code> |
  | <code>I cannot log in to the intranet portal</code>      | <code>Every employee receives a single corporate identity (SSO) in the format<br>firstname.lastname@northwind.example. This identity is used for the intranet<br>portal, the email system, the VPN and all cloud applications. Accounts are<br>provisioned automatically 24 hours after the HR onboarding record is approved.</code>                                                                                                       |
* Loss: [<code>MultipleNegativesRankingLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#multiplenegativesrankingloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "cos_sim",
      "gather_across_devices": false,
      "directions": [
          "query_to_doc"
      ],
      "partition_mode": "joint",
      "hardness_mode": null,
      "hardness_strength": 0.0
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `num_train_epochs`: 4
- `disable_tqdm`: True
- `multi_dataset_batch_sampler`: round_robin

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 8
- `num_train_epochs`: 4
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: True
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 8
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `dataloader_multiprocessing_context`: None
- `dataloader_in_order`: True
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: None
- `fsdp_config`: None
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: round_robin
- `router_mapping`: {}
- `learning_rate_mapping`: {}
- `warmup_ratio`: None

</details>

### Training Time
- **Training**: 20.3 seconds

### Framework Versions
- Python: 3.12.3
- Sentence Transformers: 6.1.0
- Transformers: 5.17.0
- PyTorch: 2.14.0+cpu
- Accelerate: 1.15.0
- Datasets: 5.0.1
- Tokenizers: 0.23.2

## Additional Resources

- [Training and Finetuning Embedding Models with Sentence Transformers](https://huggingface.co/blog/train-sentence-transformers): the end-to-end guide for training or finetuning Sentence Transformer models.
- [Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka): variable-size embeddings that can be truncated with minimal quality loss.
- [Binary and Scalar Embedding Quantization for Significantly Faster & Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization): post-training compression of embedding vectors.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): train multimodal embedding models, with a Visual Document Retrieval walkthrough.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### MultipleNegativesRankingLoss
```bibtex
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->