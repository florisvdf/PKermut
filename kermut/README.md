---
# Model identifier used for referencing this model in the benchmark system
name: "kermut"

tags: ["supervised"]

hyper_parameters:
    # WIP: Below should be handled by pg dataset
    dataset_name: RASK_HUMAN_Weng_2022_binding-DARPin_K55 
    reference_sequence: MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEYSAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTKQAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM
    data_path: data
    artifact_uri: null
    
    n_steps: 150
    device: cpu
---

# Model Card for Kermut

[Kermut](https://arxiv.org/abs/2407.00002v3. This repository is a fork of the original Kermut repository without any 
alterations to the model implementation. Only changes made are the addition of an entrypoint and hydra configuration for 
data loading. Adding Kermut to PG is a WIP.