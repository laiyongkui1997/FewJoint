# FewJoint

The code of [FewJoint](https://arxiv.org/abs/2009.08138).

## Scripts

- `scripts`
	- `single_intent_1_bert.sh`: run for single intent detection few-shot model
	- `single_slot_1_bert.sh`: run for single slot filling few-shot model
	- `joint_slu_1_bert.sh`: run for joint-slu few-shot model
	- `few_joint_slu_1_bert.sh`: run for the our main method
	- `finetune_joint_slu_1_bert.sh`: run for joint-slu few-shot model with fine-tuning first
	
**Tips: You need change these parameters in scripts:**   
- `pretrained_model_path`
- `pretrained_vocab_path`
- `base_data_dir`
- `data_dir`
