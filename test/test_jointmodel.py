from jointtrainer_multilang import JointModel, parse_args, JointTrainer


args = parse_args()
joint_trainer = JointTrainer(args)
joint_trainer.init_models()

state_dict = joint_trainer.jointmodel.state_dict()

print("My state dictionary")
print(state_dict)
