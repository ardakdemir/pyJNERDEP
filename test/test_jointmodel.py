from jointtrainer_multilang import JointModel, parse_args, JointTrainer


args = parse_args()
joint_trainer = JointTrainer(args)
joint_trainer.init_models()

crf_weights = joint_trainer.jointmodel.nermodel.crf.emission.weight

print("My crf weights")
print(crf_weights)
