from jointtrainer_multilang import JointModel, parse_args, JointTrainer
import torch

args = parse_args()
joint_trainer = JointTrainer(args)
joint_trainer.init_models()

crf_weights = joint_trainer.jointmodel.nermodel.crf.emission.weight


print("My crf weights before")
print(crf_weights)


load_path = args["load_path"]
joint_trainer.jointmodel.load_state_dict(torch.load(load_path)
                                         
crf_weights = joint_trainer.jointmodel.nermodel.crf.emission.weight
print("My crf weights after")
print(crf_weights)