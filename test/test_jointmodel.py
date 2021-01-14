from jointtrainer_multilang import JointModel, parse_args, JointTrainer
import torch

args = parse_args()
joint_trainer = JointTrainer(args)
joint_trainer.init_models()

def test_dep_module():
    hlstm = joint_trainer.jointmodel.depparser.highwaylstm.lstm[0].weight_ih_l
    print("My hlstm weights before")
    print(hlstm)
    load_path = args["load_path"]
    joint_trainer.jointmodel.load_state_dict(torch.load(load_path))

    hlstm = joint_trainer.jointmodel.depparser.highwaylstm.lstm[0].weight_ih_l
    print("My hlstm weights after")
    print(hlstm)


def test_ner_module():
    crf_w = joint_trainer.jointmodel.nermodel.crf.emission.weight
    print("My crf weights before")
    print(crf_w)

    load_path = args["load_path"]
    joint_trainer.jointmodel.load_state_dict(torch.load(load_path))

    crf_w = joint_trainer.jointmodel.nermodel.crf.emission.weight
    print("My crf weights after")
    print(crf_w)

test_ner_module()
test_dep_module()