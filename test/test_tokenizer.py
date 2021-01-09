from transformers import AutoTokenizer, AutoModel

import argparse


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load_model_path", default="key_models/mybiobert_finetunedsquad", type=str, required=False,
        help="The path to load the model to continue training."
    )

    args = parser.parse_args()
    args.device = device
    return args


sent_dict = {"tr": "Sen benim kim olduğumu biliyor musun?",
             "cs": "Potřebujete rychle poradit?",
             "hu": "Az ezredfordulós szilveszter valószínűleg az átlagos év véginél komolyabb feladatokat ró a sürgősségi betegellátás szervezeteire és a rendőrségre.",
             "jp": "為せば成る為さねば成らぬ。\n麩菓子は、麩を主材料とした日本の菓子。",
             "fi": " Showroomilla esiteltiin uusi The Garden Collection ja tarjoiltiin maukasta aamupalaa aamu-unisille muotibloggaajille."}

model_name_dict = {"jp": "cl-tohoku/bert-base-japanese",
                   "tr": "dbmdz/bert-base-turkish-cased",
                   "hu": "SZTAKI-HLT/hubert-base-cc",
                   "fi": "TurkuNLP/bert-base-finnish-cased-v1 ",
                   "cs": "DeepPavlov/bert-base-bg-cs-pl-ru-cased"}

output_file = "tokenized.txt"


def test_tokenizers():
    for lang,model_name in model_name_dict.items():
        sent = sent_dict[lang]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        tokens = tokenizer.tokenize(sent)
        print(model)
        with open(output_file, "a", encoding="utf-8") as o:
            o.write("{}\n".format(lang))
            o.write("Sentence: " + sent)
            o.write("\n")
            o.write("Tokens: " + " ".join(tokens))
            o.write("\n")

def main():
    test_tokenizers()

if __name__ == "__main__":
    main()