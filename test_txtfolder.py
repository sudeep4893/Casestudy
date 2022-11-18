import argparse
import torch
from my_data import MyDataset, VOCAB
from my_models import MyModel0
from my_utils import pred_to_dict
from my_classes import TextBox, TextLine
import json
import os
from os import path


def sort_text(txt_file):
    with open(txt_file, "r") as txt_opened:
        content = sorted([TextBox(line) for line in txt_opened], key=lambda box: box.y)

    text_lines = [TextLine(content[0])]
    for box in content[1:]:
        try:
            text_lines[-1].insert(box)
        except ValueError:
            text_lines.append(TextLine(box))

    return "\n".join([str(text_line) for text_line in text_lines])


def test():
    keys = sorted(
        path.splitext(f.name)[0]
        for f in os.scandir("data/annotated_data")
        if f.name.endswith(".txt")
    )

    #print("inside created_test_data : ", data_location)

    files = ["data/annotated_data/" + s + ".txt" for s in keys]
    #print("files created_test_data : ", files)
    test_dict = {}
    for k, f in zip(keys, files):
        test_dict[k] = sort_text(f)
    #print("Dict created_test_data : ", test_dict)
    torch.save(test_dict, "data/annotated_data/test_dict_txt.pth")

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", default="cpu")
    parser.add_argument("-i", "--hidden-size", type=int, default=256)

    args, unknown = parser.parse_known_args()
    args.device = torch.device(args.device)

    model = MyModel0(len(VOCAB), 16, args.hidden_size).to(args.device)
    dataset = MyDataset(None, args.device, test_path="data/annotated_data/test_dict_txt.pth")

    model.load_state_dict(torch.load("model.pth"))

    json_list = []

    model.eval()
    with torch.no_grad():
        for key in dataset.test_dict.keys():
            text_tensor = dataset.get_test_data(key)

            oupt = model(text_tensor)
            prob = torch.nn.functional.softmax(oupt, dim=2)
            prob, pred = torch.max(prob, dim=2)

            prob = prob.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()

            real_text = dataset.test_dict[key]
            result = pred_to_dict(real_text, pred, prob)

            #print(result)
            #return result
            #with open("data/annotated_data/json_result/" + key + ".json", "w", encoding="utf-8") as json_opened:
            #    json.dump(result, json_opened, indent=4)
            json_list.append(result)
    #print(json_list)
    return json_list

'''
if __name__ == "__main__":
    test()
'''