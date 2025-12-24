import os
import argparse
from PIL import Image
from utils.jailbreak import getAnswer
from tqdm import tqdm
from utils.steganography import extract, getImages
from utils.eval import get_asr, eval_res
from utils.encode import Base64Expert
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="JailbreakBench", choices=["SafeBench-Tiny","JailbreakBench"], help="Dataset to use.")
    parser.add_argument("--model", type=str, default="doubao-seed-1-6-251015", help="Model to evaluate.")
    parser.add_argument("--try_times", type=int, default=10, help="Number of attempts per question.")
    parser.add_argument("--cover_image", type=str, default="./cat.png", help="Path to the cover image.")
    args = parser.parse_args()

    dataset, model = args.dataset, args.model

    EN = Base64Expert()
    # Prepare images
    getImages(args, EN)

    if not os.path.exists(f"./output/image/{dataset}-{model}"):
        os.makedirs(f"./output/image/{dataset}-{model}")
    if not os.path.exists(f"./output/{dataset}-{model}-results.csv"):
        with open(f"./output/{dataset}-{model}-results.csv", "w",encoding="utf-8") as f:
            f.write("question,answer,decoded_answer,isSuccess\n")
        
    with open(f"./data/{dataset}.txt", "r") as f:
        data = f.readlines()
        for j, line in tqdm(enumerate(data), desc=f"{dataset}"):
            question = line.strip()
            img = str(j)
            tqdm.write(f"Now question is: ({img})" + question)
            k = 0
            while k < args.try_times:
                if os.path.exists(f"./output/image/{dataset}-{model}/{img}.png"):
                    os.remove(f"./output/image/{dataset}-{model}/{img}.png")
                try:
                    # jailbreak attempt
                    getAnswer(img, img, args)
                    res = extract(Image.open(f"./output/image/{dataset}-{model}/{img}.png"))
                    decode_res = res
                except Exception as e:
                    tqdm.write(f"Error occurred: {e}. Retrying...")
                    k += 1
                    continue
                if re.findall(r'\d+\.\s', decode_res):
                    for base64_string in re.findall(r'\d+\.\s', decode_res):
                        decode_res = decode_res.replace(base64_string, "")
                try:
                    decode_res = EN.decode(res)
                except:
                    pass
                tqdm.write(f"The raw result is: {res}")
                tqdm.write(f"The decoded result is: {decode_res}")
                # evaluate the result
                is_jailbreak = eval_res(question, decode_res)
                if is_jailbreak or k == args.try_times - 1:
                    with open(f"./output/{dataset}-{model}-results.csv", "a",encoding="utf-8") as f:
                        k = args.try_times
                        res = res.replace('"', '""')
                        decode_res = decode_res.replace('"', '""')
                        f.write(f'"{question}","{res}","{decode_res}",{is_jailbreak}\n')
                k += 1
    # Calculate average success rate
    avg_res = get_asr(args)
    content = f'''
        The evaluation results are as follows:
        Model: {model}
        Dataset: {dataset}
        Average results: {avg_res}
        '''
    print(content)