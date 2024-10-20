import traceback
import os
from tqdm import tqdm
import random
import pickle
import numpy as np
from LMM import GPT4VAPI, GeminiAPI,InternVLAPI,ClaudeAPI,QWENAPI
import ast
from collections import Counter
import pandas as pd

import pandas as pd
import os
import soundfile as sf
import numpy as np

def process_aiff_file(index, prefix,folder_path='./Dataset/RCW',decimal_places=3):
    """
    读取指定的AIFF文件,将其转换为数组,然后返回为指定精度的字符串
    
    参数:
    folder_path (str): 包含AIFF文件的文件夹路径
    index (int): 文件名中的索引号
    decimal_places (int): 保留的小数位数
    
    返回:
    str: 包含数组数据的字符串,每个数字保留指定的小数位数
    """
    # 构造文件名
    file_name = f"{prefix}/{index}.aiff"
    file_path = os.path.join(folder_path, file_name)
    #print(file_path)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 读取AIFF文件
    data, _ = sf.read(file_path)
    
    # 确保数据是一维数组
    if data.ndim > 1:
        data = data.flatten()
    
    # 将数组转换为指定精度的字符串
    #print(data)
    formatted_data = [f"{x:.{decimal_places}f}" for x in data]
    
    # 将格式化后的数据连接成一个字符串
    result_string = ",".join(formatted_data)
    
    return result_string
def df2str(df_value, df_index, decimal_places=1):
    row_value = df_value.iloc[df_index]
    formatted_data = []

    def format_number(num):
        if decimal_places is not None:
            return f"{num:.{decimal_places}f}"
        return str(num)

    if isinstance(row_value, pd.Series):
        for item in row_value:
            if isinstance(item, pd.Series):
                # 如果是多维时间序列，将每个维度转换为逗号分隔的字符串
                formatted_data.append(','.join(map(format_number, item.values)))
            else:
                # 如果是单个值，直接转换为字符串
                formatted_data.append(format_number(item))
    else:
        # 如果row_value只是一个单一的值，直接转换为字符串
        formatted_data = [format_number(row_value)]

    # 返回结果
    if len(formatted_data) == 1:
        return f"{formatted_data[0]}"
    else:
        return ";".join(f"dim_{i}: {data}" for i, data in enumerate(formatted_data))
def work(
    model,
    num_shot_per_class,
    location,
    num_qns_per_round,
    test_df,
    demo_df,
    classes,
    class_desp,
    SAVE_FOLDER,
    dataset_name,
    detail="auto",
    file_suffix="",
    question="What is in the image above",
    prior_knowledge="",
    similar_use=0,
    similar_num=-1,
    image_token="<<IMG>>",
    name="",
    modal="LV",
    demo_value=0,
    test_value=0,
    hint="",
):
    """
    Run queries for each test case in the test_df dataframe using demonstrating examples sampled from demo_df dataframe.

    model[str]: the specific model checkpoint to use e.g. "Gemini1.5", "gpt-4-turbo-2024-04-09"
    num_shot_per_class[int]: number of demonstrating examples to include for each class, so the total number of demo examples equals num_shot_per_class*len(classes)
    location[str]: Vertex AI location e.g. "us-central1","us-west1", not used for GPT-series models
    num_qns_per_round[int]: number of queries to be batched in one API call
    test_df, demo_df [pandas dataframe]: dataframe for test cases and demo cases, see dataset/UCMerced/demo.csv as an example
    classes[list of str]: names of categories for classification, and this should match tbe columns of test_df and demo_df.
    class_desp[list of str]: category descriptions for classification, and these are the actual options sent to the model
    SAVE_FOLDER[str]: path for the images
    dataset_name[str]: name of the dataset used
    detail[str]: resolution level for GPT4(V)-series models, not used for Gemini models
    file_suffix[str]: suffix for image filenames if not included in indexes of test_df and demo_df. e.g. ".png"
    """
    

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    # 原有的 EXP_NAME 定义
    if similar_use==0:
        num_shots=num_shot_per_class*len(classes)
    else:
        num_shots=similar_num
    if similar_use==0:
        EXP_NAME = f"{dataset_name}_{model}model_{modal}modal_{num_shots}randomshot_{num_qns_per_round}preround"

    else:
        EXP_NAME = f"{dataset_name}_{model}model_{modal}modal_{num_shots}similarshot_{num_qns_per_round}preround"

    # 创建文件夹路径
    folder_path = f"./{name}/"

    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)

    # 修改 EXP_NAME 以包含文件夹路径
    EXP_NAME = os.path.join(folder_path, EXP_NAME)

    if model.startswith("gpt"):
        api = GPT4VAPI(model=model, detail=detail,modal=modal)
    elif model.startswith("internvl"):
        api = InternVLAPI()
    elif model.startswith("claude"):
        api = ClaudeAPI()

    elif model.startswith("qwen"):
        api = QWENAPI(model=model, detail=detail,modal=modal)
    else:
        assert model == "Gemini1.5"
        api = GeminiAPI(location=location)
    print(EXP_NAME, f"test size = {len(test_df)}")

    # Prepare the demonstrating examples
    demo_examples = []
    if similar_use==0:
        for class_name in classes:
            num_cases_class = 0
            for j in demo_df[demo_df[class_name] == 1].itertuples():
                if num_cases_class == num_shot_per_class:
                    break
                if modal=="L" or modal=="LV":
                    
                    if dataset_name!="RCW":
                        ts_index=int(j.Index.split("_")[-1].split(".")[0])
                        data_L=df2str(demo_value,ts_index)
                    else:
                        ts_index=j.Index.split(".")[0]
                        data_L=process_aiff_file(ts_index, "train")
                    demo_examples.append((j.Index, class_desp[class_to_idx[class_name]],data_L))
                else:
                    demo_examples.append((j.Index, class_desp[class_to_idx[class_name]]))
                num_cases_class += 1
        assert len(demo_examples) == num_shot_per_class * len(classes)
    else:
        print("Use predifined sampled similar examples")
        num_qns_per_round = 1
        if similar_num==-1:
            similar_num = len(classes)
        
        for _, row in test_df.iterrows():
            demo_examples_temp = []
            similar_samples = ast.literal_eval(row['similar_samples_low_to_high'])
            selected_samples = similar_samples[-similar_num:]  # Select the last similar_num samples
            
            for sample_idx in selected_samples:
                sample_class = demo_df.loc[sample_idx, classes].idxmax()
                if modal=="L" or modal=="LV":
                    
                    if dataset_name!="RCW":
                        ts_index=int(sample_idx.split("_")[-1].split(".")[0])
                        data_L=df2str(demo_value,ts_index)
                    else:
                        ts_index=sample_idx.Index.split(".")[0]
                        data_L=process_aiff_file(ts_index, "train")
                    demo_examples_temp.append((j.Index, class_desp[class_to_idx[class_name]],data_L))
                else:
                    demo_examples_temp.append((sample_idx, class_desp[class_to_idx[sample_class]]))
            
            demo_examples.append(demo_examples_temp)
        
                


    # Load existing results
    if os.path.isfile(f"{EXP_NAME}.pkl"):
        with open(f"{EXP_NAME}.pkl", "rb") as f:
            results = pickle.load(f)
    else:
        results = {}

    test_df = test_df.sample(frac=1, random_state=66)  # Shuffle the test set
    for start_idx in tqdm(range(0, len(test_df), num_qns_per_round), desc=EXP_NAME):
        end_idx = min(len(test_df), start_idx + num_qns_per_round)
        if similar_use==0:
            random.shuffle(demo_examples)
            now_demo_examples = demo_examples
        else:
            now_demo_examples = demo_examples[start_idx]
            #从now_demo_examples = demo_examples[start_idx]统计出频率最高的标签
            #insert code here 
            label_counter = Counter(demo[1] for demo in now_demo_examples)
            most_common_label = label_counter.most_common(1)[0][0]
            print(f"Most common label: {most_common_label}")
        print(f"now_demo_examples: {now_demo_examples}")
        prompt = ""
        if prior_knowledge!="":
            prompt += f"""Task Description: {prior_knowledge}\n"""
        if hint!="":
            prompt += f"""Hint: {hint}\n"""
        # 修改这里的代码以适应新的结构
        image_paths = []
        for demo in now_demo_examples:
            if isinstance(demo, tuple):
                # 如果 demo 是元组，假设第一个元素是图像标识符
                image_path = os.path.join(SAVE_FOLDER, demo[0] + file_suffix)
            elif isinstance(demo, str):
                # 如果 demo 是字符串，直接使用它
                image_path = os.path.join(SAVE_FOLDER, demo + file_suffix)
            else:
                # 如果是其他类型，打印警告并跳过
                print(f"Warning: Unexpected demo type: {type(demo)}")
                continue
            image_paths.append(image_path)
        if modal=="V":
            #数据只用视觉模态
        
            for demo in now_demo_examples:
                prompt += f"""{image_token}Given the image above, answer the following question using the specified format. 
    Question: {question}
    Choices: {str(class_desp)}
    Answer Choice: {demo[1]}
    """
            qns_idx = []
            for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
                qns_idx.append(i.Index)
                image_paths.append(os.path.join(SAVE_FOLDER, i.Index + file_suffix))
                qn_idx = idx + 1

                prompt += f"""{image_token}Given the image above, answer the following question using the specified format. 
    Question {qn_idx}: {question}
    Choices {qn_idx}: {str(class_desp)}

    """
            for i in range(start_idx, end_idx):
                qn_idx = i - start_idx + 1
                prompt += f"""
    Please respond with the following format for each question:
    ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
    Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
    Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
    ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

    Do not deviate from the above format. Repeat the format template for the answer. """
            qns_id = str(qns_idx)
            for retry in range(3):
                    if (
                        (qns_id in results)
                        and (not results[qns_id].startswith("ERROR"))
                    
                        
                    ):
                        # 当结果已经存在且成功时，跳出循环
                        break

                    try:
                        res = api(
                            prompt,
                            image_paths=image_paths,
                            real_call=True,
                            max_tokens=60 * num_qns_per_round,
                        )
                    except Exception as e:
                        res = f"ERROR!!!! {traceback.format_exc()}"
                    except KeyboardInterrupt:
                        previous_usage = results.get("token_usage", (0, 0, 0))
                        total_usage = tuple(
                            a + b for a, b in zip(previous_usage, api.token_usage)
                        )
                        results["token_usage"] = total_usage
                        with open(f"{EXP_NAME}.pkl", "wb") as f:
                            pickle.dump(results, f)
                        exit()

                    print(res)
                    results[qns_id] = res

                    if (
                        (not res.startswith("ERROR"))
                        
                    ):
                        # 当调用成功时，跳出循环
                        break
        if modal=="LV":
            #数据用视觉模态+
        
            for demo in now_demo_examples:
                prompt += f"""{image_token}Given the image above, and the corresponding specific values ​​are as follows: {demo[-1]}. Answer the following question using the specified format. 
    Question: {question}
    Choices: {str(class_desp)}
    Answer Choice: {demo[1]}
    """
            qns_idx = []
            for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
                qns_idx.append(i.Index)
                image_paths.append(os.path.join(SAVE_FOLDER, i.Index + file_suffix))
                qn_idx = idx + 1

                
                if dataset_name!="RCW":
                    ts_index=int(i.Index.split("_")[-1].split(".")[0])
                    data_L=df2str(test_value,ts_index)
                else:
                    ts_index=i.Index.split(".")[0]
                    data_L=process_aiff_file(ts_index, "train")
                #data_L=df2str(test_value,ts_index)
                prompt += f"""{image_token}Given the image above, and the corresponding specific values ​​are as follows: {data_L}. Answer the following question using the specified format. 
    Question {qn_idx}: {question}
    Choices {qn_idx}: {str(class_desp)}

    """
            for i in range(start_idx, end_idx):
                qn_idx = i - start_idx + 1
                prompt += f"""
    Please respond with the following format for each question:
    ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
    Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
    Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
    ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

    Do not deviate from the above format. Repeat the format template for the answer. """
            qns_id = str(qns_idx)
            for retry in range(3):
                if (
                (qns_id in results)
                and (not results[qns_id].startswith("ERROR"))
                 
            ):
                # 当结果已经存在且成功时，跳出循环
                    break

            try:
                res = api(
                    prompt,
                    image_paths=image_paths,
                    real_call=True,
                    max_tokens=60 * num_qns_per_round,
                )
            except Exception as e:
                res = f"ERROR!!!! {traceback.format_exc()}"
            except KeyboardInterrupt:
                previous_usage = results.get("token_usage", (0, 0, 0))
                total_usage = tuple(
                    a + b for a, b in zip(previous_usage, api.token_usage)
                )
                results["token_usage"] = total_usage
                with open(f"{EXP_NAME}.pkl", "wb") as f:
                    pickle.dump(results, f)
                exit()

            print(res)
            results[qns_id] = res

            if (
                (not res.startswith("ERROR"))
                
            ):
                # 当调用成功时，跳出循环
                break
        if modal=="L":
                #数据用数值模态
            
                for demo in now_demo_examples:
                    prompt += f"""Given the corresponding specific numerical series ​​are as follows: {demo[-1]}. Answer the following question using the specified format. 
        Question: {question}
        Choices: {str(class_desp)}
        Answer Choice: {demo[1]}
        """
                qns_idx = []
                for idx, i in enumerate(test_df.iloc[start_idx:end_idx].itertuples()):
                    qns_idx.append(i.Index)
                    image_paths.append(os.path.join(SAVE_FOLDER, i.Index + file_suffix))
                    qn_idx = idx + 1

                    
                    if dataset_name!="RCW":
                        ts_index=int(i.Index.split("_")[-1].split(".")[0])
                        data_L=df2str(test_value,ts_index)
                    else:
                        
                        ts_index=i.Index.split(".")[0]
                        data_L=process_aiff_file(ts_index, "train")
                        #RCW是train划分的
                    prompt += f"""Given the corresponding specific numerical series ​​are as follows: {data_L}. Answer the following question using the specified format. 
        Question {qn_idx}: {question}
        Choices {qn_idx}: {str(class_desp)}

        """
                for i in range(start_idx, end_idx):
                    qn_idx = i - start_idx + 1
                    prompt += f"""
        Please respond with the following format for each question:
        ---BEGIN FORMAT TEMPLATE FOR QUESTION {qn_idx}---
        Answer Choice {qn_idx}: [Your Answer Choice Here for Question {qn_idx}]
        Confidence Score {qn_idx}: [Your Numerical Prediction Confidence Score Here From 0 To 1 for Question {qn_idx}]
        ---END FORMAT TEMPLATE FOR QUESTION {qn_idx}---

        Do not deviate from the above format. Repeat the format template for the answer. """
                qns_id = str(qns_idx)
                for retry in range(3):
                    if (
                        (qns_id in results)
                        and (not results[qns_id].startswith("ERROR"))
                    
                        
                    ):
                        # 当结果已经存在且成功时，跳出循环
                        break

                    try:
                        res = api(
                            prompt,
                            image_paths=image_paths,
                            real_call=True,
                            max_tokens=60 * num_qns_per_round,
                        )
                    except Exception as e:
                        res = f"ERROR!!!! {traceback.format_exc()}"
                    except KeyboardInterrupt:
                        previous_usage = results.get("token_usage", (0, 0, 0))
                        total_usage = tuple(
                            a + b for a, b in zip(previous_usage, api.token_usage)
                        )
                        results["token_usage"] = total_usage
                        with open(f"{EXP_NAME}.pkl", "wb") as f:
                            pickle.dump(results, f)
                        exit()

                    print(res)
                    results[qns_id] = res

                    if (
                        (not res.startswith("ERROR"))
                        
                    ):
                        # 当调用成功时，跳出循环
                        break
    # Update token usage and save the results
    #previous_usage = results.get("token_usage", (0, 0, 0))
    #total_usage = tuple(a + b for a, b in zip(previous_usage, api.token_usage))
    #results["token_usage"] = total_usage
    results["token_usage"] = -1

    #这里是我加的
    pickle_file = f"{EXP_NAME}.pkl"
    if os.path.exists(pickle_file):
        try:
            os.remove(pickle_file)
            print(f"Existing file {pickle_file} has been removed.")
        except OSError as e:
            print(f"Error removing existing file {pickle_file}: {e}")
            # You might want to add additional error handling here if needed

    # Now proceed with dumping the new data
    with open(pickle_file, "wb") as f:
        pickle.dump(results, f)

    print(f"Data successfully saved to {pickle_file}")