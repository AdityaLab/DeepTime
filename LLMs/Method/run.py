import pandas as pd
import argparse
from prompt import work
import pandas as pd
import random
from typing import Dict, List
from sktime.datasets import load_from_tsfile_to_dataframe
if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description="Experiment script.")
    # Adding the arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default="UCMerced",
        help="The dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        #default="Gemini1.5",#default="Gemini1.5",
        default="gpt-4o-mini-2024-07-18",

        help="The model to use",
    )
    parser.add_argument(
        "--location",
        type=str,
        required=False,
        default="us-central1",
        help="The location for the experiment",
    )
    parser.add_argument(
        "--num_shot_per_class",
        type=int,
        default=3,
        required=False,
        help="The number of shots per class",
    )
    parser.add_argument(
        "--num_qns_per_round",
        type=int,
        required=False,
        default=1,
        help="The number of questions asked each time",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=False,
        default="What is in the image above",
        help="The question to ask",
    )

    parser.add_argument(
        "--proir_knowledge",
        type=str,
        required=False,
        default="",
        help="The given prior knowledge",
    )
    parser.add_argument(
        "--hint",
        type=str,
        required=False,
        default="",
        help="The given hint",
    )
    #similar_use=0,
    #similar_num=-1
    parser.add_argument(
        "--similar_use",
        type=int,
        required=False,
        default=0,
        help="The similar use",
    )
    parser.add_argument(
        "--similar_num",
        type=int,
        required=False,
        default=-1,
        help="The similar number",
    )
    #image_token="<<IMG>>" InterVL是<image>
    parser.add_argument(
        "--image_token",
        type=str,
        required=False,
        default="<<IMG>>",
        help="The image token",
    )
    #name
    parser.add_argument(
        "--name",
        type=str,
        required=False,
        default="",
        help="The name",
    )   
    parser.add_argument(
        "--modal",
        type=str,
        required=False,
        default="V",
        help="L: language, V: vision, LV: language and vision",
    ) 
    # Parsing the arguments
    args = parser.parse_args()

    # Using the arguments
    dataset_name = args.dataset
    model = args.model
    location = args.location
    num_shot_per_class = args.num_shot_per_class
    num_qns_per_round = args.num_qns_per_round
    question=args.question
    # Folder to load the images, and this will be prepended to the filename stored in the index column of the dataframe.
    IMAGE_FOLDER = f"./Dataset/{dataset_name}/images"

    # Read the two dataframes for the dataset
    demo_df = pd.read_csv(f"./Dataset/{dataset_name}/demo.csv", index_col=0)
    test_df = pd.read_csv(f"./Dataset/{dataset_name}/test.csv", index_col=0)
    demo_ts=f"./Dataset/{dataset_name}/{dataset_name}_TRAIN.ts"
    test_ts=f"./Dataset/{dataset_name}/{dataset_name}_TEST.ts"
    if dataset_name!="RCW":
        demo_value, labels = load_from_tsfile_to_dataframe(demo_ts, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
        test_value, labels = load_from_tsfile_to_dataframe(test_ts, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
    #这里指定ts的路径
    else:
        demo_value, labels = 0,0
        test_value, labels = 0,0
    classes = list(demo_df.columns)  # classes for classification
    class_desp = classes  # The actual list of options given to the model. If the column names are informative enough, we can just use them.
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}

    work(
        model=model,
        num_shot_per_class=num_shot_per_class,
        location=location,
        num_qns_per_round=num_qns_per_round,
        test_df=test_df,
        demo_df=demo_df,
        classes=classes,
        class_desp=class_desp,
        SAVE_FOLDER=IMAGE_FOLDER,
        dataset_name=dataset_name,
        question=question,
        prior_knowledge=args.proir_knowledge,
        similar_use=args.similar_use,
        similar_num=args.similar_num,
        image_token=args.image_token,
        name=args.name,
        modal=args.modal,
        demo_value=demo_value,
        test_value=test_value,
        hint=args.hint
    )
