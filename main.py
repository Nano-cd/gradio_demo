"""这个部分可以读取单个图像的预测结果"""
import gradio as gr
import pandas as pd
from ultralytics import YOLO
from skimage import data
from PIL import Image

model = YOLO('model/best.pt')


def predict(img):
    result = model.predict(source=img)
    # 替换名称
    name_mapping = {0: '试管', 1: '微量杯', 2: '无样本管'}
    replaced_names = [name_mapping.get(int(name), name) for name in result[0].names]

    df = pd.Series(replaced_names).to_frame()
    df.columns = ['names']
    df['probs'] = result[0].probs.data.cpu().numpy()
    df = df.sort_values('probs', ascending=False)
    res = dict(zip(df['names'], df['probs']))
    # output_text = "\n".join([f"{name}: {prob:.2f}" for name, prob in res.items()])
    # return output_text  # 返回格式化的文本
    return res


gr.close_all()
demo = gr.Interface(fn=predict, inputs=gr.Image(type='pil'), outputs=gr.Label(num_top_classes=3),
                    examples=['examples/250116073019.094-6-4.jpg'])
demo.launch(share=True)


# """这个部分可以读取文件夹内的全部图像"""
# import gradio as gr
# import pandas as pd
# from ultralytics import YOLO
# from PIL import Image
# import os
#
# model = YOLO('model/best.pt')
#
#
# def predict_folder(folder_path):
#     results = {}
#     for file in os.listdir(folder_path):
#         if file.endswith('.jpg') or file.endswith('.png'):
#             img_path = os.path.join(folder_path, file)
#             result = model.predict(source=img_path)
#             df = pd.Series(result[0].names).to_frame()
#             df.columns = ['names']
#             df['probs'] = result[0].probs.data.cpu().numpy()
#             df = df.sort_values('probs', ascending=False)
#             res = dict(zip(df['names'], df['probs']))
#             results[file] = res
#     return results
#
#
# gr.close_all()
# demo = gr.Interface(fn=predict_folder, inputs=gr.Textbox(lines=2, placeholder="输入文件夹路径"),
#                     outputs=gr.JSON(), examples=['examples/'])
# demo.launch()
