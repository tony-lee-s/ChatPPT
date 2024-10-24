import os
import argparse
from input_parser import parse_input_text
from ppt_generator import generate_presentation
from session_history import get_session_history
from template_manager import load_template, get_layout_mapping, print_layouts
from layout_manager import LayoutManager
from config import Config
from logger import LOG  # 引入 LOG 模块
import gradio as gr
from langchain_ollama.chat_models import ChatOllama  # 导入 ChatOllama 模型
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage  # 导入人类消息和 AI 消息类
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

# 定义全局变量
chatbot_with_history = None  # 全局变量，用于存储聊天机器人实例
session_id = "123"  # 会话ID

def convert_chat_to_powerpoint(chat_content):
    config = Config()  # 加载配置文件

    # 加载 PowerPoint 模板，并打印模板中的可用布局
    prs = load_template(config.ppt_template)  # 加载模板文件
    LOG.info("可用的幻灯片布局:")  # 记录信息日志，打印可用布局
    print_layouts(prs)  # 打印模板中的布局

    # 初始化 LayoutManager，使用配置文件中的 layout_mapping
    layout_manager = LayoutManager(config.layout_mapping)

    # 调用 parse_input_text 函数，解析输入文本，生成 PowerPoint 数据结构
    powerpoint_data, presentation_title = parse_input_text(chat_content, layout_manager)

    LOG.info(f"解析转换后的 ChatPPT PowerPoint 数据结构:\n{powerpoint_data}")  # 记录调试日志，打印解析后的 PowerPoint 数据

    # 定义输出 PowerPoint 文件的路径
    output_pptx = f"outputs/{presentation_title}.pptx"
    
    # 调用 generate_presentation 函数生成 PowerPoint 演示文稿
    generate_presentation(powerpoint_data, config.ppt_template, output_pptx)

    return output_pptx  # 返回生成的 PPT 文件路径

# 更新生成 PPT 的函数
def generate_ppt():
    global session_id  # 声明使用全局变量
    history = get_session_history(session_id)
    
    messages = history.messages  # 获取所有消息
    if not messages:
        raise ValueError("聊天记录为空，无法生成 PPT。")
    
    last_chat_content = messages[-1].content   # 获取聊天记录的最后一条内容
    ppt_file = convert_chat_to_powerpoint(last_chat_content)  # 调用新的函数
    return ppt_file  # 返回生成的 PPT 文件路径

# 更新 handele_chat 函数以适应新的 generate_ppt 函数
def handele_chat(user_input, chat_history):
    global chatbot_with_history  # 声明使用全局变量
    global session_id  # 声明使用全局变量
    if chatbot_with_history is None:
        raise RuntimeError("聊天机器人尚未初始化，请确保在调用之前创建聊天机器人。")
    
    response = chatbot_with_history.invoke(
            [HumanMessage(content=user_input)],  # 将用户输入封装为 HumanMessage
            {"configurable": {"session_id": session_id}},  # 传入配置，包括会话ID
        )
    
    LOG.debug(response)  # 记录调试日志
    return response.content  # 返回生成的回复内容

with gr.Blocks(title='Chat PPT') as chat_ptt_app:
    with gr.Tab("生成PPT"):
        chatbot = gr.Chatbot(
            placeholder="输入要生成的话题",
            height=600,
        )

# 使用 gr.File 组件来实现下载功能
        ppt_file_output = gr.File(label="生成的 PPT")  # 显示生成的文件路径

        # 添加生成 PPT 按钮
        generate_button = gr.Button("生成 PPT")
        
        # 定义按钮点击事件
        def on_generate_ppt():
            ppt_file = generate_ppt()  # 生成 PPT
            print(f"PPT 已生成: {ppt_file}")
            return ppt_file  # 返回生成的 PPT 文件路径

        generate_button.click(on_generate_ppt, outputs=ppt_file_output)  # 绑定按钮点击事件
        
        gr.ChatInterface(
            fn=handele_chat,  # 处理对话的函数
            chatbot=chatbot,  # 聊天机器人组件
            retry_btn=None,  # 不显示重试按钮
            undo_btn=None,  # 不显示撤销按钮
            clear_btn="清除历史记录",  # 清除历史记录按钮文本
            submit_btn="发送",  # 发送按钮文本
        )


def create_chatbot():
    global chatbot_with_history  # 声明使用全局变量
    prompt_file = "prompts/formatter.txt"  # 聊天提示文件路径

    try:
        with open(prompt_file, "r", encoding="utf-8") as file:
            prompt = file.read().strip()  # 读取文件并去除首尾空格
        
        # 创建聊天提示模板，包括系统提示和消息占位符
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),  # 系统提示部分
            MessagesPlaceholder(variable_name="messages"),  # 消息占位符
        ])

        # 初始化 ChatOllama 模型，配置模型参数
        chatbot = system_prompt | ChatOllama(
            model="llama3.1:8b-instruct-q8_0",  # 使用的模型名称
            max_tokens=8192,  # 最大生成的token数
            temperature=0.8,  # 生成文本的随机性
        )

        # 将聊天机器人与消息历史记录关联起来
        chatbot_with_history = RunnableWithMessageHistory(chatbot, get_session_history)
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到提示文件 {prompt_file}!")
    
    

# 程序入口
if __name__ == "__main__":

    create_chatbot()

    # 密码读取.env文件
    # 获取环境变量
    load_dotenv()
    admin_password = os.getenv("ADMIN_PASSWORD")
    chat_ptt_app.launch(auth=("admin", admin_password))  # 启动 Gradio 应用并设置访问密码
