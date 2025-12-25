import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from openai import OpenAI, APIError, RateLimitError, Timeout
from datetime import datetime

def clone_github_repo(repo_url, target_path):
    """克隆GitHub仓库到指定路径"""
    if os.path.exists(target_path):
        if os.listdir(target_path):
            print(f"错误: 目标路径 '{target_path}' 非空，请指定空目录或新目录")
            sys.exit(1)
    else:
        os.makedirs(target_path, exist_ok=True)
    
    try:
        print(f"正在克隆仓库: {repo_url} 到 {target_path}")
        subprocess.run(
            ["git", "clone", "--depth=1", repo_url, target_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print("仓库克隆成功")
    except subprocess.CalledProcessError:
        print("错误: Git克隆失败，请检查仓库URL和网络连接")
        sys.exit(1)
    except FileNotFoundError:
        print("错误: 未找到Git命令，请先安装Git")
        sys.exit(1)

def find_python_files(root_dir):
    """递归查找目录中所有.py文件"""
    python_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):  # 排除__init__.py等
                python_files.append(Path(os.path.join(root, file)))
    return python_files

def convert_to_go(client, python_code, retries=3):
    """调用API将Python代码转换为Go代码"""
    # 获取当前日期（根据系统提示）
    current_date = datetime.now().strftime("%A, %B %d, %Y")
    
    system_prompt = (
        f"You are MiMo, an AI assistant developed by Xiaomi. Today is date: {current_date}. "
        "Your knowledge cutoff date is December 2024. You are an expert in programming languages conversion. "
        "Your task is to convert Python code to Go code with high accuracy and idiomatic Go style."
    )
    
    user_prompt = (
        "Convert the following Python code to idiomatic Go code:\n\n"
        "Requirements:\n"
        "1. Preserve the original logic exactly\n"
        "2. Use Go best practices and standard library where appropriate\n"
        "3. Add necessary type declarations and handle Python dynamic features appropriately\n"
        "4. Output ONLY the Go code with no additional text, markdown formatting, or explanations\n\n"
        "Python code:\n```python\n" + python_code + "\n```"
    )
    
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="mimo-v2-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=4096,
                temperature=0.2,
                top_p=0.95,
                stream=False,
                stop=None,
                frequency_penalty=0,
                presence_penalty=0,
                extra_body={
                    "thinking": {"type": "disabled"}
                }
            )
            
            go_code = completion.choices[0].message.content.strip()
            
            # 清理可能的Markdown代码块标记
            if go_code.startswith("```go"):
                go_code = go_code[5:].lstrip()
            if go_code.startswith("```"):
                go_code = go_code[3:].lstrip()
            if go_code.endswith("```"):
                go_code = go_code[:-3].rstrip()
                
            return go_code.strip()
            
        except (RateLimitError, Timeout) as e:
            wait_time = 2 ** attempt
            print(f"API限制/超时 (尝试 {attempt+1}/{retries}): {str(e)}，{wait_time}秒后重试...")
            time.sleep(wait_time)
        except APIError as e:
            print(f"API错误 (尝试 {attempt+1}/{retries}): {str(e)}")
            if attempt == retries - 1:
                raise
            time.sleep(1)
    
    raise Exception("API转换失败，已达到最大重试次数")

def main():
    parser = argparse.ArgumentParser(description='GitHub Python转Go转换工具')
    parser.add_argument('github_url', help='GitHub仓库URL (e.g., https://github.com/user/repo.git)')
    parser.add_argument('target_path', help='本地存储路径 (必须为空目录)')
    parser.add_argument('api_key', help='API Key, 用于访问MiMo API')
    args = parser.parse_args()

    # 验证API密钥
    api_key = args.api_key
    if not api_key:
        print("错误: 未设置MIMO_API_KEY环境变量")
        print("请通过以下方式设置: export MIMO_API_KEY='your_api_key'")
        sys.exit(1)

    # 严格遵循官方例程初始化客户端（注意base_url末尾空格）
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.xiaomimimo.com/v1"  # 保留官方例程中的空格
        )
        # 测试API连接
        completion = client.chat.completions.create(
            model="mimo-v2-flash",
            messages=[
                {
                    "role": "system",
                    "content": "You are MiMo, an AI assistant developed by Xiaomi. Today is date: Tuesday, December 16, 2025. Your knowledge cutoff date is December 2024."
                },
                {
                    "role": "user",
                    "content": "please introduce yourself"
                }
            ],
            max_completion_tokens=1024,
            temperature=0.8,
            top_p=0.95,
            stream=False,
            stop=None,
            frequency_penalty=0,
            presence_penalty=0,
            extra_body={
                "thinking": {"type": "disabled"}
            }
        )
        print(completion.model_dump_json())
    except Exception as e:
        print(f"API初始化失败: {str(e)}")
        print("\n可能的原因及解决方案:")
        print("1. API密钥无效 - 检查MIMO_API_KEY是否正确设置")
        print("2. base_url配置问题 - 官方例程中URL末尾有空格，如果失败可尝试移除空格")
        print("3. 网络连接问题 - 确保能访问api.xiaomimimo.com")
        print("4. 模型名称变更 - 确认'mimo-v2-flash'是当前有效模型")
        sys.exit(1)

    # 克隆仓库
    clone_github_repo(args.github_url, args.target_path)
    
    # 查找Python文件
    py_files = find_python_files(args.target_path)
    if not py_files:
        print("未找到任何Python文件，程序退出")
        sys.exit(0)
    
    print(f"找到 {len(py_files)} 个Python文件，开始转换...")
    
    # 创建转换后代码的保存目录
    converted_dir = Path(args.target_path) / "go_converted"
    converted_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理每个Python文件
    success_count = 0
    failure_details = []
    
    for i, py_file in enumerate(py_files, 1):
        relative_path = py_file.relative_to(args.target_path)
        print(f"\n[{i}/{len(py_files)}] 处理文件: {relative_path}")
        
        try:
            # 读取Python代码
            with open(py_file, 'r', encoding='utf-8') as f:
                python_code = f.read()
            
            if not python_code.strip():
                print(f"  ⚠️ 跳过空文件")
                continue
                
            # 调用API转换
            go_code = convert_to_go(client, python_code)
            
            # 生成Go文件路径 (保持目录结构)
            go_file = converted_dir / relative_path.with_suffix('.go')
            go_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存Go代码
            with open(go_file, 'w', encoding='utf-8') as f:
                f.write(go_code)
                
            print(f"  ✅ 转换成功 → {go_file.relative_to(converted_dir)}")
            success_count += 1
            
            # 遵守API速率限制
            if i < len(py_files):  # 最后一个文件不需要等待
                time.sleep(1.5)  # 保守的请求间隔
                
        except Exception as e:
            error_msg = f"  ❌ 转换失败: {str(e)}"
            print(error_msg)
            failure_details.append(f"{relative_path}: {str(e)}")
            continue
    
    # 生成转换报告
    print("\n" + "="*50)
    print(f"转换完成! 成功: {success_count}/{len(py_files)} 个文件")
    print(f"转换后的Go代码保存在: {converted_dir}")
    
    if failure_details:
        print("\n失败文件详情:")
        for detail in failure_details:
            print(f"- {detail}")
        
        report_path = converted_dir / "conversion_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"转换报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"成功: {success_count}/{len(py_files)}\n")
            f.write("失败文件:\n")
            for detail in failure_details:
                f.write(f"- {detail}\n")
        print(f"\n完整报告已保存至: {report_path}")

if __name__ == "__main__":
    main()