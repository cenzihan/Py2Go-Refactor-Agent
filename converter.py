import os
import sys
import argparse
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- 修改 1: 简化导入，移除引起冲突的 Timeout ---
try:
    # 只导入核心类和基础异常
    from openai import OpenAI, APIError
except ImportError:
    print("错误: 缺少必要的库 'openai'。请运行: pip install openai")
    sys.exit(1)

# ==========================================
# 0. 基础服务 (LLM Service)
# ==========================================
class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str = "mimo-v2-flash"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def call_ai(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, json_mode: bool = False) -> str:
        """通用的 AI 调用方法"""
        retries = 3
        extra_body = {"thinking": {"type": "disabled"}}
        
        if json_mode:
            system_prompt += "\nIMPORTANT: Output strictly in valid JSON format. No markdown."

        for attempt in range(retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    max_completion_tokens=4096,
                    extra_body=extra_body
                )
                content = completion.choices[0].message.content.strip()
                return self._clean_markdown(content)
            
            # --- 修改 2: 移除具体的 Timeout/RateLimitError 捕获 ---
            # 直接捕获 APIError (OpenAI所有错误的基类)
            except APIError as e:
                wait = 2 ** attempt
                print(f"    [LLM警告] API返回错误 (尝试 {attempt+1}/{retries}): {e}")
                print(f"    -> 等待 {wait} 秒后重试...")
                time.sleep(wait)
                
            except Exception as e:
                # 捕获其他网络或系统错误
                print(f"    [LLM错误] 未知异常: {str(e)}")
                if attempt == retries - 1: raise
                time.sleep(1)
                
        return ""

    def _clean_markdown(self, text: str) -> str:
        if text.startswith("```"):
            lines = text.split('\n')
            if len(lines) > 1:
                return '\n'.join(lines[1:-1]).strip()
        return text.strip()

# ==========================================
# 4. 记忆层 (MemoryLayer)
# ==========================================
class MemoryLayer:
    def __init__(self):
        self.records: List[Dict] = []
        self.project_context: Dict = {} 

    def save_context(self, file_path: str, analysis: Dict, strategy: Dict):
        self.records.append({
            "file": file_path,
            "analysis": analysis,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat()
        })

    def get_summary(self):
        return {
            "processed_files": len(self.records),
            "details": self.records
        }


# ==========================================
# 1. 感知层 (PerceptionLayer)
# ==========================================
class PerceptionLayer:
    def __init__(self, llm: LLMService, target_path: Path):
        self.llm = llm
        self.target_path = target_path
        
        self.ANALYSIS_PROMPT = """
        You are a Senior Python Code Analyst. 
        Analyze the provided Python code.
        Output a JSON object with the following fields:
        1. "summary": One sentence describing the core function.
        2. "dependencies": List of external libraries used (e.g., numpy, requests).
        3. "internal_refs": List of potential internal module references.
        4. "complexity": "High", "Medium", or "Low".
        5. "is_script": Boolean (true if it has if __name__ == "__main__", else false).
        """

    def prepare_repo(self, repo_url: str):
        if self.target_path.exists() and any(self.target_path.iterdir()):
             # 注意：实际使用时可能需要更灵活的处理，这里保持原样
             raise FileExistsError(f"目录 {self.target_path} 非空")
        
        print(f"[感知] 正在克隆仓库: {repo_url}")
        subprocess.run(["git", "clone", "--depth=1", repo_url, str(self.target_path)], 
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def scan_files(self) -> List[Path]:
        py_files = []
        for root, _, files in os.walk(self.target_path):
            for file in files:
                if file.endswith(".py"):
                    py_files.append(Path(root) / file)
        return py_files

    def analyze_code(self, code_content: str) -> Dict[str, Any]:
        if not code_content.strip():
            return {"summary": "Empty file", "complexity": "Low"}
            
        response = self.llm.call_ai(
            system_prompt=self.ANALYSIS_PROMPT,
            user_prompt=f"Python Code:\n{code_content[:3000]}",
            json_mode=True
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"summary": "Analysis failed", "raw": response}


# ==========================================
# 2. 决策层 (DecisionLayer)
# ==========================================
class DecisionLayer:
    def __init__(self, llm: LLMService):
        self.llm = llm
        
        self.STRATEGY_PROMPT = """
        You are a Software Architect specializing in Python-to-Go migration.
        Based on the provided Code Analysis, generate a migration strategy.
        Output a JSON object with:
        1. "go_libraries": Suggested Go standard or third-party libraries to replace Python deps.
        2. "risk_assessment": Potential risks (e.g., dynamic typing, reflection usage).
        3. "todo_list": A ordered list of steps for the developer to implement this in Go.
        4. "optimization": One suggestion to improve performance or structure in Go.
        """

    def generate_plan(self, analysis: Dict) -> Dict[str, Any]:
        user_prompt = f"Code Analysis Data: {json.dumps(analysis)}"
        
        response = self.llm.call_ai(
            system_prompt=self.STRATEGY_PROMPT,
            user_prompt=user_prompt,
            json_mode=True
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "go_libraries": ["standard_library"],
                "todo_list": ["Direct translation"], 
                "risk_assessment": "Parse failed"
            }


# ==========================================
# 3. 执行层 (ExecutionLayer)
# ==========================================
class ExecutionLayer:
    def __init__(self, llm: LLMService, source_root: Path):
        self.llm = llm
        self.source_root = source_root.resolve() # 使用绝对路径更安全
        self.output_root = (source_root / "go_converted").resolve()
        
        self.CODER_PROMPT = """
        You are a Senior Go Developer. 
        Convert the Python code to idiomatic Go based on the Architect's Strategy.
        
        Rules:
        1. Use the suggested Go libraries from the strategy.
        2. Follow the Todo List.
        3. Preserve the exact logic.
        4. Add comments explaining complex translations.
        5. OUTPUT ONLY THE GO CODE. No explanation text outside the code block.
        """

    def copy_assets(self):
        """
        资源同步：将非 Python 文件（资源、配置、文档）复制到目标目录，
        保持原有的目录结构。
        """
        print("[执行] 正在同步资源文件 (Assets)...")
        count = 0
        for root, dirs, files in os.walk(self.source_root):
            # 防止递归：跳过 .git 和 输出目录自身
            # 使用绝对路径字符串进行判断更稳健
            root_path = Path(root).resolve()
            if ".git" in str(root_path) or str(self.output_root) in str(root_path):
                continue
                
            for file in files:
                # 只有非py文件才会被复制
                if not file.endswith(".py"):
                    src_file = root_path / file
                    try:
                        # 计算相对路径
                        rel_path = src_file.relative_to(self.source_root)
                        dest_file = self.output_root / rel_path
                        
                        dest_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dest_file)
                        print(f"    -> 复制资源: {rel_path}") # [DEBUG] 打印复制的文件
                        count += 1
                    except Exception as e:
                        print(f"    [警告] 资源 {file} 复制失败: {e}")

        print(f"[执行] 资源同步完成。共复制 {count} 个文件。")

    def execute_conversion(self, code_content: str, analysis: Dict, plan: Dict) -> str:
        context_info = (
            f"--- Analysis ---\nSummary: {analysis.get('summary')}\n"
            f"--- Architect's Plan ---\n"
            f"Libs: {', '.join(plan.get('go_libraries', []))}\n"
            f"Todos: {plan.get('todo_list', [])}\n"
        )
        
        user_input = f"{context_info}\n\n--- Python Code ---\n{code_content}"
        
        return self.llm.call_ai(
            system_prompt=self.CODER_PROMPT,
            user_prompt=user_input,
            temperature=0.2 
        )

    def save_go_code(self, original_file: Path, go_code: str):
        rel_path = original_file.resolve().relative_to(self.source_root)
        dest_file = self.output_root / rel_path.with_suffix('.go')
        
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(go_code)
        return dest_file


# ==========================================
# 5. 主控层 (Orchestrator)
# ==========================================
class Orchestrator:
    def __init__(self, github_url: str, local_path: str, api_key: str):
        self.root_path = Path(local_path)
        
        # 初始化基础服务
        self.llm = LLMService(api_key, "https://api.xiaomimimo.com/v1")
        
        # 初始化各层 Agent
        self.memory = MemoryLayer()
        self.perception = PerceptionLayer(self.llm, self.root_path)
        self.decision = DecisionLayer(self.llm)
        self.execution = ExecutionLayer(self.llm, self.root_path)
        
        self.repo_url = github_url

    def run(self):
        print("🚀 多Agent智能重构系统启动...")
        
        # 1. 环境准备
        try:
            self.perception.prepare_repo(self.repo_url)
        except Exception as e:
            print(f"❌ 初始化失败: {e}")
            return

        # 2. 扫描文件
        py_files = self.perception.scan_files()
        total = len(py_files)
        print(f"📂 发现 {total} 个 Python 文件，准备处理...")
        
        # 3. 循环处理
        for i, py_file in enumerate(py_files, 1):
            rel_name = py_file.relative_to(self.root_path)
            print(f"\n[{i}/{total}] 正在处理: {rel_name}")
            
            try:
                # 读取代码
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                # --- Stage 1: 感知 (Perception) ---
                print("  👁️  [感知] 分析代码意图...")
                analysis = self.perception.analyze_code(content)
                
                # [新增] 打印感知层完整输出
                print(f"    ------------ 感知层输出 (JSON) ------------")
                print(json.dumps(analysis, indent=2, ensure_ascii=False))
                print(f"    ------------------------------------------")

                # --- Stage 2: 决策 (Decision) ---
                print("  🧠 [决策] 生成迁移策略...")
                plan = self.decision.generate_plan(analysis)
                
                # [新增] 打印决策层完整输出
                print(f"    ------------ 决策层输出 (JSON) ------------")
                print(json.dumps(plan, indent=2, ensure_ascii=False))
                print(f"    ------------------------------------------")

                # --- Stage 3: 执行 (Execution) ---
                print("  🔨 [执行] 编写 Go 代码...")
                go_code = self.execution.execute_conversion(content, analysis, plan)
                
                saved_path = self.execution.save_go_code(py_file, go_code)
                print(f"      -> 已保存: {saved_path.name}")

                # --- Stage 4: 记忆 (Memory) ---
                self.memory.save_context(str(rel_name), analysis, plan)
                
                time.sleep(1)

            except Exception as e:
                print(f"  ❌ 处理失败: {str(e)}")

        # 4. 资源处理
        print("\n📦 处理静态资源 (保持原有目录结构)...")
        # 这里的 copy_assets 会把非py文件全部复制过去，确保资源路径一致
        self.execution.copy_assets()
        
        # 5. 生成报告
        print("\n✅ 任务完成！")
        summary = self.memory.get_summary()
        report_path = self.root_path / "go_converted" / "migration_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"详细报告已生成: {report_path}")

def main():
    parser = argparse.ArgumentParser(description='多Agent架构 Python 转 Go 工具')
    parser.add_argument('github_url', help='GitHub仓库URL')
    parser.add_argument('target_path', help='本地存储路径')
    parser.add_argument('--api_key', default=os.environ.get("MIMO_API_KEY"), help='API Key')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("请设置环境变量 MIMO_API_KEY 或使用参数 --api_key")
        sys.exit(1)

    orchestrator = Orchestrator(args.github_url, args.target_path, args.api_key)
    orchestrator.run()

if __name__ == "__main__":
    main()