# 作者：顾涛
# 创建时间：2025/6/29

import os
from sys import prefix

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'LangChainDemo'

# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)


# 生成一些结构化的数据：5个参数
# 1、定义一个模型类
class MedicalBilling(BaseModel):
    patient_id: int  # 患者ID，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_code: str  # 诊断代码，字符串类型
    procedure_code: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2、提供一些样例数据，给到大模型
examples = [
    {
        "example": "Patient ID: 823456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 69203, Total Charge: $300, Insurance Claim Amount: $250"
    },
    {
        "example": "Patient ID: 689012, Patient Name: 王鹏, Diagnosis Code: M54.5, Procedure Code: 89213, Total Charge: $250, Insurance Claim Amount: $220"
    },
    {
        "example": "Patient ID: 345688, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 19214, Total Charge: $500, Insurance Claim Amount: $350"
    },
]

# 3、创建一个提示词模版，用来知道AI造数据
openai_template = PromptTemplate(input_variables=['example'], template='{example}')

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt=openai_template,
    input_variables=['subject', 'extra']
)

# 4、创建一个数据生成器
generator = create_openai_data_generator(
    output_schema=MedicalBilling,  # 指定输出数据的格式
    llm=model,
    prompt=prompt_template
)

# 5、调用生成器
result = generator.generate(
    subject='医疗账单',  # 指定生成数据的主题
    extra='账单的最小total_charge不要低于500',  # 额外的补充
    runs=6
)
print(result)
